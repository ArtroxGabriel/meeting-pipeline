from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
from typer.testing import CliRunner

from clerk.cli import app, format_time_hhmmssmm


import re

runner = CliRunner()


def _clean_output(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text)


def test_format_time_hhmmssmm() -> None:
    assert format_time_hhmmssmm(0.0) == "00:00:00:00"
    assert format_time_hhmmssmm(10.5) == "00:00:10:50"
    assert format_time_hhmmssmm(120.0) == "00:02:00:00"
    assert format_time_hhmmssmm(3661.12) == "01:01:01:12"


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"], env={"NO_COLOR": "1", "TERM": "dumb"})
    assert result.exit_code == 0
    clean_out = _clean_output(result.output)
    assert "--target" in clean_out


def test_cli_missing_argument() -> None:
    result = runner.invoke(app, [])
    assert result.exit_code != 0


def test_cli_nonexistent_file() -> None:
    result = runner.invoke(app, ["--target", "nonexistent_file.mp3"])
    assert result.exit_code != 0
    assert "Input file does not exist" in result.output


def test_cli_success_local_file(tmp_path: Path) -> None:
    input_file = tmp_path / "sample.mp3"
    input_file.write_text("mock audio content")
    output_dir = tmp_path / "output_dir"

    mock_res_meta = {
        "language": "pt",
        "language_probability": 0.99,
        "duration": 120.0,
        "duration_after_vad": 115.0,
        "timings": {"total_seconds": 10.5, "audio_extraction_seconds": 1.0, "transcription_seconds": 5.0, "summarization_seconds": 4.5},
        "models": {"whisper_model": "tiny", "whisper_device": "cpu", "whisper_compute_type": "int8", "whisper_batch_size": 2, "llm_model": "LiquidAI/lfm2.5-1.2b-instruct"},
        "word_counts": {"transcript_words": 150, "summary_words": 50},
    }

    with patch("clerk.cli.run_pipeline", return_value=(Path("out/transcript.srt"), Path("out/meeting_points.md"), mock_res_meta)) as mock_run:
        result = runner.invoke(
            app,
            [
                "--target",
                str(input_file),
                "--output-dir",
                str(output_dir),
                "--whisper-model",
                "tiny",
                "--llm-model",
                "LiquidAI/lfm2.5-1.2b-instruct",
                "--language",
                "pt",
                "--verbose"
            ]
        )
        assert result.exit_code == 0
        clean_out = _clean_output(result.output)
        assert "Pipeline Status" in clean_out
        assert "Transcript (SRT) : out/transcript.srt" in clean_out
        assert "Summary          : out/meeting_points.md" in clean_out
        mock_run.assert_called_once_with(
            input_path=input_file,
            output_dir=output_dir,
            whisper_model="tiny",
            whisper_device="cpu",
            whisper_compute_type="int8",
            llm_model="LiquidAI/lfm2.5-1.2b-instruct",
            language="pt",
            whisper_batch_size=2,
            is_video=False,
            verbose=True,
            transcribe_only=False,
            summarize_only=False,
            custom_prompt=None,
            custom_consolidation_prompt=None,
            resume=False,
        )


def test_cli_success_youtube_url(tmp_path: Path) -> None:
    mock_temp_file = tmp_path / "yt_download.wav"
    mock_temp_file.write_text("yt audio")
    yt_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    mock_res_meta = {"timings": {}, "models": {}, "word_counts": {}}

    with patch("clerk.cli.download_youtube_audio", return_value=mock_temp_file) as mock_dl, \
         patch("clerk.cli.run_pipeline", return_value=(Path("out/transcript.srt"), Path("out/meeting_points.md"), mock_res_meta)):
        result = runner.invoke(app, ["--target", yt_url])
        assert result.exit_code == 0
        mock_dl.assert_called_once_with(yt_url)
        # Temp file should be deleted in finally block
        assert not mock_temp_file.exists()


def test_cli_keyboard_interrupt(tmp_path: Path) -> None:
    input_file = tmp_path / "sample.mp3"
    input_file.write_text("mock audio content")

    with patch("clerk.cli.run_pipeline", side_effect=KeyboardInterrupt()):
        result = runner.invoke(app, ["--target", str(input_file)])
        assert result.exit_code == 130
        assert "Process interrupted by user" in result.output


def test_cli_gpu_flag(tmp_path: Path) -> None:
    input_file = tmp_path / "sample.mp3"
    input_file.write_text("mock audio content")
    mock_res_meta = {"timings": {}, "models": {}, "word_counts": {}}

    with patch("clerk.cli.is_gpu_available", return_value=True), \
         patch("clerk.cli.run_pipeline", return_value=(Path("out/transcript.srt"), Path("out/meeting_points.md"), mock_res_meta)) as mock_run:
        result = runner.invoke(app, ["--target", str(input_file), "--gpu"])
        assert result.exit_code == 0
        mock_run.assert_called_once_with(
            input_path=input_file,
            output_dir=Path("output"),
            whisper_model="large-v3",
            whisper_device="cuda",
            whisper_compute_type="float16",
            llm_model="llama3.1:8b",
            language="pt",
            whisper_batch_size=8,
            is_video=False,
            verbose=False,
            transcribe_only=False,
            summarize_only=False,
            custom_prompt=None,
            custom_consolidation_prompt=None,
            resume=False,
        )


def test_cli_preset_override(tmp_path: Path) -> None:
    input_file = tmp_path / "sample.mp3"
    input_file.write_text("mock audio content")
    mock_res_meta = {"timings": {}, "models": {}, "word_counts": {}}

    with patch("clerk.cli.is_gpu_available", return_value=True), \
         patch("clerk.cli.run_pipeline", return_value=(Path("out/transcript.srt"), Path("out/meeting_points.md"), mock_res_meta)) as mock_run:
        result = runner.invoke(app, ["--target", str(input_file), "--preset", "gpu", "--whisper-model", "large-v3", "--whisper-batch-size", "4"])
        assert result.exit_code == 0
        mock_run.assert_called_once_with(
            input_path=input_file,
            output_dir=Path("output"),
            whisper_model="large-v3",
            whisper_device="cuda",
            whisper_compute_type="float16",
            llm_model="llama3.1:8b",
            language="pt",
            whisper_batch_size=4,
            is_video=False,
            verbose=False,
            transcribe_only=False,
            summarize_only=False,
            custom_prompt=None,
            custom_consolidation_prompt=None,
            resume=False,
        )




def test_cli_invalid_preset(tmp_path: Path) -> None:
    input_file = tmp_path / "sample.mp3"
    input_file.write_text("mock audio content")

    result = runner.invoke(app, ["--target", str(input_file), "--preset", "invalid_name"])
    assert result.exit_code == 1
    assert "Unknown preset 'invalid_name'" in result.output


def test_cli_gpu_disabled_in_cpu_mode(tmp_path: Path) -> None:
    input_file = tmp_path / "sample.mp3"
    input_file.write_text("mock audio content")

    with patch("clerk.cli.is_gpu_available", return_value=False):
        # 1. Test --gpu flag when GPU is disabled
        result_gpu = runner.invoke(app, ["--target", str(input_file), "--gpu"])
        assert result_gpu.exit_code == 1
        assert "GPU execution is disabled or unavailable in CPU mode" in result_gpu.output

        # 2. Test --preset gpu when GPU is disabled
        result_preset = runner.invoke(app, ["--target", str(input_file), "--preset", "gpu"])
        assert result_preset.exit_code == 1
        assert "requires GPU execution, which is disabled or unavailable in CPU mode" in result_preset.output

        # 3. Test --whisper-device cuda when GPU is disabled
        result_device = runner.invoke(app, ["--target", str(input_file), "--whisper-device", "cuda"])
        assert result_device.exit_code == 1
        assert "--whisper-device cuda' was specified, but GPU execution is disabled or unavailable in CPU mode" in result_device.output


def test_cli_meeting_flag(tmp_path: Path) -> None:
    input_file = tmp_path / "sample.mp3"
    input_file.write_text("mock audio content")
    mock_res_meta = {"timings": {}, "models": {}, "word_counts": {}}

    with patch("clerk.cli.run_pipeline", return_value=(Path("out/sample_transcript.srt"), Path("out/sample_meeting_points.md"), mock_res_meta)) as mock_run:
        result = runner.invoke(app, ["--target", str(input_file), "--meeting"])
        assert result.exit_code == 0
        mock_run.assert_called_once()
        assert mock_run.call_args[1]["is_video"] is False


def test_cli_video_and_meeting_mutually_exclusive(tmp_path: Path) -> None:
    input_file = tmp_path / "sample.mp3"
    input_file.write_text("mock audio content")

    result = runner.invoke(app, ["--target", str(input_file), "--video", "--meeting"])
    assert result.exit_code == 1
    assert "Cannot specify both --video and --meeting options simultaneously" in result.output


def test_cli_gpu_and_fast_mutually_exclusive(tmp_path: Path) -> None:
    input_file = tmp_path / "sample.mp3"
    input_file.write_text("mock audio content")

    result = runner.invoke(app, ["--target", str(input_file), "--gpu", "--fast"])
    assert result.exit_code == 1
    assert "Cannot specify both --gpu and --fast options simultaneously" in result.output


def test_cli_invalid_batch_size(tmp_path: Path) -> None:
    input_file = tmp_path / "sample.mp3"
    input_file.write_text("mock audio content")

    result = runner.invoke(app, ["--target", str(input_file), "--whisper-batch-size", "0"])
    assert result.exit_code == 1
    assert "--whisper-batch-size must be a positive integer" in result.output


def test_cli_invalid_compute_type(tmp_path: Path) -> None:
    input_file = tmp_path / "sample.mp3"
    input_file.write_text("mock audio content")

    result = runner.invoke(app, ["--target", str(input_file), "--whisper-compute-type", "invalid_type"])
    assert result.exit_code == 1
    assert "Invalid --whisper-compute-type 'invalid_type'" in result.output


def test_cli_invalid_device(tmp_path: Path) -> None:
    input_file = tmp_path / "sample.mp3"
    input_file.write_text("mock audio content")

    result = runner.invoke(app, ["--target", str(input_file), "--whisper-device", "invalid_dev"])
    assert result.exit_code == 1
    assert "Invalid --whisper-device 'invalid_dev'" in result.output


def test_cli_upfront_option_validation_before_download() -> None:
    # Option error must be caught BEFORE attempting YouTube download
    yt_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    with patch("clerk.cli.download_youtube_audio") as mock_dl:
        result = runner.invoke(app, ["--target", yt_url, "--whisper-batch-size", "0"])
        assert result.exit_code == 1
        assert "--whisper-batch-size must be a positive integer" in result.output
        mock_dl.assert_not_called()


def test_cli_cpu_float16_not_supported(tmp_path: Path) -> None:
    input_file = tmp_path / "sample.mp3"
    input_file.write_text("mock audio content")

    result = runner.invoke(app, ["--target", str(input_file), "-p", "cpu", "--whisper-compute-type", "float16"])
    assert result.exit_code == 1
    assert "requires GPU (cuda)" in result.output


def test_cli_transcribe_only_and_summarize_only_mutually_exclusive(tmp_path: Path) -> None:
    input_file = tmp_path / "sample.mp3"
    input_file.write_text("mock audio content")

    result = runner.invoke(app, ["--target", str(input_file), "--transcribe-only", "--summarize-only"])
    assert result.exit_code == 1
    assert "Cannot specify both --transcribe-only and --summarize-only options simultaneously" in result.output


def test_cli_transcribe_only_flag(tmp_path: Path) -> None:
    input_file = tmp_path / "sample.mp3"
    input_file.write_text("mock audio content")
    mock_res_meta = {"timings": {}, "models": {"llm_model": "skipped"}, "word_counts": {}}

    with patch("clerk.cli.run_pipeline", return_value=(Path("out/sample_transcript.srt"), None, mock_res_meta)) as mock_run:
        result = runner.invoke(app, ["--target", str(input_file), "--transcribe-only"])
        assert result.exit_code == 0
        clean_out = _clean_output(result.output)
        assert "Summary          : N/A (skipped)" in clean_out
        assert mock_run.call_args[1]["transcribe_only"] is True
        assert mock_run.call_args[1]["summarize_only"] is False


def test_cli_summarize_only_flag(tmp_path: Path) -> None:
    srt_file = tmp_path / "sample_transcript.srt"
    srt_file.write_text("1\n00:00:00,000 --> 00:00:02,000\nMock srt")
    mock_res_meta = {"timings": {}, "models": {"whisper_model": "skipped"}, "word_counts": {}}

    with patch("clerk.cli.run_pipeline", return_value=(srt_file, Path("out/sample_meeting_points.md"), mock_res_meta)) as mock_run:
        result = runner.invoke(app, ["--target", str(srt_file), "--summarize-only"])
        assert result.exit_code == 0
        clean_out = _clean_output(result.output)
        assert "Whisper          : Skipped" in clean_out
        assert mock_run.call_args[1]["transcribe_only"] is False
        assert mock_run.call_args[1]["summarize_only"] is True


def test_cli_srt_target_auto_summarize_only(tmp_path: Path) -> None:
    srt_file = tmp_path / "meeting_transcript.srt"
    srt_file.write_text("1\n00:00:00,000 --> 00:00:02,000\nMock srt content")
    mock_res_meta = {"timings": {}, "models": {"whisper_model": "skipped"}, "word_counts": {}}

    with patch("clerk.cli.run_pipeline", return_value=(srt_file, Path("out/meeting_points.md"), mock_res_meta)) as mock_run:
        result = runner.invoke(app, ["--target", str(srt_file)])
        assert result.exit_code == 0
        assert mock_run.call_args[1]["summarize_only"] is True


def test_prompt_pipeline_recovery_interactive() -> None:
    from clerk.cli import _prompt_pipeline_recovery

    with patch("sys.stdin.isatty", return_value=False):
        llm, whisper, compute, dev, retry = _prompt_pipeline_recovery(
            RuntimeError("test error"), "llm1", "whisper1", "int8", "cpu"
        )
        assert retry is False
        assert llm == "llm1"

def test_cli_custom_prompt_flags(tmp_path: Path) -> None:
    input_file = tmp_path / "sample.mp3"
    input_file.write_text("mock audio content")
    prompt_file = tmp_path / "custom_p.txt"
    prompt_file.write_text("My custom prompt template")
    mock_res_meta = {"timings": {}, "models": {}, "word_counts": {}}

    with patch("clerk.cli.run_pipeline", return_value=(Path("out/transcript.srt"), Path("out/meeting_points.md"), mock_res_meta)) as mock_run:
        result = runner.invoke(app, ["--target", str(input_file), "--prompt-file", str(prompt_file)])
        assert result.exit_code == 0
        assert mock_run.call_args[1]["custom_prompt"] == "My custom prompt template"


def test_cli_prompt_and_prompt_file_mutually_exclusive(tmp_path: Path) -> None:
    input_file = tmp_path / "sample.mp3"
    input_file.write_text("mock audio content")
    prompt_file = tmp_path / "custom_p.txt"
    prompt_file.write_text("My custom prompt template")

    result = runner.invoke(
        app,
        ["--target", str(input_file), "--prompt", "inline prompt", "--prompt-file", str(prompt_file)],
    )
    assert result.exit_code == 1
    assert "Cannot specify both --prompt and --prompt-file options simultaneously" in result.output


def test_cli_resume_flag(tmp_path: Path) -> None:
    input_file = tmp_path / "sample.mp3"
    input_file.write_text("mock audio content")
    mock_res_meta = {"timings": {}, "models": {}, "word_counts": {}}

    with patch("clerk.cli.run_pipeline", return_value=(Path("out/transcript.srt"), Path("out/meeting_points.md"), mock_res_meta)) as mock_run:
        result = runner.invoke(app, ["--target", str(input_file), "--resume"])
        assert result.exit_code == 0
        assert mock_run.call_args[1]["resume"] is True


def test_cli_force_flag(tmp_path: Path) -> None:
    input_file = tmp_path / "sample.mp3"
    input_file.write_text("mock audio content")
    mock_res_meta = {"timings": {}, "models": {}, "word_counts": {}}

    with patch("clerk.cli.run_pipeline", return_value=(Path("out/transcript.srt"), Path("out/meeting_points.md"), mock_res_meta)) as mock_run:
        result = runner.invoke(app, ["--target", str(input_file), "-f"])
        assert result.exit_code == 0
        assert mock_run.call_args[1]["resume"] is False


def test_cli_resume_and_force_mutually_exclusive(tmp_path: Path) -> None:
    input_file = tmp_path / "sample.mp3"
    input_file.write_text("mock audio content")

    result = runner.invoke(app, ["--target", str(input_file), "--resume", "--force"])
    assert result.exit_code == 1
    assert "Cannot specify both --resume and --force simultaneously" in result.output


def test_cli_verbose_configuration_output(tmp_path: Path) -> None:
    input_file = tmp_path / "sample.mp3"
    input_file.write_text("mock audio content")
    mock_res_meta = {"timings": {}, "models": {}, "word_counts": {}}

    with patch("clerk.cli.run_pipeline", return_value=(Path("out/transcript.srt"), Path("out/meeting_points.md"), mock_res_meta)):
        result = runner.invoke(app, ["--target", str(input_file), "--verbose", "--resume"])
        assert result.exit_code == 0
        clean_out = _clean_output(result.output)
        assert "--- Configuration ---" in clean_out
        assert "Target:" in clean_out
        assert "Resume: True" in clean_out
        assert "Force: False" in clean_out


def test_cli_recovery_retries_with_resume(tmp_path: Path) -> None:
    input_file = tmp_path / "sample.mp3"
    input_file.write_text("mock audio content")
    mock_res_meta = {"timings": {}, "models": {}, "word_counts": {}}

    side_effects = [RuntimeError("summarization failed"), (Path("out/transcript.srt"), Path("out/meeting_points.md"), mock_res_meta)]
    with patch("clerk.cli.run_pipeline", side_effect=side_effects) as mock_run, \
         patch("clerk.cli._prompt_pipeline_recovery", return_value=("new-llm", "tiny", "int8", "cpu", True)):
        result = runner.invoke(app, ["--target", str(input_file)])
        assert result.exit_code == 0
        assert mock_run.call_count == 2
        # First attempt: resume=False
        assert mock_run.call_args_list[0][1]["resume"] is False
        # Retry attempt: resume=True
        assert mock_run.call_args_list[1][1]["resume"] is True






