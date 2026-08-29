from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from clerk.pipeline import run_pipeline



def test_run_pipeline(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp3"
    input_path.write_text("audio contents")
    output_dir = tmp_path / "output"

    mock_metadata = {"language": "pt", "duration": 120.0}
    mock_srt = "1\n00:00:00,000 --> 00:00:02,000\nMock transcription"

    with patch("clerk.pipeline.extract_audio", return_value=output_dir / "input_normalized.wav") as mock_extract, \
         patch("clerk.pipeline.transcribe_file", return_value=("Mock transcription", mock_srt, mock_metadata)) as mock_transcribe, \
         patch("clerk.pipeline.summarize_transcript", return_value="Mock summary") as mock_summarize:

        tx_path, sum_path, metadata_res = run_pipeline(
            input_path=input_path,
            output_dir=output_dir,
            whisper_model="tiny",
            whisper_device="cpu",
            whisper_compute_type="int8",
            llm_model="LiquidAI/lfm2.5-1.2b-instruct",
            language="pt",
        )

        assert tx_path == output_dir / "input_transcript.srt"
        assert sum_path == output_dir / "input_meeting_points.md"
        assert metadata_res["language"] == "pt"

        mock_extract.assert_called_once_with(input_path, output_dir / "input_normalized.wav")
        mock_transcribe.assert_called_once_with(
            output_dir / "input_normalized.wav",
            model_name="tiny",
            device="cpu",
            compute_type="int8",
            language="pt",
            batch_size=2,
            verbose=False,
        )
        mock_summarize.assert_called_once_with(
            transcript="Mock transcription",
            model_name="LiquidAI/lfm2.5-1.2b-instruct",
            language="pt",
            is_video=False,
            custom_prompt=None,
            custom_consolidation_prompt=None,
        )


        assert (output_dir / "input_transcript.srt").read_text(encoding="utf-8") == mock_srt + "\n"
        assert (output_dir / "input_meeting_points.md").read_text(encoding="utf-8") == "Mock summary\n"

        metadata_content = json.loads((output_dir / "input_metadata.json").read_text(encoding="utf-8"))
        assert metadata_content["language"] == "pt"
        assert metadata_content["duration"] == 120.0
        assert metadata_content["models"]["whisper_model"] == "tiny"



def test_run_pipeline_video_mode(tmp_path: Path) -> None:
    input_path = tmp_path / "presentation.mp4"
    input_path.write_text("video contents")
    output_dir = tmp_path / "output"

    mock_metadata = {"language": "pt", "duration": 60.0}
    mock_srt = "1\n00:00:00,000 --> 00:00:02,000\nVideo text"

    with patch("clerk.pipeline.extract_audio", return_value=output_dir / "presentation_normalized.wav"), \
         patch("clerk.pipeline.transcribe_file", return_value=("Video text", mock_srt, mock_metadata)), \
         patch("clerk.pipeline.summarize_transcript", return_value="Video summary"):

        tx_path, sum_path, metadata_res = run_pipeline(
            input_path=input_path,
            output_dir=output_dir,
            whisper_model="tiny",
            whisper_device="cpu",
            whisper_compute_type="int8",
            llm_model="LiquidAI/lfm2.5-1.2b-instruct",
            language="pt",
            is_video=True,
        )

        assert tx_path == output_dir / "presentation_transcript.srt"
        assert sum_path == output_dir / "presentation_resume.md"
        assert (output_dir / "presentation_resume.md").read_text(encoding="utf-8") == "Video summary\n"


def test_run_pipeline_transcribe_only(tmp_path: Path) -> None:
    input_path = tmp_path / "audio.mp3"
    input_path.write_text("audio contents")
    output_dir = tmp_path / "output"

    mock_metadata = {"language": "pt", "duration": 30.0}
    mock_srt = "1\n00:00:00,000 --> 00:00:02,000\nOnly transcribe test"

    with patch("clerk.pipeline.extract_audio", return_value=output_dir / "audio_normalized.wav"), \
         patch("clerk.pipeline.transcribe_file", return_value=("Only transcribe test", mock_srt, mock_metadata)), \
         patch("clerk.pipeline.summarize_transcript") as mock_summarize:

        tx_path, sum_path, metadata_res = run_pipeline(
            input_path=input_path,
            output_dir=output_dir,
            whisper_model="tiny",
            whisper_device="cpu",
            whisper_compute_type="int8",
            llm_model="LiquidAI/lfm2.5-1.2b-instruct",
            language="pt",
            transcribe_only=True,
        )

        assert tx_path == output_dir / "audio_transcript.srt"
        assert sum_path is None
        mock_summarize.assert_not_called()
        assert (output_dir / "audio_transcript.srt").read_text(encoding="utf-8") == mock_srt + "\n"
        assert metadata_res["models"]["llm_model"] == "skipped"


def test_run_pipeline_summarize_only(tmp_path: Path) -> None:
    srt_path = tmp_path / "sample_transcript.srt"
    srt_path.write_text("1\n00:00:00,000 --> 00:00:02,000\nHello from transcript", encoding="utf-8")
    output_dir = tmp_path / "output"

    with patch("clerk.pipeline.extract_audio") as mock_extract, \
         patch("clerk.pipeline.transcribe_file") as mock_transcribe, \
         patch("clerk.pipeline.summarize_transcript", return_value="Summary from SRT") as mock_summarize:

        tx_path, sum_path, metadata_res = run_pipeline(
            input_path=srt_path,
            output_dir=output_dir,
            whisper_model="tiny",
            whisper_device="cpu",
            whisper_compute_type="int8",
            llm_model="LiquidAI/lfm2.5-1.2b-instruct",
            language="pt",
            summarize_only=True,
        )

        mock_extract.assert_not_called()
        mock_transcribe.assert_not_called()
        mock_summarize.assert_called_once_with(
            transcript="Hello from transcript",
            model_name="LiquidAI/lfm2.5-1.2b-instruct",
            language="pt",
            is_video=False,
            custom_prompt=None,
            custom_consolidation_prompt=None,
        )

        assert tx_path == srt_path
        assert sum_path == output_dir / "sample_meeting_points.md"
        assert (output_dir / "sample_meeting_points.md").read_text(encoding="utf-8") == "Summary from SRT\n"
        assert metadata_res["models"]["whisper_model"] == "skipped"


def test_run_pipeline_summarize_only_media_file(tmp_path: Path) -> None:
    video_path = tmp_path / "arquitetura.mp4"
    video_path.write_bytes(b"\x00\x00\x00\x1cftypisom")  # Binary content
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    srt_path = output_dir / "arquitetura_transcript.srt"
    srt_path.write_text("1\n00:00:00,000 --> 00:00:02,000\nVideo subtitle text", encoding="utf-8")

    with patch("clerk.pipeline.extract_audio") as mock_extract, \
         patch("clerk.pipeline.transcribe_file") as mock_transcribe, \
         patch("clerk.pipeline.summarize_transcript", return_value="Video Summary") as mock_summarize:

        tx_path, sum_path, metadata_res = run_pipeline(
            input_path=video_path,
            output_dir=output_dir,
            whisper_model="tiny",
            whisper_device="cpu",
            whisper_compute_type="int8",
            llm_model="LiquidAI/lfm2.5-1.2b-instruct",
            language="pt",
            summarize_only=True,
        )

        mock_extract.assert_not_called()
        mock_transcribe.assert_not_called()
        mock_summarize.assert_called_once_with(
            transcript="Video subtitle text",
            model_name="LiquidAI/lfm2.5-1.2b-instruct",
            language="pt",
            is_video=False,
            custom_prompt=None,
            custom_consolidation_prompt=None,
        )


        assert tx_path == srt_path
        assert sum_path == output_dir / "arquitetura_meeting_points.md"


def test_run_pipeline_summarize_only_missing_srt(tmp_path: Path) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"binary video data")
    output_dir = tmp_path / "output"

    with pytest.raises(FileNotFoundError, match="Run transcription first"):
        run_pipeline(
            input_path=video_path,
            output_dir=output_dir,
            whisper_model="tiny",
            whisper_device="cpu",
            whisper_compute_type="int8",
            llm_model="LiquidAI/lfm2.5-1.2b-instruct",
            language="pt",
            summarize_only=True,
        )


def test_run_pipeline_saves_transcript_ahead_of_summarization_failure(tmp_path: Path) -> None:
    input_path = tmp_path / "crash_test.mp3"
    input_path.write_text("audio contents")
    output_dir = tmp_path / "output"

    mock_metadata = {"language": "pt", "duration": 45.0}
    mock_srt = "1\n00:00:00,000 --> 00:00:02,000\nAhead transcript"

    with patch("clerk.pipeline.extract_audio", return_value=output_dir / "crash_test_normalized.wav"), \
         patch("clerk.pipeline.transcribe_file", return_value=("Ahead transcript", mock_srt, mock_metadata)), \
         patch("clerk.pipeline.summarize_transcript", side_effect=RuntimeError("Ollama crashed")):

        with pytest.raises(RuntimeError, match="Ollama crashed"):
            run_pipeline(
                input_path=input_path,
                output_dir=output_dir,
                whisper_model="tiny",
                whisper_device="cpu",
                whisper_compute_type="int8",
                llm_model="LiquidAI/lfm2.5-1.2b-instruct",
                language="pt",
            )

        # Transcript and interim metadata must have been written ahead of the crash
        assert (output_dir / "crash_test_transcript.srt").exists()
        assert (output_dir / "crash_test_transcript.srt").read_text(encoding="utf-8") == mock_srt + "\n"
        assert (output_dir / "crash_test_metadata.json").exists()


def test_run_pipeline_resume_with_existing_transcript(tmp_path: Path) -> None:
    input_path = tmp_path / "resume_test.mp3"
    input_path.write_text("audio contents")
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    srt_path = output_dir / "resume_test_transcript.srt"
    srt_path.write_text("1\n00:00:00,000 --> 00:00:02,000\nExisting transcript line", encoding="utf-8")

    with patch("clerk.pipeline.extract_audio") as mock_extract, \
         patch("clerk.pipeline.transcribe_file") as mock_transcribe, \
         patch("clerk.pipeline.summarize_transcript", return_value="Resumed summary") as mock_summarize:

        tx_path, sum_path, _ = run_pipeline(
            input_path=input_path,
            output_dir=output_dir,
            whisper_model="tiny",
            whisper_device="cpu",
            whisper_compute_type="int8",
            llm_model="LiquidAI/lfm2.5-1.2b-instruct",
            language="pt",
            resume=True,
        )

        mock_extract.assert_not_called()
        mock_transcribe.assert_not_called()
        mock_summarize.assert_called_once_with(
            transcript="Existing transcript line",
            model_name="LiquidAI/lfm2.5-1.2b-instruct",
            language="pt",
            is_video=False,
            custom_prompt=None,
            custom_consolidation_prompt=None,
        )
        assert tx_path == srt_path
        assert sum_path == output_dir / "resume_test_meeting_points.md"
        assert (output_dir / "resume_test_meeting_points.md").read_text(encoding="utf-8") == "Resumed summary\n"


def test_run_pipeline_resume_with_existing_audio(tmp_path: Path) -> None:
    input_path = tmp_path / "audio_resume.mp3"
    input_path.write_text("audio contents")
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_path = output_dir / "audio_resume_normalized.wav"
    wav_path.write_bytes(b"RIFF dummy wav data")

    mock_metadata = {"language": "pt", "duration": 50.0}
    mock_srt = "1\n00:00:00,000 --> 00:00:02,000\nFrom existing wav"

    with patch("clerk.pipeline.extract_audio") as mock_extract, \
         patch("clerk.pipeline.transcribe_file", return_value=("From existing wav", mock_srt, mock_metadata)) as mock_transcribe, \
         patch("clerk.pipeline.summarize_transcript", return_value="Audio resumed summary"):

        tx_path, sum_path, _ = run_pipeline(
            input_path=input_path,
            output_dir=output_dir,
            whisper_model="tiny",
            whisper_device="cpu",
            whisper_compute_type="int8",
            llm_model="LiquidAI/lfm2.5-1.2b-instruct",
            language="pt",
            resume=True,
        )

        mock_extract.assert_not_called()
        mock_transcribe.assert_called_once_with(
            wav_path,
            model_name="tiny",
            device="cpu",
            compute_type="int8",
            language="pt",
            batch_size=2,
            verbose=False,
        )
        assert tx_path == output_dir / "audio_resume_transcript.srt"
        assert sum_path == output_dir / "audio_resume_meeting_points.md"


def test_run_pipeline_resume_with_existing_summary(tmp_path: Path) -> None:
    input_path = tmp_path / "done_test.mp3"
    input_path.write_text("audio contents")
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    sum_path = output_dir / "done_test_meeting_points.md"
    sum_path.write_text("Already done summary", encoding="utf-8")

    with patch("clerk.pipeline.extract_audio") as mock_extract, \
         patch("clerk.pipeline.transcribe_file") as mock_transcribe, \
         patch("clerk.pipeline.summarize_transcript") as mock_summarize:

        tx_path, res_sum_path, _ = run_pipeline(
            input_path=input_path,
            output_dir=output_dir,
            whisper_model="tiny",
            whisper_device="cpu",
            whisper_compute_type="int8",
            llm_model="LiquidAI/lfm2.5-1.2b-instruct",
            language="pt",
            resume=True,
        )

        mock_extract.assert_not_called()
        mock_transcribe.assert_not_called()
        mock_summarize.assert_not_called()
        assert res_sum_path == sum_path


def test_run_pipeline_auto_language_alignment(tmp_path: Path) -> None:
    input_path = tmp_path / "english_talk.mp3"
    input_path.write_text("english audio")
    output_dir = tmp_path / "output"

    # Whisper detects English
    mock_metadata = {"language": "en", "language_probability": 0.99, "duration": 40.0}
    mock_srt = "1\n00:00:00,000 --> 00:00:02,000\nHello English talk"

    with patch("clerk.pipeline.extract_audio", return_value=output_dir / "english_talk_normalized.wav"), \
         patch("clerk.pipeline.transcribe_file", return_value=("Hello English talk", mock_srt, mock_metadata)), \
         patch("clerk.pipeline.summarize_transcript", return_value="English summary") as mock_summarize:

        tx_path, sum_path, metadata_res = run_pipeline(
            input_path=input_path,
            output_dir=output_dir,
            whisper_model="tiny",
            whisper_device="cpu",
            whisper_compute_type="int8",
            llm_model="LiquidAI/lfm2.5-1.2b-instruct",
            language="auto",
        )

        # Summarizer must have been called with detected language 'en'
        mock_summarize.assert_called_once_with(
            transcript="Hello English talk",
            model_name="LiquidAI/lfm2.5-1.2b-instruct",
            language="en",
            is_video=False,
            custom_prompt=None,
            custom_consolidation_prompt=None,
        )
        assert metadata_res["language"] == "en"



