import logging
import os
from pathlib import Path
import time

import typer

from .audio import download_youtube_audio
from .pipeline import run_pipeline

app = typer.Typer(add_completion=False)
logger = logging.getLogger(__name__)

PRESETS: dict[str, dict[str, str | int]] = {
    "cpu": {
        "whisper_model": "large-v3",
        "whisper_device": "cpu",
        "whisper_compute_type": "int8",
        "whisper_batch_size": 2,
        "llm_model": "LiquidAI/lfm2.5-1.2b-instruct",
    },
    "fast": {
        "whisper_model": "small",
        "whisper_device": "cpu",
        "whisper_compute_type": "int8",
        "whisper_batch_size": 2,
        "llm_model": "LiquidAI/lfm2.5-1.2b-instruct",
    },
    "gpu": {
        "whisper_model": "large-v3",
        "whisper_device": "cuda",
        "whisper_compute_type": "float16",
        "whisper_batch_size": 8,
        "llm_model": "llama3.1:8b",
    },
    "accurate": {
        "whisper_model": "large-v3",
        "whisper_device": "cuda",
        "whisper_compute_type": "float16",
        "whisper_batch_size": 4,
        "llm_model": "llama3.1:8b",
    },
}
PRESETS["cuda"] = PRESETS["gpu"]

EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_INTERRUPTED = 130

GPU_PRESETS = {"gpu", "cuda", "accurate"}
VALID_COMPUTE_TYPES = {"int8", "int8_float16", "int8_bfloat16", "int8_float32", "float16", "bfloat16", "float32", "default"}
VALID_DEVICES = {"cpu", "cuda", "gpu", "auto"}


def is_gpu_available() -> bool:
    env_gpu = os.environ.get("ENABLE_GPU", "").strip().lower()
    if env_gpu in ("false", "0", "no", "off", "disable", "disabled"):
        return False
    if env_gpu in ("true", "1", "yes", "on", "enable", "enabled"):
        return True

    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    if verbose:
        logging.getLogger("faster_whisper").setLevel(logging.DEBUG)


def format_time_hhmmssmm(seconds: float) -> str:
    total_cs = round(seconds * 100)
    hours, rem = divmod(total_cs, 360_000)
    mins, rem = divmod(rem, 6000)
    secs, cs = divmod(rem, 100)
    return f"{hours:02d}:{mins:02d}:{secs:02d}:{cs:02d}"


def print_pipeline_status(
    transcript_path: Path | None,
    summary_path: Path | None,
    output_dir: Path,
    metadata: dict,
    verbose: bool,
) -> None:
    timings = metadata.get("timings", {})
    models = metadata.get("models", {})
    word_counts = metadata.get("word_counts", {})
    output_files = metadata.get("output_files", {})
    metadata_file = output_files.get("metadata_path", str(output_dir / "transcript_metadata.json"))

    transcript_display = str(transcript_path) if transcript_path else "N/A (skipped)"
    summary_display = str(summary_path) if summary_path else "N/A (skipped)"

    typer.echo("\n==================================================")
    typer.echo("                 Pipeline Status                  ")
    typer.echo("==================================================")
    typer.echo("📁 Output Paths:")
    typer.echo(f"  • Transcript (SRT) : {transcript_display}")
    typer.echo(f"  • Summary          : {summary_display}")
    typer.echo(f"  • Metadata JSON    : {metadata_file}")

    typer.echo("\n🤖 Models Used:")
    whisper_model = models.get("whisper_model")
    if whisper_model == "skipped":
        typer.echo("  • Whisper          : Skipped")
    else:
        typer.echo(
            f"  • Whisper          : {whisper_model} "
            f"(device: {models.get('whisper_device')}, compute: {models.get('whisper_compute_type')}, batch_size: {models.get('whisper_batch_size')})"
        )
    llm_model = models.get("llm_model")
    if llm_model == "skipped":
        typer.echo("  • LLM              : Skipped")
    else:
        typer.echo(f"  • LLM              : {llm_model}")

    typer.echo("\n⏱️ Execution Time:")
    typer.echo(f"  • Total Time       : {format_time_hhmmssmm(float(timings.get('total_seconds', 0.0)))}")
    if verbose:
        typer.echo(f"    - Audio Extract  : {format_time_hhmmssmm(float(timings.get('audio_extraction_seconds', 0.0)))}")
        typer.echo(f"    - Transcription  : {format_time_hhmmssmm(float(timings.get('transcription_seconds', 0.0)))}")
        typer.echo(f"    - Summarization  : {format_time_hhmmssmm(float(timings.get('summarization_seconds', 0.0)))}")

    typer.echo("\n📊 Audio & Content Metrics:")
    lang_prob = metadata.get("language_probability")
    prob_str = f" ({lang_prob:.0%})" if lang_prob is not None else ""
    typer.echo(f"  • Language         : {metadata.get('language')}{prob_str}")
    duration = float(metadata.get("duration", 0.0))
    speech_duration = float(metadata.get("duration_after_vad", 0.0))
    typer.echo(
        f"  • Audio Duration   : {format_time_hhmmssmm(duration)} "
        f"(Speech: {format_time_hhmmssmm(speech_duration)})"
    )
    typer.echo(
        f"  • Word Counts      : Transcript ({word_counts.get('transcript_words', 0)} words) -> Summary ({word_counts.get('summary_words', 0)} words)"
    )
    typer.echo("==================================================\n")


def _prompt_pipeline_recovery(
    error: Exception,
    current_llm: str,
    current_whisper: str,
    current_compute: str,
    current_device: str,
) -> tuple[str, str, str, str, bool]:
    """Displays interactive recovery options on pipeline error. Returns updated configs and retry flag."""
    import sys

    if not sys.stdin.isatty():
        return current_llm, current_whisper, current_compute, current_device, False

    typer.echo(f"\n⚠️  Pipeline error: {error}", err=True)
    typer.echo("Model or pipeline failure detected. Choose recovery option:", err=True)
    typer.echo(f"  [1] Enter a new LLM model name (current: {current_llm})")
    typer.echo(f"  [2] Enter a new Whisper model name (current: {current_whisper})")
    typer.echo(f"  [3] Change Whisper compute type (current: {current_compute})")
    typer.echo(f"  [4] Change Whisper device (current: {current_device})")
    typer.echo("  [5] Retry pipeline with current configuration")
    typer.echo("  [6] Exit")

    choice = typer.prompt("Select option [1-6]", default="6")
    if choice == "1":
        new_llm = typer.prompt("Enter new LLM model name").strip()
        if new_llm:
            return new_llm, current_whisper, current_compute, current_device, True
    elif choice == "2":
        new_whisper = typer.prompt("Enter new Whisper model name").strip()
        if new_whisper:
            return current_llm, new_whisper, current_compute, current_device, True
    elif choice == "3":
        new_compute = typer.prompt("Enter new Whisper compute type (e.g. int8, float32, default)").strip()
        if new_compute:
            return current_llm, current_whisper, new_compute, current_device, True
    elif choice == "4":
        new_dev = typer.prompt("Enter new Whisper device (e.g. cpu, cuda)").strip()
        if new_dev:
            return current_llm, current_whisper, current_compute, new_dev, True
    elif choice == "5":
        return current_llm, current_whisper, current_compute, current_device, True

    return current_llm, current_whisper, current_compute, current_device, False


def _resolve_prompt_option(text: str | None, path: Path | None, opt_name: str) -> str | None:
    if text and path:
        typer.echo(f"Error: Cannot specify both --{opt_name} and --{opt_name}-file options simultaneously.", err=True)
        raise typer.Exit(code=EXIT_ERROR)
    if not path:
        return text
    if not path.exists():
        typer.echo(f"Error: Prompt file does not exist: {path}", err=True)
        raise typer.Exit(code=EXIT_ERROR)
    if not path.is_file():
        typer.echo(f"Error: Prompt file path is not a file: {path}", err=True)
        raise typer.Exit(code=EXIT_ERROR)
    return path.read_text(encoding="utf-8")


@app.command()
def main(
    target: str = typer.Option(
        ...,
        "--target",
        help="Input file path or YouTube URL of the video/audio to process.",
    ),
    output_dir: Path = typer.Option(Path("output"), "--output-dir"),
    preset: str | None = typer.Option(
        None,
        "--preset",
        "-p",
        help="Configuration profile ('cpu', 'fast', or GPU profiles 'gpu'/'cuda'/'accurate' when GPU mode is enabled).",
    ),
    gpu: bool = typer.Option(False, "--gpu", help="Shortcut for GPU configuration (--preset gpu)."),
    fast: bool = typer.Option(False, "--fast", help="Shortcut for fast CPU configuration (--preset fast)."),
    whisper_model: str | None = typer.Option(None, "--whisper-model", help="Whisper model name ('tiny'., 'small', 'medium', 'large-v3')."),
    whisper_device: str | None = typer.Option(None, "--whisper-device", help="Whisper model device (e.g., 'cpu', 'cuda')."),
    whisper_compute_type: str | None = typer.Option(None, "--whisper-compute-type", help="Whisper model compute type (e.g., 'int8', 'int8_float16', 'float16', 'float32')."),
    whisper_batch_size: int | None = typer.Option(
        None,
        "--whisper-batch-size",
        help="Batch size for faster-whisper transcription. Higher values increase transcription speed but consume significantly more RAM/VRAM memory (Default: 2).",
    ),
    llm_model: str | None = typer.Option(None, "--llm-model"),
    language: str = typer.Option("pt", "--language"),
    video: bool = typer.Option(
        False,
        "--video",
        help="Enforce video summary prompt template (saves summary to resume.md).",
    ),
    meeting: bool = typer.Option(
        False,
        "--meeting",
        help="Enforce meeting summary prompt template (saves summary to meeting_points.md).",
    ),
    transcribe_only: bool = typer.Option(
        False,
        "--transcribe-only",
        help="Execute only audio extraction and speech transcription steps.",
    ),
    summarize_only: bool = typer.Option(
        False,
        "--summarize-only",
        help="Execute only transcript summarization step (target can be .srt file).",
    ),
    prompt: str | None = typer.Option(
        None,
        "--prompt",
        help="Inline custom summary prompt template (supports {transcript} and {language}).",
    ),
    prompt_file: Path | None = typer.Option(
        None,
        "--prompt-file",
        help="Path to custom summary prompt template file.",
    ),
    consolidation_prompt: str | None = typer.Option(
        None,
        "--consolidation-prompt",
        help="Inline custom consolidation prompt template (supports {category}, {items}, and {language}).",
    ),
    consolidation_prompt_file: Path | None = typer.Option(
        None,
        "--consolidation-prompt-file",
        help="Path to custom consolidation prompt template file.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        "-r",
        help="Reuse existing intermediate files (.wav / .srt / summary) if available.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-generation of all pipeline steps.",
    ),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    configure_logging(verbose)

    if not target or not target.strip():
        typer.echo("Error: --target cannot be empty.", err=True)
        raise typer.Exit(code=EXIT_ERROR)
    target = target.strip()

    if resume and force:
        typer.echo("Error: Cannot specify both --resume and --force simultaneously.", err=True)
        raise typer.Exit(code=EXIT_ERROR)

    if video and meeting:
        typer.echo("Error: Cannot specify both --video and --meeting options simultaneously.", err=True)
        raise typer.Exit(code=EXIT_ERROR)

    if gpu and fast:
        typer.echo("Error: Cannot specify both --gpu and --fast options simultaneously.", err=True)
        raise typer.Exit(code=EXIT_ERROR)

    if transcribe_only and summarize_only:
        typer.echo("Error: Cannot specify both --transcribe-only and --summarize-only options simultaneously.", err=True)
        raise typer.Exit(code=EXIT_ERROR)

    effective_custom_prompt = _resolve_prompt_option(prompt, prompt_file, "prompt")
    effective_custom_consolidation_prompt = _resolve_prompt_option(
        consolidation_prompt, consolidation_prompt_file, "consolidation-prompt"
    )


    gpu_supported = is_gpu_available()
    allowed_presets = (
        sorted(list(PRESETS.keys()))
        if gpu_supported
        else sorted([k for k in PRESETS.keys() if k not in GPU_PRESETS])
    )

    if gpu and not gpu_supported:
        typer.echo(
            "Error: GPU execution is disabled or unavailable in CPU mode. "
            f"Available presets: {', '.join(allowed_presets)}",
            err=True,
        )
        raise typer.Exit(code=EXIT_ERROR)

    # Determine preset profile
    selected_preset = preset.lower() if preset else None
    if not selected_preset:
        if gpu:
            selected_preset = "gpu"
        elif fast:
            selected_preset = "fast"
        else:
            selected_preset = "cpu"

    if selected_preset not in allowed_presets:
        available = ", ".join(allowed_presets)
        if selected_preset in GPU_PRESETS and not gpu_supported:
            typer.echo(
                f"Error: Preset '{selected_preset}' requires GPU execution, which is disabled or unavailable in CPU mode. "
                f"Available presets: {available}",
                err=True,
            )
        else:
            typer.echo(f"Error: Unknown preset '{selected_preset}'. Available presets: {available}", err=True)
        raise typer.Exit(code=EXIT_ERROR)

    defaults = PRESETS[selected_preset]

    # Explicit flags override preset defaults
    effective_whisper_model = (whisper_model or str(defaults["whisper_model"])).strip()
    effective_whisper_device = (whisper_device or str(defaults["whisper_device"])).strip()
    effective_whisper_compute_type = (whisper_compute_type or str(defaults["whisper_compute_type"])).strip()
    effective_whisper_batch_size = whisper_batch_size if whisper_batch_size is not None else int(defaults.get("whisper_batch_size", 2))
    effective_llm_model = (llm_model or str(defaults["llm_model"])).strip()

    if not effective_whisper_model:
        typer.echo("Error: --whisper-model cannot be empty.", err=True)
        raise typer.Exit(code=EXIT_ERROR)

    if not effective_llm_model:
        typer.echo("Error: --llm-model cannot be empty.", err=True)
        raise typer.Exit(code=EXIT_ERROR)

    if effective_whisper_batch_size < 1:
        typer.echo(
            f"Error: --whisper-batch-size must be a positive integer (got {effective_whisper_batch_size}).",
            err=True,
        )
        raise typer.Exit(code=EXIT_ERROR)

    if effective_whisper_device.lower() not in VALID_DEVICES:
        allowed_devs = ", ".join(sorted(list(VALID_DEVICES)))
        typer.echo(
            f"Error: Invalid --whisper-device '{effective_whisper_device}'. Allowed values: {allowed_devs}",
            err=True,
        )
        raise typer.Exit(code=EXIT_ERROR)

    if (effective_whisper_device.lower() in ("cuda", "gpu")) and not gpu_supported:
        typer.echo(
            "Error: '--whisper-device cuda' was specified, but GPU execution is disabled or unavailable in CPU mode.",
            err=True,
        )
        raise typer.Exit(code=EXIT_ERROR)

    if effective_whisper_compute_type.lower() not in VALID_COMPUTE_TYPES:
        allowed_types = ", ".join(sorted(list(VALID_COMPUTE_TYPES)))
        typer.echo(
            f"Error: Invalid --whisper-compute-type '{effective_whisper_compute_type}'. Allowed values: {allowed_types}",
            err=True,
        )
        raise typer.Exit(code=EXIT_ERROR)

    if effective_whisper_device.lower() in ("cpu", "auto") and effective_whisper_compute_type.lower() in (
        "float16",
        "int8_float16",
        "bfloat16",
        "int8_bfloat16",
    ):
        typer.echo(
            f"Error: '--whisper-compute-type {effective_whisper_compute_type}' requires GPU (cuda). On CPU, available compute types are: default, float32, int8, int8_float32.",
            err=True,
        )
        raise typer.Exit(code=EXIT_ERROR)

    is_url = (
        target.startswith(("http://", "https://", "www."))
        or "youtube.com" in target
        or "youtu.be" in target
    )

    if not is_url:
        local_path = Path(target)
        if not local_path.exists():
            logger.error("Input file does not exist: %s", local_path)
            typer.echo(f"Error: Input file does not exist: {local_path}", err=True)
            raise typer.Exit(code=EXIT_ERROR)
        if not local_path.is_file():
            logger.error("Input path is not a file: %s", local_path)
            typer.echo(f"Error: Input path is not a file: {local_path}", err=True)
            raise typer.Exit(code=EXIT_ERROR)

        if local_path.suffix.lower() == ".srt" and not transcribe_only:
            summarize_only = True

    if video:
        is_video = True
    elif meeting:
        is_video = False
    else:
        is_video = is_url

    if verbose:
        typer.echo("\n--- Configuration ---")
        typer.echo(f"  Target: {target}")
        typer.echo(f"  Output Dir: {output_dir}")
        typer.echo(f"  Preset: {selected_preset}")
        typer.echo(f"  Whisper Model: {effective_whisper_model}")
        typer.echo(f"  Whisper Device: {effective_whisper_device}")
        typer.echo(f"  Whisper Compute Type: {effective_whisper_compute_type}")
        typer.echo(f"  Whisper Batch Size: {effective_whisper_batch_size}")
        typer.echo(f"  LLM Model: {effective_llm_model}")
        typer.echo(f"  Language: {language}")
        typer.echo(f"  Mode: {'Video' if is_video else 'Meeting'}")
        typer.echo(f"  Resume: {resume}")
        typer.echo(f"  Force: {force}")
        if transcribe_only:
            typer.echo("  Step: Transcribe Only")
        elif summarize_only:
            typer.echo("  Step: Summarize Only")
        typer.echo("---------------------\n")
        logger.debug(
            "Resolved config: target=%s, output_dir=%s, preset=%s, whisper_model=%s, whisper_device=%s, "
            "whisper_compute_type=%s, whisper_batch_size=%d, llm_model=%s, language=%s, is_video=%s, "
            "resume=%s, force=%s, transcribe_only=%s, summarize_only=%s",
            target,
            output_dir,
            selected_preset,
            effective_whisper_model,
            effective_whisper_device,
            effective_whisper_compute_type,
            effective_whisper_batch_size,
            effective_llm_model,
            language,
            is_video,
            resume,
            force,
            transcribe_only,
            summarize_only,
        )

    temp_file: Path | None = None
    effective_resume = resume

    try:
        if is_url:
            t0 = time.perf_counter()
            temp_file = download_youtube_audio(target)
            t_dl = time.perf_counter() - t0
            logger.info("YouTube audio download completed in %.2fs", t_dl)
            input_path = temp_file
        else:
            input_path = Path(target)

        while True:
            try:
                transcript_path, summary_path, metadata = run_pipeline(
                    input_path=input_path,
                    output_dir=output_dir,
                    whisper_model=effective_whisper_model,
                    whisper_device=effective_whisper_device,
                    whisper_compute_type=effective_whisper_compute_type,
                    llm_model=effective_llm_model,
                    language=language,
                    whisper_batch_size=effective_whisper_batch_size,
                    is_video=is_video,
                    verbose=verbose,
                    transcribe_only=transcribe_only,
                    summarize_only=summarize_only,
                    custom_prompt=effective_custom_prompt,
                    custom_consolidation_prompt=effective_custom_consolidation_prompt,
                    resume=effective_resume,
                )
                break


            except (KeyboardInterrupt, typer.Exit):
                raise
            except Exception as e:
                (
                    effective_llm_model,
                    effective_whisper_model,
                    effective_whisper_compute_type,
                    effective_whisper_device,
                    retry,
                ) = _prompt_pipeline_recovery(
                    error=e,
                    current_llm=effective_llm_model,
                    current_whisper=effective_whisper_model,
                    current_compute=effective_whisper_compute_type,
                    current_device=effective_whisper_device,
                )
                if retry:
                    effective_resume = True
                    continue
                logger.exception("Pipeline execution failed")
                raise typer.Exit(code=EXIT_ERROR)



        print_pipeline_status(
            transcript_path=transcript_path,
            summary_path=summary_path,
            output_dir=output_dir,
            metadata=metadata,
            verbose=verbose,
        )

    except KeyboardInterrupt:
        typer.echo("\nProcess interrupted by user. Exiting...", err=True)
        raise typer.Exit(code=EXIT_INTERRUPTED)
    except typer.Exit:
        raise
    except Exception:
        logger.exception("Pipeline execution failed")
        raise typer.Exit(code=EXIT_ERROR)
    finally:
        if temp_file and temp_file.exists():
            try:
                temp_file.unlink()
                logger.debug("Deleted temporary YouTube audio file: %s", temp_file)
            except Exception as e:
                logger.warning("Failed to delete temporary file %s: %s", temp_file, e)
