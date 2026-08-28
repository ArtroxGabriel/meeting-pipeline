from __future__ import annotations

import logging
from pathlib import Path

from faster_whisper import BatchedInferencePipeline, WhisperModel

import os

logger = logging.getLogger(__name__)

MS_PER_HOUR = 3_600_000
MS_PER_MINUTE = 60_000
MS_PER_SECOND = 1000
DEFAULT_BATCH_SIZE = 2


def _resolve_download_root() -> str | None:
    cache_env = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    target_dir = Path(cache_env) if cache_env else (Path.home() / ".cache" / "huggingface" / "hub")
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        test_file = target_dir / ".permission_check"
        test_file.touch()
        test_file.unlink()
        return None
    except (PermissionError, OSError) as e:
        fallback = Path("/tmp/huggingface_cache")
        fallback.mkdir(parents=True, exist_ok=True)
        logger.warning(
            "Cache directory '%s' is not writable (%s). Falling back to '%s'",
            target_dir,
            e,
            fallback,
        )
        return str(fallback)


def format_timestamp(seconds: float) -> str:
    total_ms = round(seconds * MS_PER_SECOND)
    hours, remainder = divmod(total_ms, MS_PER_HOUR)
    minutes, remainder = divmod(remainder, MS_PER_MINUTE)
    secs, ms = divmod(remainder, MS_PER_SECOND)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def segments_to_srt(segments: list) -> tuple[str, str]:
    lines: list[str] = []
    srt_blocks: list[str] = []
    index = 1

    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        lines.append(text)
        start_str = format_timestamp(segment.start)
        end_str = format_timestamp(segment.end)
        srt_blocks.append(f"{index}\n{start_str} --> {end_str}\n{text}\n")
        index += 1

    plain_text = "\n".join(lines).strip()
    srt_text = "\n".join(srt_blocks).strip()
    return plain_text, srt_text


def transcribe_file(
    audio_path: Path,
    model_name: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = "pt",
    batch_size: int = DEFAULT_BATCH_SIZE,
    verbose: bool = False,
    log_progress: bool = True,
) -> tuple[str, str, dict]:
    if not audio_path.exists():
        logger.error("Audio file does not exist: %s", audio_path)
        raise FileNotFoundError(audio_path)

    if batch_size < 1:
        logger.error("batch_size must be >= 1 (got %s)", batch_size)
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    if verbose:
        logging.getLogger("faster_whisper").setLevel(logging.DEBUG)

    effective_compute_type = compute_type
    if compute_type == "int8" and (device.lower() in ("cuda", "gpu") or device.lower().startswith("cuda")):
        effective_compute_type = "int8_float16"

    download_root = _resolve_download_root()
    model = WhisperModel(
        model_name,
        device=device,
        compute_type=effective_compute_type,
        download_root=download_root,
    )
    batched_model = BatchedInferencePipeline(model=model)

    segments, info = batched_model.transcribe(
        str(audio_path),
        language=language,
        vad_filter=True,
        word_timestamps=False,
        batch_size=batch_size,
        log_progress=log_progress,
    )

    # Convert generator to list to iterate for both SRT and plain text
    segments_list = list(segments)
    plain_text, srt_text = segments_to_srt(segments_list)

    metadata = {
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "duration_after_vad": info.duration_after_vad,
    }

    if plain_text:
        return plain_text, srt_text, metadata

    logger.error("Empty transcript generated")
    raise RuntimeError("empty transcript")

