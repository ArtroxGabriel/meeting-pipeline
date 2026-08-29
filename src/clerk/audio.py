from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path

import uuid

logger = logging.getLogger(__name__)


DEFAULT_SAMPLE_RATE = "16000"
DEFAULT_CHANNELS = "1"
DEFAULT_AUDIO_CODEC = "pcm_s16le"


def ensure_binary(binary_name: str) -> None:
    if shutil.which(binary_name):
        return

    logger.error("%s not found in PATH", binary_name)
    raise RuntimeError(f"{binary_name} not found in PATH")


def ensure_ffmpeg() -> None:
    ensure_binary("ffmpeg")


def ensure_yt_dlp() -> None:
    ensure_binary("yt-dlp")


def extract_youtube_id(url: str) -> str | None:
    match = re.search(r"(?:v=|\/|embed\/|shorts\/)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None


def download_youtube_audio(url: str) -> Path:
    ensure_yt_dlp()
    temp_dir = Path("/tmp")
    yt_id = extract_youtube_id(url)
    default_filename = f"yt_{yt_id}" if yt_id else f"yt_{uuid.uuid4().hex[:8]}"

    command = [
        "yt-dlp",
        "-x",
        "--audio-format",
        "wav",
        "--restrict-filenames",
        "--print",
        "after_move:filepath",
        "-o",
        str(temp_dir / "yt_%(title)s.%(ext)s"),
        url,
    ]

    logger.info("Downloading YouTube audio from %s...", url)
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        logger.error("yt-dlp failed: %s", result.stderr.strip())
        raise RuntimeError(f"Failed to download YouTube audio: {result.stderr.strip()}")

    if result.stdout.strip():
        lines = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
        if lines:
            candidate = Path(lines[-1])
            if candidate.exists():
                return candidate

    fallback_path = temp_dir / f"{default_filename}.wav"
    if fallback_path.exists():
        return fallback_path

    logger.error("Downloaded file not found")
    raise RuntimeError("Downloaded YouTube audio file not found")



def extract_audio(input_path: Path, output_path: Path) -> Path:
    if not input_path.exists():
        logger.error("Input file does not exist: %s", input_path)
        raise FileNotFoundError(input_path)

    ensure_ffmpeg()

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        logger.error("Permission denied creating output directory %s: %s", output_path.parent, e)
        raise PermissionError(f"Permission denied creating directory {output_path.parent}. Check write permissions.") from e

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-acodec",
        DEFAULT_AUDIO_CODEC,
        "-ar",
        DEFAULT_SAMPLE_RATE,
        "-ac",
        DEFAULT_CHANNELS,
        str(output_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode == 0:
        return output_path

    stderr_msg = result.stderr.strip()
    logger.error("ffmpeg failed: %s", stderr_msg)
    if "Permission denied" in stderr_msg:
        raise PermissionError(f"Permission denied writing to {output_path}. Check output directory permissions.")
    raise RuntimeError("audio extraction failed")

