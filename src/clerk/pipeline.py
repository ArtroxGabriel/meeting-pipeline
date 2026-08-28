from __future__ import annotations

import json
import logging
from pathlib import Path

from .audio import extract_audio
from .prompts import clean_srt_for_prompt
from .summarize import summarize_transcript
from .transcribe import transcribe_file

import time

logger = logging.getLogger(__name__)


def _build_pipeline_metadata(
    *,
    language: str,
    language_probability: float | None = None,
    duration: float = 0.0,
    duration_after_vad: float = 0.0,
    t_audio: float = 0.0,
    t_transcribe: float = 0.0,
    t_summarize: float = 0.0,
    t_total: float = 0.0,
    whisper_model: str,
    whisper_device: str,
    whisper_compute_type: str,
    whisper_batch_size: int,
    llm_model: str,
    transcript_words: int = 0,
    summary_words: int = 0,
    audio_path: Path | None = None,
    transcript_path: Path | None = None,
    summary_path: Path | None = None,
    metadata_path: Path | None = None,
) -> dict:
    """Constructs uniform metadata dictionary structure across pipeline execution modes."""
    return {
        "language": language,
        "language_probability": language_probability,
        "duration": duration,
        "duration_after_vad": duration_after_vad,
        "timings": {
            "audio_extraction_seconds": round(t_audio, 3),
            "transcription_seconds": round(t_transcribe, 3),
            "summarization_seconds": round(t_summarize, 3),
            "total_seconds": round(t_total, 3),
        },
        "models": {
            "whisper_model": whisper_model,
            "whisper_device": whisper_device,
            "whisper_compute_type": whisper_compute_type,
            "whisper_batch_size": whisper_batch_size,
            "llm_model": llm_model,
        },
        "word_counts": {
            "transcript_words": transcript_words,
            "summary_words": summary_words,
        },
        "output_files": {
            "audio_path": str(audio_path) if audio_path else None,
            "transcript_path": str(transcript_path) if transcript_path else None,
            "summary_path": str(summary_path) if summary_path else None,
            "metadata_path": str(metadata_path) if metadata_path else None,
        },
    }


def run_pipeline(
    input_path: Path,
    output_dir: Path,
    whisper_model: str,
    whisper_device: str,
    whisper_compute_type: str,
    llm_model: str,
    language: str | None,
    whisper_batch_size: int = 2,
    is_video: bool = False,
    verbose: bool = False,
    transcribe_only: bool = False,
    summarize_only: bool = False,
    custom_prompt: str | None = None,
    custom_consolidation_prompt: str | None = None,
    resume: bool = False,
) -> tuple[Path | None, Path | None, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    effective_lang = language or "pt"

    if summarize_only:
        base_stem = stem[:-11] if stem.endswith("_transcript") else stem
        summary_filename = f"{base_stem}_resume.md" if is_video else f"{base_stem}_meeting_points.md"
        summary_path = output_dir / summary_filename
        metadata_path = output_dir / f"{base_stem}_metadata.json"

        if input_path.suffix.lower() in (".srt", ".txt"):
            transcript_path = input_path
        else:
            default_srt = output_dir / f"{base_stem}_transcript.srt"
            alt_srt = output_dir / f"{base_stem}.srt"
            same_dir_srt = input_path.with_suffix(".srt")
            if default_srt.exists():
                transcript_path = default_srt
            elif alt_srt.exists():
                transcript_path = alt_srt
            elif same_dir_srt.exists():
                transcript_path = same_dir_srt
            else:
                transcript_path = default_srt

        if not transcript_path.exists():
            logger.error("Transcript file not found for summarize-only mode: %s", transcript_path)
            raise FileNotFoundError(
                f"Transcript file '{transcript_path}' does not exist. Run transcription first or provide an existing .srt file."
            )

        if resume and summary_path.exists() and summary_path.stat().st_size > 0:
            logger.info("Summary already exists at %s. Re-run without --resume (-r) to regenerate.", summary_path)
            existing_meta = {}
            if metadata_path.exists():
                try:
                    existing_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            return transcript_path, summary_path, existing_meta

        t_start = time.perf_counter()
        logger.info("Reading transcript from %s for summarize-only execution...", transcript_path)
        raw_content = transcript_path.read_text(encoding="utf-8")
        is_srt = transcript_path.suffix.lower() == ".srt"
        plain_text_transcript = clean_srt_for_prompt(raw_content) if is_srt else raw_content.strip()

        logger.info("Starting summarization...")
        t0 = time.perf_counter()
        summary = summarize_transcript(
            transcript=plain_text_transcript,
            model_name=llm_model,
            language=effective_lang,
            is_video=is_video,
            custom_prompt=custom_prompt,
            custom_consolidation_prompt=custom_consolidation_prompt,
        )

        t_summarize = time.perf_counter() - t0
        t_total = time.perf_counter() - t_start
        logger.info("Summarization completed in %.2fs", t_summarize)

        metadata = _build_pipeline_metadata(
            language=effective_lang,
            language_probability=1.0,
            t_summarize=t_summarize,
            t_total=t_total,
            whisper_model="skipped",
            whisper_device="skipped",
            whisper_compute_type="skipped",
            whisper_batch_size=0,
            llm_model=llm_model,
            transcript_words=len(plain_text_transcript.split()),
            summary_words=len(summary.split()),
            transcript_path=transcript_path,
            summary_path=summary_path,
            metadata_path=metadata_path,
        )

        summary_path.write_text(summary + "\n", encoding="utf-8")
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info("Summary written to %s", summary_path)
        return transcript_path, summary_path, metadata

    summary_filename = f"{stem}_resume.md" if is_video else f"{stem}_meeting_points.md"
    audio_path = output_dir / f"{stem}_normalized.wav"
    transcript_path = output_dir / f"{stem}_transcript.srt"
    summary_path = output_dir / summary_filename
    metadata_path = output_dir / f"{stem}_metadata.json"

    if resume and summary_path.exists() and summary_path.stat().st_size > 0:
        logger.info("Summary already exists at %s. Re-run without --resume (-r) to regenerate.", summary_path)
        existing_meta = {}
        if metadata_path.exists():
            try:
                existing_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return transcript_path, summary_path, existing_meta

    t_start = time.perf_counter()
    t_audio = 0.0
    t_transcribe = 0.0
    plain_text_transcript: str | None = None
    srt_transcript: str | None = None
    metadata: dict = {}

    if resume and transcript_path.exists() and transcript_path.stat().st_size > 0:
        logger.info("Reusing existing transcript from %s...", transcript_path)
        srt_content = transcript_path.read_text(encoding="utf-8")
        plain_text_transcript = clean_srt_for_prompt(srt_content)
        srt_transcript = srt_content.strip()
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                metadata = {}
    else:
        if resume and audio_path.exists() and audio_path.stat().st_size > 0:
            logger.info("Reusing existing normalized audio from %s...", audio_path)
            normalized_audio = audio_path
        else:
            logger.info("Starting audio extraction from %s...", input_path)
            t0 = time.perf_counter()
            normalized_audio = extract_audio(input_path, audio_path)
            t_audio = time.perf_counter() - t0
            logger.info("Audio extraction completed in %.2fs", t_audio)

        logger.info("Starting transcription...")
        t0 = time.perf_counter()
        plain_text_transcript, srt_transcript, metadata = transcribe_file(
            normalized_audio,
            model_name=whisper_model,
            device=whisper_device,
            compute_type=whisper_compute_type,
            language=language,
            batch_size=whisper_batch_size,
            verbose=verbose,
        )
        t_transcribe = time.perf_counter() - t0
        logger.info("Transcription completed in %.2fs", t_transcribe)

        # Write transcript and intermediate metadata ahead of summarization
        transcript_path.write_text(srt_transcript + "\n", encoding="utf-8")
        logger.info("Transcript written to %s", transcript_path)

        interim_metadata = _build_pipeline_metadata(
            language=metadata.get("language", effective_lang),
            language_probability=metadata.get("language_probability"),
            duration=float(metadata.get("duration", 0.0)),
            duration_after_vad=float(metadata.get("duration_after_vad", 0.0)),
            t_audio=t_audio,
            t_transcribe=t_transcribe,
            t_total=time.perf_counter() - t_start,
            whisper_model=whisper_model,
            whisper_device=whisper_device,
            whisper_compute_type=whisper_compute_type,
            whisper_batch_size=whisper_batch_size,
            llm_model="in_progress" if not transcribe_only else "skipped",
            transcript_words=len(plain_text_transcript.split()),
            summary_words=0,
            audio_path=audio_path,
            transcript_path=transcript_path,
            metadata_path=metadata_path,
        )
        metadata_path.write_text(
            json.dumps(interim_metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    if transcribe_only:
        t_total = time.perf_counter() - t_start
        logger.info("Transcribe-only execution completed in %.2fs", t_total)
        final_tx_meta = _build_pipeline_metadata(
            language=metadata.get("language", effective_lang),
            language_probability=metadata.get("language_probability"),
            duration=float(metadata.get("duration", 0.0)),
            duration_after_vad=float(metadata.get("duration_after_vad", 0.0)),
            t_audio=t_audio,
            t_transcribe=t_transcribe,
            t_total=t_total,
            whisper_model=whisper_model,
            whisper_device=whisper_device,
            whisper_compute_type=whisper_compute_type,
            whisper_batch_size=whisper_batch_size,
            llm_model="skipped",
            transcript_words=len(plain_text_transcript.split()),
            summary_words=0,
            audio_path=audio_path,
            transcript_path=transcript_path,
            metadata_path=metadata_path,
        )
        metadata_path.write_text(
            json.dumps(final_tx_meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return transcript_path, None, final_tx_meta

    logger.info("Starting summarization...")
    t0 = time.perf_counter()
    summary = summarize_transcript(
        transcript=plain_text_transcript,
        model_name=llm_model,
        language=effective_lang,
        is_video=is_video,
        custom_prompt=custom_prompt,
        custom_consolidation_prompt=custom_consolidation_prompt,
    )

    t_summarize = time.perf_counter() - t0
    t_total = time.perf_counter() - t_start
    logger.info("Summarization completed in %.2fs", t_summarize)
    logger.info("Total pipeline execution time: %.2fs", t_total)

    updated_metadata = _build_pipeline_metadata(
        language=metadata.get("language", effective_lang),
        language_probability=metadata.get("language_probability"),
        duration=float(metadata.get("duration", 0.0)),
        duration_after_vad=float(metadata.get("duration_after_vad", 0.0)),
        t_audio=t_audio,
        t_transcribe=t_transcribe,
        t_summarize=t_summarize,
        t_total=t_total,
        whisper_model=whisper_model if whisper_model else str(metadata.get("models", {}).get("whisper_model", "skipped")),
        whisper_device=whisper_device if whisper_device else str(metadata.get("models", {}).get("whisper_device", "skipped")),
        whisper_compute_type=whisper_compute_type if whisper_compute_type else str(metadata.get("models", {}).get("whisper_compute_type", "skipped")),
        whisper_batch_size=whisper_batch_size if whisper_batch_size else int(metadata.get("models", {}).get("whisper_batch_size", 0)),
        llm_model=llm_model,
        transcript_words=len(plain_text_transcript.split()),
        summary_words=len(summary.split()),
        audio_path=audio_path,
        transcript_path=transcript_path,
        summary_path=summary_path,
        metadata_path=metadata_path,
    )

    summary_path.write_text(summary + "\n", encoding="utf-8")
    metadata_path.write_text(
        json.dumps(updated_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    logger.info("Summary written to %s", summary_path)

    return transcript_path, summary_path, updated_metadata




