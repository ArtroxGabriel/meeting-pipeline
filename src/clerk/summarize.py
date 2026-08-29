from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import re

import httpx

from .prompts import (
    PromptManager,
    clean_llm_output,
    clean_srt_for_prompt,
    get_language_name,
    is_meaningful_transcript,
)

logger = logging.getLogger(__name__)


DEFAULT_LLM_MODEL = "LiquidAI/lfm2.5-1.2b-instruct"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_TIMEOUT_SECONDS = 300.0
UNLOAD_TIMEOUT_SECONDS = 10.0
PULL_TIMEOUT_SECONDS = 600.0
DEFAULT_CPU_MAX_WORDS_PER_CHUNK = 3000
DEFAULT_GPU_MAX_WORDS_PER_CHUNK = 4500
DEFAULT_MAX_WORDS_PER_CHUNK = DEFAULT_CPU_MAX_WORDS_PER_CHUNK


def split_transcript_smart(
    transcript: str,
    max_words: int = DEFAULT_MAX_WORDS_PER_CHUNK,
) -> list[str]:
    """Splits transcript into chunks without breaking mid-sentence, word, or phrase."""
    cleaned = clean_srt_for_prompt(transcript)
    if not cleaned:
        return []

    lines = cleaned.splitlines()
    units: list[str] = []

    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue
        sub_sentences = re.split(r"(?<=[.!?])\s+", line_str)
        for s in sub_sentences:
            if s.strip():
                units.append(s.strip())

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_word_count = 0

    for unit in units:
        words = unit.split()
        unit_word_count = len(words)
        if not words:
            continue

        if unit_word_count > max_words:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_word_count = 0

            clauses = re.split(r"(?<=[,;:])\s+", unit)
            if len(clauses) == 1:
                for i in range(0, len(words), max_words):
                    chunks.append(" ".join(words[i : i + max_words]))
            else:
                sub_chunk: list[str] = []
                sub_count = 0
                for clause in clauses:
                    c_words = clause.split()
                    if not c_words:
                        continue
                    if sub_count + len(c_words) > max_words and sub_chunk:
                        chunks.append(" ".join(sub_chunk))
                        sub_chunk = [clause]
                        sub_count = len(c_words)
                    else:
                        sub_chunk.append(clause)
                        sub_count += len(c_words)
                if sub_chunk:
                    chunks.append(" ".join(sub_chunk))
            continue

        if current_word_count + unit_word_count > max_words:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [unit]
            current_word_count = unit_word_count
        else:
            current_chunk.append(unit)
            current_word_count += unit_word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


split_transcript_by_words = split_transcript_smart


@dataclass(frozen=True)
class SummaryConfig:
    sections: list[str]
    primary_section: str


VIDEO_CONFIG = SummaryConfig(
    sections=["Resumo geral", "Principais tópicos", "Momentos importantes", "Conclusões ou mensagens finais"],
    primary_section="Resumo geral",
)
MEETING_CONFIG = SummaryConfig(
    sections=["Pontos principais", "Decisões", "Ações", "Pendências"],
    primary_section="Pontos principais",
)


def get_summary_config(is_video: bool) -> SummaryConfig:
    return VIDEO_CONFIG if is_video else MEETING_CONFIG


def format_empty_fallback(section: str, primary_section: str) -> str:
    return f"- Nenhum {section.lower()} registrado." if section == primary_section else "- Nenhuma registrada."


def unload_ollama_model(
    model_name: str,
    base_url: str,
    timeout_seconds: float = UNLOAD_TIMEOUT_SECONDS,
) -> None:
    """Unload model from Ollama memory by posting keep_alive: 0."""
    payload = {
        "model": model_name,
        "keep_alive": 0,
    }
    try:
        with httpx.Client(base_url=base_url, timeout=timeout_seconds) as client:
            client.post("/api/generate", json=payload)
        logger.info("Unloaded Ollama model '%s' from memory", model_name)
    except Exception as e:
        logger.warning("Failed to unload Ollama model '%s': %s", model_name, e)


def parse_summary_sections(summary: str, is_video: bool = False) -> dict[str, list[str]]:
    config = get_summary_config(is_video)
    sections: dict[str, list[str]] = {sec: [] for sec in config.sections}
    current_section: str | None = None

    regex_pattern = r"^##\s*(" + "|".join(re.escape(sec) for sec in config.sections) + r")\b"

    for line in summary.splitlines():
        line_strip = line.strip()
        if not line_strip:
            continue

        header_match = re.match(regex_pattern, line_strip, re.IGNORECASE)

        if header_match:
            matched_name = header_match.group(1).lower()
            for key in sections.keys():
                if key.lower() == matched_name:
                    current_section = key
                    break
            continue

        if current_section:
            item_match = re.match(r"^([-*]|\d+\.)\s*(.*)$", line_strip)
            content = item_match.group(2).strip() if item_match else line_strip
            lower_content = content.lower()

            if content and not any(
                phrase in lower_content
                for phrase in [
                    "nenhuma registrada",
                    "nenhum ponto",
                    "nenhum resumo",
                    "não há",
                    "none registered",
                    "no summary",
                ]
            ):
                sections[current_section].append(content)

    return sections


def _call_ollama_generate(
    prompt: str,
    model_name: str,
    base_url: str,
    timeout_seconds: float,
) -> str:
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
    }

    try:
        with httpx.Client(base_url=base_url, timeout=timeout_seconds) as client:
            response = client.post("/api/generate", json=payload)

            if response.status_code != 200 and ("not found" in response.text.lower() or response.status_code == 404):
                logger.info("Model '%s' not found locally in Ollama. Attempting automatic model pull...", model_name)
                try:
                    pull_resp = client.post(
                        "/api/pull",
                        json={"name": model_name, "stream": False},
                        timeout=PULL_TIMEOUT_SECONDS,
                    )
                    if pull_resp.status_code == 200:
                        logger.info("Model '%s' successfully pulled. Resuming generation...", model_name)
                        response = client.post("/api/generate", json=payload)
                    else:
                        logger.error("Failed to pull model '%s' from Ollama: %s", model_name, pull_resp.text)
                        raise RuntimeError(
                            f"ollama request failed: Model '{model_name}' not found locally and auto-pull failed. "
                            f"To pull manually, run 'ollama pull {model_name}'."
                        )
                except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as pull_err:
                    logger.error("Network error while pulling model '%s': %s", model_name, pull_err)
                    raise RuntimeError(
                        f"ollama request failed: Model '{model_name}' not found locally and cannot be pulled while offline. "
                        f"Run 'ollama pull {model_name}'."
                    ) from pull_err

            if response.status_code != 200:
                logger.error("Ollama request failed: %s %s", response.status_code, response.text)
                raise RuntimeError(f"ollama request failed: {response.status_code} {response.text.strip()}")

            data = response.json()
            content = data.get("response", "").strip()
            return clean_llm_output(content)
    except (httpx.ConnectError, httpx.ConnectTimeout) as e:
        logger.error("Could not connect to Ollama at %s: %s", base_url, e)
        raise RuntimeError(
            f"ollama request failed: Could not connect to local Ollama server at {base_url}. Ensure Ollama is running ('ollama serve')."
        ) from e


def summarize_transcript(
    transcript: str,
    model_name: str = DEFAULT_LLM_MODEL,
    base_url: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_words_per_chunk: int | None = None,
    language: str = "pt",
    is_video: bool = False,
    is_gpu_model: bool = False,
    custom_prompt: str | None = None,
    custom_consolidation_prompt: str | None = None,
) -> str:
    if base_url is None:
        base_url = os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)

    cleaned_transcript = clean_srt_for_prompt(transcript)
    if not transcript or not transcript.strip() or not is_meaningful_transcript(transcript):
        logger.error("Transcript is empty, garbled, or contains insufficient speech content")
        raise ValueError("transcript is empty")

    lang_name = get_language_name(language)
    config = get_summary_config(is_video)

    # Detect GPU model if not explicitly specified
    if not is_gpu_model:
        lower_m = model_name.lower()
        if "llama3" in lower_m or "gpu" in lower_m or "cuda" in lower_m or "8b" in lower_m:
            is_gpu_model = True

    effective_max_words = (
        max_words_per_chunk
        if max_words_per_chunk is not None
        else (DEFAULT_GPU_MAX_WORDS_PER_CHUNK if is_gpu_model else DEFAULT_CPU_MAX_WORDS_PER_CHUNK)
    )

    prompt_strategy = PromptManager.get_strategy(
        is_gpu_model=is_gpu_model,
        custom_prompt=custom_prompt,
        custom_consolidation_prompt=custom_consolidation_prompt,
    )

    words = cleaned_transcript.split()
    try:
        if len(words) <= effective_max_words:
            prompt = prompt_strategy.build_summary_prompt(
                transcript=cleaned_transcript,
                language=lang_name,
                is_video=is_video,
            )
            content = _call_ollama_generate(
                prompt=prompt,
                model_name=model_name,
                base_url=base_url,
                timeout_seconds=timeout_seconds,
            )
            if content:
                return content
            logger.error("Ollama returned empty response")
            raise RuntimeError("empty summary")

        logger.info(
            "Transcript length (%d words) exceeds chunk size (%d words). Processing in chunks...",
            len(words),
            effective_max_words,
        )
        chunks = split_transcript_smart(cleaned_transcript, effective_max_words)

        combined_sections: dict[str, list[str]] = {sec: [] for sec in config.sections}

        for i, chunk in enumerate(chunks):
            logger.info("Summarizing chunk %d/%d...", i + 1, len(chunks))
            chunk_prompt = prompt_strategy.build_summary_prompt(
                transcript=chunk,
                language=lang_name,
                is_video=is_video,
            )
            chunk_summary = _call_ollama_generate(
                prompt=chunk_prompt,
                model_name=model_name,
                base_url=base_url,
                timeout_seconds=timeout_seconds,
            )
            if chunk_summary:
                chunk_sections = parse_summary_sections(chunk_summary, is_video=is_video)
                for sec, items in chunk_sections.items():
                    if sec in combined_sections:
                        combined_sections[sec].extend(items)

        logger.info("Consolidating section summaries...")
        consolidated_summaries: dict[str, str] = {}

        # 1. Zero-item and single-item bypass (0 LLM calls)
        sections_needing_consolidation: dict[str, list[str]] = {}
        for sec, items in combined_sections.items():
            if not items:
                consolidated_summaries[sec] = format_empty_fallback(sec, config.primary_section)
            elif len(items) == 1:
                consolidated_summaries[sec] = f"- {items[0]}"
            else:
                sections_needing_consolidation[sec] = items

        # 2. Consolidate sections with multiple items
        if sections_needing_consolidation:
            if (
                is_gpu_model
                and not custom_consolidation_prompt
                and len(sections_needing_consolidation) > 1
                and hasattr(prompt_strategy, "build_multi_consolidation_prompt")
            ):
                logger.info("Running unified GPU multi-category consolidation...")
                multi_prompt = prompt_strategy.build_multi_consolidation_prompt(
                    items_by_section=sections_needing_consolidation,
                    language=lang_name,
                )
                multi_content = _call_ollama_generate(
                    prompt=multi_prompt,
                    model_name=model_name,
                    base_url=base_url,
                    timeout_seconds=timeout_seconds,
                )
                if multi_content:
                    parsed_multi = parse_summary_sections(multi_content, is_video=is_video)
                    for sec, items in sections_needing_consolidation.items():
                        if parsed_multi.get(sec):
                            consolidated_summaries[sec] = "\n".join(f"- {item}" for item in parsed_multi[sec])
                        else:
                            consolidated_summaries[sec] = "\n".join(f"- {item}" for item in items)
                else:
                    for sec, items in sections_needing_consolidation.items():
                        consolidated_summaries[sec] = "\n".join(f"- {item}" for item in items)
            else:
                for sec, items in sections_needing_consolidation.items():
                    items_text = "\n".join(f"- {item}" for item in items)
                    prompt = prompt_strategy.build_consolidation_prompt(
                        category=sec,
                        items=items_text,
                        language=lang_name,
                    )
                    consolidated_content = _call_ollama_generate(
                        prompt=prompt,
                        model_name=model_name,
                        base_url=base_url,
                        timeout_seconds=timeout_seconds,
                    )
                    if not consolidated_content:
                        consolidated_content = "\n".join(f"- {item}" for item in items)
                    consolidated_summaries[sec] = consolidated_content

        final_parts = [f"## {sec}\n{consolidated_summaries[sec]}" for sec in config.sections]
        return "\n\n".join(final_parts).strip()
    finally:
        unload_ollama_model(model_name=model_name, base_url=base_url)
