from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from clerk.summarize import (
    parse_summary_sections,
    split_transcript_by_words,
    summarize_transcript,
)


def test_split_transcript_by_words() -> None:
    text = "word1 word2 word3\nword4 word5 word6"
    chunks = split_transcript_by_words(text, max_words=3)
    assert len(chunks) == 2
    assert chunks[0] == "word1 word2 word3"
    assert chunks[1] == "word4 word5 word6"


def test_parse_summary_sections() -> None:
    summary_text = """
## Pontos principais
- Ponto 1
* Ponto 2

## Decisões
1. Decisão A

## Ações
- Nenhuma registrada.

## Pendências
- Pendência X
"""
    sections = parse_summary_sections(summary_text)
    assert sections["Pontos principais"] == ["Ponto 1", "Ponto 2"]
    assert sections["Decisões"] == ["Decisão A"]
    assert sections["Ações"] == []
    assert sections["Pendências"] == ["Pendência X"]


def test_summarize_empty_transcript() -> None:
    with pytest.raises(ValueError, match="transcript is empty"):
        summarize_transcript("   ")


def test_summarize_success() -> None:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": "## Pontos principais\n- Reunião produtiva"}

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__.return_value = mock_client

    with patch("httpx.Client", return_value=mock_client):
        result = summarize_transcript("Some transcript content")
        assert result == "## Pontos principais\n- Reunião produtiva"
        assert mock_client.post.call_count == 2
        generate_call = mock_client.post.call_args_list[0]
        unload_call = mock_client.post.call_args_list[1]
        payload = generate_call[1]["json"]
        assert payload["model"] == "LiquidAI/lfm2.5-1.2b-instruct"
        assert "Some transcript content" in payload["prompt"]
        assert unload_call[1]["json"] == {
            "model": "LiquidAI/lfm2.5-1.2b-instruct",
            "keep_alive": 0,
        }


def test_summarize_video_prompt() -> None:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": "## Resumo geral\n- Video summary point"}

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__.return_value = mock_client

    with patch("httpx.Client", return_value=mock_client):
        result = summarize_transcript("Video transcript content", is_video=True)
        assert result == "## Resumo geral\n- Video summary point"
        assert mock_client.post.call_count == 2
        payload = mock_client.post.call_args_list[0][1]["json"]
        assert "transcript of a video" in payload["prompt"]
        assert "## Resumo geral" in payload["prompt"]


def test_summarize_multi_chunk() -> None:
    # 6 words total, max_words_per_chunk=3 -> 2 chunks
    long_transcript = "one two three\nfour five six"

    # Mock Ollama responses: 2 chunk summaries + 4 consolidation responses + 1 unload response
    chunk_1_resp = MagicMock()
    chunk_1_resp.status_code = 200
    chunk_1_resp.json.return_value = {"response": "## Pontos principais\n- Point 1"}

    chunk_2_resp = MagicMock()
    chunk_2_resp.status_code = 200
    chunk_2_resp.json.return_value = {"response": "## Pontos principais\n- Point 2"}

    cons_resp = MagicMock()
    cons_resp.status_code = 200
    cons_resp.json.return_value = {"response": "- Consolidated Point"}

    unload_resp = MagicMock()
    unload_resp.status_code = 200

    mock_client = MagicMock()
    mock_client.post.side_effect = [chunk_1_resp, chunk_2_resp, cons_resp, unload_resp]
    mock_client.__enter__.return_value = mock_client

    with patch("httpx.Client", return_value=mock_client):
        result = summarize_transcript(long_transcript, max_words_per_chunk=3)
        assert "## Pontos principais" in result
        assert "- Consolidated Point" in result
        assert mock_client.post.call_count == 4



def test_summarize_http_error() -> None:
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__.return_value = mock_client

    with patch("httpx.Client", return_value=mock_client):
        with pytest.raises(RuntimeError, match="ollama request failed"):
            summarize_transcript("Some transcript content")


def test_summarize_empty_response_from_ollama() -> None:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": ""}

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__.return_value = mock_client

    with patch("httpx.Client", return_value=mock_client):
        with pytest.raises(RuntimeError, match="empty summary"):
            summarize_transcript("Some transcript content")


def test_split_transcript_giant_line() -> None:
    giant_line = "word " * 10
    chunks = split_transcript_by_words(giant_line.strip(), max_words=3)
    assert len(chunks) == 4
    assert chunks[0] == "word word word"


def test_summarize_auto_model_pull() -> None:
    # 404 response triggers pull endpoint, then succeeds on retry
    not_found_resp = MagicMock()
    not_found_resp.status_code = 404
    not_found_resp.text = "model not found"

    pull_resp = MagicMock()
    pull_resp.status_code = 200

    success_resp = MagicMock()
    success_resp.status_code = 200
    success_resp.json.return_value = {"response": "## Pontos principais\n- Pulled model summary"}

    mock_client = MagicMock()
    mock_client.post.side_effect = [not_found_resp, pull_resp, success_resp, MagicMock(status_code=200)]
    mock_client.__enter__.return_value = mock_client

    with patch("httpx.Client", return_value=mock_client):
        res = summarize_transcript("transcript for pulled model")
        assert res == "## Pontos principais\n- Pulled model summary"
        assert mock_client.post.call_count == 4


def test_unload_ollama_model_handles_exception() -> None:
    from clerk.summarize import unload_ollama_model

    mock_client = MagicMock()
    mock_client.post.side_effect = Exception("Connection error")
    mock_client.__enter__.return_value = mock_client

    with patch("httpx.Client", return_value=mock_client):
        # Should not raise exception when unloading fails
        unload_ollama_model("test_model", "http://localhost:11434")


def test_summarize_connection_error() -> None:
    import httpx

    mock_client = MagicMock()
    mock_client.post.side_effect = httpx.ConnectError("Connection refused")
    mock_client.__enter__.return_value = mock_client

    with patch("httpx.Client", return_value=mock_client):
        with pytest.raises(RuntimeError, match="Ensure Ollama is running"):
            summarize_transcript("This is a valid meeting transcript with content")


def test_summarize_auto_pull_offline_failure() -> None:
    import httpx

    not_found_resp = MagicMock()
    not_found_resp.status_code = 404
    not_found_resp.text = "model not found"

    mock_client = MagicMock()
    mock_client.post.side_effect = [not_found_resp, httpx.ConnectError("Network unreachable")]
    mock_client.__enter__.return_value = mock_client

    with patch("httpx.Client", return_value=mock_client):
        with pytest.raises(RuntimeError, match="cannot be pulled while offline"):
            summarize_transcript("This is a valid meeting transcript with content")


