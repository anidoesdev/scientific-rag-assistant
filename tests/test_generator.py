from unittest.mock import MagicMock, patch
from conftest import SAMPLE_CHUNKS
from app.services.generator import build_context, build_prompt, generate_answer


# ── build_context ─────────────────────────────────────────────────────────────

def test_build_context_numbers_sources_from_one():
    context = build_context(SAMPLE_CHUNKS)
    assert "[Source 1]" in context
    assert "[Source 2]" in context
    assert "[Source 3]" in context


def test_build_context_includes_chunk_text():
    context = build_context(SAMPLE_CHUNKS)
    for chunk in SAMPLE_CHUNKS:
        assert chunk["text"] in context


def test_build_context_includes_paper_id():
    context = build_context(SAMPLE_CHUNKS)
    assert "paper_001" in context
    assert "paper_002" in context


def test_build_context_empty_chunks():
    assert build_context([]) == ""


# ── build_prompt ──────────────────────────────────────────────────────────────

def test_build_prompt_includes_question():
    prompt = build_prompt("What is self-attention?", SAMPLE_CHUNKS)
    assert "What is self-attention?" in prompt


def test_build_prompt_includes_source_context():
    prompt = build_prompt("What is self-attention?", SAMPLE_CHUNKS)
    assert "[Source 1]" in prompt


# ── generate_answer ───────────────────────────────────────────────────────────

def test_generate_answer_maps_used_sources_to_citations():
    mock_response = MagicMock()
    mock_response.answer = "Attention focuses on relevant parts [1]."
    mock_response.used_sources = [1]
    mock_response.unsupported = False

    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.return_value = mock_response

    with patch("app.services.generator.ChatOpenAI", return_value=mock_llm):
        result = generate_answer("What is attention?", SAMPLE_CHUNKS)

    assert result["unsupported"] is False
    assert len(result["citations"]) == 1
    assert result["citations"][0]["source_number"] == 1
    assert result["citations"][0]["chunk_id"] == SAMPLE_CHUNKS[0]["chunk_id"]


def test_generate_answer_marks_unsupported_when_flag_set():
    mock_response = MagicMock()
    mock_response.answer = "I cannot answer."
    mock_response.used_sources = []
    mock_response.unsupported = True

    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.return_value = mock_response

    with patch("app.services.generator.ChatOpenAI", return_value=mock_llm):
        result = generate_answer("Unanswerable question?", SAMPLE_CHUNKS)

    assert result["unsupported"] is True
    assert result["citations"] == []


def test_generate_answer_ignores_out_of_range_source_numbers():
    mock_response = MagicMock()
    mock_response.answer = "Answer citing source 99."
    mock_response.used_sources = [99]  # out of range
    mock_response.unsupported = False

    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.return_value = mock_response

    with patch("app.services.generator.ChatOpenAI", return_value=mock_llm):
        result = generate_answer("test", SAMPLE_CHUNKS)

    assert result["citations"] == []
    assert result["unsupported"] is True  # no valid citations → unsupported


def test_generate_answer_deduplicates_source_numbers():
    mock_response = MagicMock()
    mock_response.answer = "Answer."
    mock_response.used_sources = [1, 1, 1]
    mock_response.unsupported = False

    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.return_value = mock_response

    with patch("app.services.generator.ChatOpenAI", return_value=mock_llm):
        result = generate_answer("test", SAMPLE_CHUNKS)

    assert len(result["citations"]) == 1
