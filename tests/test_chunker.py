from app.services.chunker import build_paper_text, clean_page_text


# ── clean_page_text ───────────────────────────────────────────────────────────

def test_clean_empty_string_returns_empty():
    assert clean_page_text("") == ""


def test_clean_removes_nul_bytes():
    assert clean_page_text("hello\x00world") == "helloworld"


def test_clean_normalises_crlf_to_lf():
    result = clean_page_text("line1\r\nline2\rline3")
    assert "\r" not in result
    assert "line1" in result
    assert "line2" in result
    assert "line3" in result


def test_clean_collapses_multiple_blank_lines_to_one():
    result = clean_page_text("para1\n\n\n\npara2")
    assert "\n\n\n" not in result
    assert "para1" in result
    assert "para2" in result


def test_clean_strips_leading_and_trailing_whitespace():
    assert clean_page_text("  hello  ") == "hello"


def test_clean_strips_whitespace_inside_lines():
    result = clean_page_text("  first line  \n  second line  ")
    assert "first line" in result
    assert "second line" in result


def test_clean_preserves_plain_content_unchanged():
    text = "This is a clean sentence."
    assert clean_page_text(text) == text


def test_clean_whitespace_only_lines_treated_as_blank():
    # Lines with only spaces collapse the same as empty lines
    result = clean_page_text("a\n   \n   \nb")
    assert "\n\n\n" not in result
    assert "a" in result
    assert "b" in result


# ── build_paper_text ──────────────────────────────────────────────────────────

def _paper(pages: list[str]) -> dict:
    """Minimal paper record with a list of page texts."""
    return {
        "paper_id": "paper_001",
        "file_name": "test.pdf",
        "file_path": "data/raw/test.pdf",
        "total_pages_loaded": len(pages),
        "pages": [{"text": t} for t in pages],
    }


def test_build_single_page_returns_its_text():
    assert build_paper_text(_paper(["Hello world"])) == "Hello world"


def test_build_multiple_pages_joined_with_double_newline():
    result = build_paper_text(_paper(["Page one", "Page two"]))
    assert "Page one" in result
    assert "Page two" in result
    assert "\n\n" in result


def test_build_empty_pages_are_skipped():
    result = build_paper_text(_paper(["Good content", "", "More content"]))
    assert "Good content" in result
    assert "More content" in result


def test_build_all_empty_pages_returns_empty_string():
    assert build_paper_text(_paper(["", "   ", "\n\n"])) == ""


def test_build_whitespace_only_page_is_skipped():
    result = build_paper_text(_paper(["real content", "   "]))
    assert result == "real content"


def test_build_no_pages_returns_empty_string():
    assert build_paper_text(_paper([])) == ""


def test_build_preserves_page_order():
    result = build_paper_text(_paper(["First", "Second", "Third"]))
    assert result.index("First") < result.index("Second") < result.index("Third")
