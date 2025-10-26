import pathlib
import sys

import requests

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from arxiv_analysis import fetch_arxiv_papers, get_paper_contexts


def test_fetch_arxiv_papers_timeout_returns_error(monkeypatch):
    """fetch_arxiv_papers should return structured error data on timeout."""

    def fake_get(*args, **kwargs):
        raise requests.exceptions.Timeout("timed out")

    monkeypatch.setattr("arxiv_analysis.requests.get", fake_get)

    result = fetch_arxiv_papers()

    assert isinstance(result, dict)
    assert result.get("error") is True
    assert result.get("type") == "timeout"
    assert "timed out" in result.get("details", "") or "timed out" in result.get("message", "")


def test_get_paper_contexts_handles_timeout(monkeypatch):
    """get_paper_contexts should capture timeout errors per paper."""

    def fake_get(*args, **kwargs):
        raise requests.exceptions.Timeout("download timed out")

    monkeypatch.setattr("arxiv_analysis.requests.get", fake_get)

    sample_paper = {
        "id": "1234.56789",
        "title": "Sample Paper",
        "pdf_link": "https://arxiv.org/pdf/1234.56789.pdf",
    }

    contexts = get_paper_contexts([sample_paper])

    assert len(contexts) == 1
    context = contexts[0]
    assert context.get("title") == sample_paper["title"]
    assert context.get("error", {}).get("type") == "timeout"
    assert "timed out" in context.get("excerpt", "")
