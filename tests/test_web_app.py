from pathlib import Path

import joblib
import pytest

from app import create_app, DEFAULT_MODEL


@pytest.fixture(scope="module")
def client():
    if not DEFAULT_MODEL.is_file():
        pytest.skip("Train the local model before running the real-model web smoke test")
    app = create_app(joblib.load(DEFAULT_MODEL))
    app.config.update(TESTING=True)
    return app.test_client()


def test_empty_article_is_rejected(client):
    response = client.post("/", data={"title": "Headline", "article": "   "})
    assert response.status_code == 400
    assert b"Paste article text" in response.data


def test_overlong_article_is_rejected(client):
    response = client.post("/", data={"article": "x" * 50_001})
    assert response.status_code == 413
    assert b"50,000 characters" in response.data


def test_real_model_prediction_and_explanation(client):
    response = client.post("/", data={
        "title": "Local model demonstration",
        "article": "This is a short article about policy, reporting, and public records.",
    })
    assert response.status_code == 200
    assert b"DATASET LABEL" in response.data
    assert b"model score" in response.data
    assert b"not a credibility verdict" in response.data


def test_missing_model_has_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_NEWS_MODEL", str(tmp_path / "missing.joblib"))
    with pytest.raises(FileNotFoundError, match="Run the evaluation command first"):
        create_app()
