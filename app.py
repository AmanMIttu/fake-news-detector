"""Local editorial-style interface for the saved Fake/Real text model."""
from __future__ import annotations

import os
from pathlib import Path

import joblib
from flask import Flask, render_template, request

from fake_news.pipeline import predict_texts

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "fake_news_pipeline.joblib"
MAX_ARTICLE_CHARS = 50_000


def explain_prediction(model, text: str, label: str, limit: int = 6) -> tuple[list[dict], list[dict]]:
    """Return local TF-IDF/logistic-regression term contributions for one label."""
    try:
        vectorizer = model.named_steps["tfidf"]
        classifier = model.named_steps["classifier"]
        vector = vectorizer.transform([text]).tocoo()
        names = vectorizer.get_feature_names_out()
        if not vector.nnz:
            return [], []
        # For binary sklearn logistic regression, coef_[0] is oriented toward
        # classes_[1]. Flip it when explaining classes_[0].
        direction = 1.0 if str(classifier.classes_[1]) == label else -1.0
        contributions = vector.data * classifier.coef_[0, vector.col] * direction
        terms = [
            {"term": str(names[index]), "weight": float(weight)}
            for index, weight in zip(vector.col, contributions)
        ]
        supports = sorted((item for item in terms if item["weight"] > 0),
                          key=lambda item: item["weight"], reverse=True)[:limit]
        opposes = sorted((item for item in terms if item["weight"] < 0),
                         key=lambda item: item["weight"])[:limit]
        return supports, opposes
    except (KeyError, AttributeError, IndexError, ValueError):
        return [], []


def create_app(model=None) -> Flask:
    app = Flask(__name__)
    model_path = Path(os.environ.get("FAKE_NEWS_MODEL", DEFAULT_MODEL))
    if model is None:
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Saved model not found at {model_path}. Run the evaluation command first."
            )
        model = joblib.load(model_path)

    @app.get("/")
    def index():
        return render_template("index.html", result=None, error=None, title="", article="")

    @app.post("/")
    def analyze():
        title = request.form.get("title", "").strip()
        article = request.form.get("article", "").strip()
        if not article:
            return render_template(
                "index.html", result=None, error="Paste article text to analyze.",
                title=title, article=article,
            ), 400
        if len(article) > MAX_ARTICLE_CHARS or len(title) > MAX_ARTICLE_CHARS:
            return render_template(
                "index.html", result=None,
                error=f"Keep the title and article under {MAX_ARTICLE_CHARS:,} characters each.",
                title=title, article=article,
            ), 413
        model_text = f"{title} {article}".strip()
        prediction = predict_texts(model, [model_text])[0]
        label = prediction["label"]
        scores = prediction["class_scores"]
        supporting, opposing = explain_prediction(model, model_text, label)
        result = {
            "label": label,
            "scores": scores,
            "supporting_terms": supporting,
            "opposing_terms": opposing,
            "word_count": len(article.split()),
            "character_count": len(article),
        }
        return render_template(
            "index.html", result=result, error=None, title=title, article=article,
        )

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)
