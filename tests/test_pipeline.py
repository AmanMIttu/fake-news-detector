import json

import pandas as pd
import pytest

from fake_news.pipeline import evaluate, load_dataset, make_model, predict_texts


def write_source_files(path, fake_texts=None, real_texts=None):
    fake_texts = fake_texts or ["fake body"]
    real_texts = real_texts or ["real body"]
    for filename, label, texts in (
        ("Fake.csv", "fake", fake_texts), ("True.csv", "real", real_texts),
    ):
        pd.DataFrame({
            "title": ["shared title"] * len(texts),
            "text": texts,
            "subject": ["news"] * len(texts),
            "date": ["today"] * len(texts),
        }).to_csv(path / filename, index=False)


def test_dataset_schema_labels_and_duplicate_audit(tmp_path):
    write_source_files(tmp_path, ["repeat", "repeat", "unique fake"], ["real one", "real two"])
    data = load_dataset(tmp_path)
    assert data.label.value_counts().to_dict() == {"Fake": 2, "Real": 2}
    assert data.attrs["dataset_audit"]["source_rows"] == 5
    assert data.attrs["dataset_audit"]["normalized_title_text_duplicate_rows_removed"] == 1
    assert data.attrs["dataset_audit"]["files"]["Fake.csv"]["columns"] == [
        "title", "text", "subject", "date",
    ]


def test_missing_dataset_file_is_clear(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path)


def test_conflicting_duplicate_model_text_is_rejected(tmp_path):
    write_source_files(tmp_path, ["same article"], ["same article"])
    with pytest.raises(ValueError, match="conflicting labels"):
        load_dataset(tmp_path)


def test_grouped_evaluate_saves_metrics_plots_model_and_inference(tmp_path):
    texts = [f"fake report evidence sample {i}" for i in range(20)]
    texts += [f"real report evidence sample {i}" for i in range(20)]
    labels = ["Fake"] * 20 + ["Real"] * 20
    data = pd.DataFrame({"text": texts, "label": labels, "article_group": texts})
    metrics = evaluate(data, tmp_path / "results", model_path=tmp_path / "model.joblib")
    assert metrics["train_rows"] + metrics["test_rows"] == 40
    assert metrics["train_test_group_overlap"] == 0
    assert metrics["test_is_used_for_model_selection"] is False
    saved = json.loads((tmp_path / "results" / "metrics.json").read_text(encoding="utf-8"))
    assert saved["confusion_matrix"] == metrics["confusion_matrix"]
    assert (tmp_path / "model.joblib").is_file()
    for name in ("confusion_matrix.png", "roc_curve.png", "precision_recall_curve.png"):
        assert (tmp_path / "results" / name).is_file()
    model = make_model().fit(texts, labels)
    assert len(predict_texts(model, ["example article"])) == 1
    assert predict_texts(model, []) == []
