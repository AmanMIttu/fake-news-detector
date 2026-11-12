"""Dataset loading, leakage-aware evaluation, plots, and inference helpers."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

LABELS = ("Fake", "Real")
REQUIRED_COLUMNS = {"title", "text", "subject", "date"}
SEED = 42


def normalize_text(value: object) -> str:
    """Normalize case and whitespace for duplicate detection only."""
    return re.sub(r"\s+", " ", str(value).casefold()).strip()


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load original Kaggle files, derive labels from filenames, and deduplicate."""
    root = Path(path)
    pieces = []
    file_audit = {}
    for filename, label in (("Fake.csv", "Fake"), ("True.csv", "Real")):
        csv_path = root / filename
        if not csv_path.is_file():
            raise FileNotFoundError(f"Expected dataset file: {csv_path}")
        frame = pd.read_csv(csv_path)
        missing_columns = REQUIRED_COLUMNS - set(frame.columns)
        if missing_columns:
            raise ValueError(f"{filename} is missing columns: {sorted(missing_columns)}")
        file_audit[filename] = {
            "rows": int(len(frame)),
            "columns": list(frame.columns),
            "missing_values": {str(k): int(v) for k, v in frame.isna().sum().items()},
            "exact_duplicate_rows": int(frame.duplicated().sum()),
        }
        frame["label"] = label
        frame["article_group"] = frame["text"].fillna("").map(normalize_text)
        frame["model_text"] = (
            frame["title"].fillna("").astype(str) + " " +
            frame["text"].fillna("").astype(str)
        ).str.strip()
        pieces.append(frame[["model_text", "article_group", "label"]])

    raw = pd.concat(pieces, ignore_index=True)
    raw_rows = int(len(raw))
    empty_rows = int(raw["model_text"].map(normalize_text).eq("").sum())
    raw = raw.loc[raw["model_text"].map(normalize_text).ne("")].copy()
    raw["_dedupe_key"] = raw["model_text"].map(normalize_text)
    label_counts_by_text = raw.groupby("_dedupe_key")["label"].nunique()
    conflicts = int(label_counts_by_text.gt(1).sum())
    if conflicts:
        raise ValueError(
            f"Found {conflicts} normalized title/text duplicates with conflicting labels; "
            "resolve them before training."
        )
    duplicate_model_text_rows = int(raw["_dedupe_key"].duplicated().sum())
    data = raw.drop_duplicates("_dedupe_key", keep="first").copy()
    data.rename(columns={"model_text": "text"}, inplace=True)
    raw_class_counts = raw["label"].value_counts().sort_index()
    retained_class_counts = data["label"].value_counts().sort_index()
    audit = {
        "files": file_audit,
        "source_rows": raw_rows,
        "empty_text_rows_removed": empty_rows,
        "normalized_title_text_duplicate_rows_removed": duplicate_model_text_rows,
        "retained_unique_model_text_rows": int(len(data)),
        "raw_class_counts": {str(k): int(v) for k, v in raw_class_counts.items()},
        "retained_class_counts": {str(k): int(v) for k, v in retained_class_counts.items()},
        "conflicting_normalized_title_text_groups": conflicts,
        "label_source": "Fake.csv -> Fake; True.csv -> Real",
        "model_input": "title + space + text",
    }
    result = data[["text", "label", "article_group"]].reset_index(drop=True)
    result.attrs["dataset_audit"] = audit
    return result


def make_model() -> Pipeline:
    """Create TF-IDF + logistic regression; TF-IDF is fit inside the pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english", max_features=100_000,
            ngram_range=(1, 2), min_df=2, sublinear_tf=True,
        )),
        ("classifier", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=SEED,
        )),
    ])


def _save_plots(output_dir: Path, y_test, predictions, fake_scores, classes) -> None:
    """Write confusion matrix, ROC, and precision-recall plots for held-out rows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    output_dir.mkdir(parents=True, exist_ok=True)
    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, labels=classes, display_labels=classes,
        cmap="Blues", values_format="d",
    )
    plt.title("Fake News Classifier — Held-out Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close()

    y_fake = (y_test == "Fake").astype(int)
    fpr, tpr, _ = roc_curve(y_fake, fake_scores)
    auc = roc_auc_score(y_fake, fake_scores)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Fake vs Real — Held-out Test ROC")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=160)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_fake, fake_scores)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall (Fake)")
    plt.ylabel("Precision (Fake)")
    plt.title("Fake Class — Held-out Test Precision–Recall")
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_curve.png", dpi=160)
    plt.close()


def evaluate(
    data: pd.DataFrame,
    output_dir: str | Path,
    seed: int = SEED,
    model_path: str | Path | None = None,
) -> dict:
    """Fit on four grouped folds and report exactly once on the fifth fold."""
    if not {"text", "label"}.issubset(data.columns):
        raise ValueError("data must include text and label columns")
    if set(data["label"].astype(str).unique()) != set(LABELS):
        raise ValueError(f"labels must be exactly {LABELS}")

    clean = data.copy()
    clean["text"] = clean["text"].fillna("").astype(str)
    clean["label"] = clean["label"].astype(str)
    clean = clean.loc[clean["text"].map(normalize_text).ne("")].copy()
    if "article_group" not in clean:
        clean["article_group"] = clean["text"].map(normalize_text)
    else:
        clean["article_group"] = clean["article_group"].fillna("").map(normalize_text)
    clean = clean.reset_index(drop=True)

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_indices, test_indices = next(splitter.split(
        clean["text"], clean["label"], groups=clean["article_group"]
    ))
    train = clean.iloc[train_indices]
    test = clean.iloc[test_indices]
    overlapping_groups = set(train["article_group"]) & set(test["article_group"])
    if overlapping_groups:
        raise RuntimeError("Article groups overlap between train and test")

    model = make_model()
    model.fit(train["text"], train["label"])
    predictions = model.predict(test["text"])
    probabilities = model.predict_proba(test["text"])
    classes = list(model.classes_)
    fake_index = classes.index("Fake")
    fake_scores = probabilities[:, fake_index]
    y_test = test["label"]

    metrics = {
        "seed": int(seed),
        "split": "first fold of shuffled StratifiedGroupKFold (5 folds)",
        "grouping": "normalized article body; repeated title+body model inputs deduplicated",
        "test_is_used_for_model_selection": False,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_groups": int(train["article_group"].nunique()),
        "test_groups": int(test["article_group"].nunique()),
        "train_test_group_overlap": int(len(overlapping_groups)),
        "dataset_audit": data.attrs.get("dataset_audit", {}),
        "class_counts_after_deduplication": {
            str(k): int(v) for k, v in clean["label"].value_counts().sort_index().items()
        },
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_fake": float(precision_score(y_test, predictions, pos_label="Fake", zero_division=0)),
        "recall_fake": float(recall_score(y_test, predictions, pos_label="Fake", zero_division=0)),
        "f1_fake": float(f1_score(y_test, predictions, pos_label="Fake", zero_division=0)),
        "precision_macro": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "roc_auc_fake_vs_real": float(roc_auc_score((y_test == "Fake").astype(int), fake_scores)),
        "confusion_matrix_labels": classes,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=classes).tolist(),
        "class_report": classification_report(
            y_test, predictions, labels=classes, output_dict=True, zero_division=0,
        ),
        "model_configuration": {
            "features": "title + body",
            "vectorizer": "TfidfVectorizer(stop_words='english', max_features=100000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)",
            "classifier": "LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)",
        },
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    target_model = Path(model_path) if model_path else out / "fake_news_pipeline.joblib"
    target_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target_model)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _save_plots(out, y_test, predictions, fake_scores, classes)
    return metrics


def predict_texts(model: Pipeline, texts: Iterable[str]) -> list[dict]:
    """Return labels and model scores; scores are not calibrated truth probabilities."""
    items = [str(text) for text in texts]
    if not items:
        return []
    labels = model.predict(items)
    scores = model.predict_proba(items)
    return [
        {"label": str(label), "class_scores": dict(zip(model.classes_, row.tolist()))}
        for label, row in zip(labels, scores)
    ]
