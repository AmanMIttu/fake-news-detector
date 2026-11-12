# Fake News Detection with TF-IDF

An educational text classifier trained on the Kaggle **Fake and Real News Dataset**. It learns the dataset's file-derived labels; it does not establish whether a real-world claim is true.

## Dataset and attribution

The supplied files are under `data/raw/` and match the original notebook's Kaggle reference, [clmentbisaillon/fake-and-real-news-dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset). Both CSVs are readable and have the expected `title`, `text`, `subject`, and `date` columns. The loader uses `title` and `text`; it assigns `Fake` to rows in `Fake.csv` and `Real` to rows in `True.csv`.

| Source file | Raw rows | Label | Missing values | Exact duplicate rows |
| --- | ---: | --- | ---: | ---: |
| `Fake.csv` | 23,481 | Fake | 0 | 3 |
| `True.csv` | 21,417 | Real | 0 | 206 |
| **Total** | **44,898** |  | **0** | **209** |

Repeated normalized `title + text` inputs account for 5,798 additional rows. The loader removes those repeated model inputs before splitting; it found no such repeated input with conflicting labels. This leaves 39,100 unique model inputs: 17,905 Fake and 21,195 Real. `subject` and `date` are preserved in the source files but are not model features.

The files are consistent with the named Kaggle dataset by the supplied download provenance, file names, schema, and row counts. A checksum for the Kaggle-hosted copy was not available for comparison, so byte-for-byte source authenticity is not independently established. The data is attributed to the Kaggle dataset account `clmentbisaillon`; check Kaggle's dataset terms before redistribution. The repository has no separate license file.

## Method and evaluation

The reproducible baseline concatenates `title` and `text`, uses English-stopword unigram/bigram TF-IDF, then balanced logistic regression. TF-IDF is fitted inside a scikit-learn pipeline on training rows only. The fixed model is evaluated with the first fold from shuffled, stratified five-fold group splitting (seed 42); normalized article bodies are groups, so repeated bodies cannot cross from train to test. The held-out fold contains 7,374 rows and the training fold 31,726. There is no hyperparameter or threshold selection on the test fold.

### Held-out test results

| Class / average | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Fake | 0.9933 | 0.9844 | 0.9888 | 3,330 |
| Real | 0.9872 | 0.9946 | 0.9909 | 4,044 |
| Macro average | 0.9903 | 0.9895 | 0.9899 | 7,374 |
| **Accuracy** |  |  | **0.9900** | **7,374** |

Fake-vs-Real ROC-AUC: **0.9994**.

Confusion matrix (rows = actual; columns = predicted; order = Fake, Real):

|  | Predicted Fake | Predicted Real |
| --- | ---: | ---: |
| Actual Fake | 3,278 | 52 |
| Actual Real | 22 | 4,022 |

These are measured held-out results for this dataset and split, not estimates of factual accuracy in current news. The corpus is historical and contains publisher, topic, and style signals; duplicates and near-duplicates beyond exact normalized article-body groups, temporal shift, and source shift may still affect generalization.

The generated [confusion matrix](results/reproduced/confusion_matrix.png), [ROC curve](results/reproduced/roc_curve.png), [precision-recall curve](results/reproduced/precision_recall_curve.png), and full [metrics JSON](results/reproduced/metrics.json) are stored under `results/reproduced/`. The saved model is written to ignored `models/fake_news_pipeline.joblib`.

## Historical notebook results

`FakeNews_DKAI_Project.ipynb` is preserved as the original exploratory notebook, but it contains stale Colab paths and refers to missing `src/` scripts in its older saved cells. A saved notebook cell predicts over all 38,646 processed rows and reports 0.47 accuracy; that is not a held-out score. The original README's approximately 0.996 F1 claim is unsupported by a reproducible test split and conflicts with that saved output. Neither historical figure is directly comparable to this grouped holdout because the old processed data, label mapping, split, and model provenance are unavailable.

## Setup and commands (Windows PowerShell)

The pinned dependencies and test suite have been verified with Python 3.13. From the repository root:

```powershell
python --version  # confirm Python 3.13
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pytest
$env:PYTHONPATH = "$PWD\src"
python -m fake_news.cli evaluate --data data/raw --output results/reproduced --model-output models/fake_news_pipeline.joblib
python -m fake_news.cli predict "Example article text" --model models/fake_news_pipeline.joblib
python -m pytest -q
```

To evaluate the supplied Kaggle files again, use the same command and seed (42). The CLI writes `metrics.json` and plots to `results/reproduced/`, and the fitted pipeline to ignored `models/`. `predict` returns a model label and class scores. These scores are not calibrated probabilities or a determination of truth.

## Signal / Noise article workspace

The project includes a local Flask interface with an editorial article desk, headline and article inputs, Fake/Real dataset-label readout, score bars, word-count details, and local TF-IDF/logistic-regression term contributions. Empty and overlong submissions receive clear validation messages. It does not fetch URLs, store analysis history, or verify facts; submitted text is processed locally by the saved model.

Start the interface after installing dependencies and training the model once:

```powershell
python -m pip install -r requirements.txt
$env:PYTHONPATH = "$PWD\src"
python -m fake_news.cli evaluate --data data/raw --output results/reproduced --model-output models/fake_news_pipeline.joblib
python app.py
```

Then open `http://127.0.0.1:5000`. For a local prediction endpoint check without a browser, use `python -m fake_news.cli predict "Example article text" --model models/fake_news_pipeline.joblib`. The interface reads the same saved pipeline and does not retrain or alter the recorded evaluation. A screenshot is not included; run the local page to see the live interface.

## Repository structure

- `FakeNews_DKAI_Project.ipynb` — preserved original exploration; not the reproducible runner.
- `src/fake_news/pipeline.py` — data loading/audit, deduplication, grouped split, metrics, plots, inference helpers.
- `src/fake_news/cli.py` — evaluation and prediction commands.
- `data/raw/` — locally supplied original CSVs; excluded from Git.
- `results/reproduced/` — measured test metrics and plots.
- `results/`, `explainability/` — original historical plots and LIME HTML; their original model/split provenance remains unknown.
- `tests/` — loader, duplicate-label, evaluation-artifact, and inference tests.

## Limitations

A high score on this historical dataset does not mean the model can verify claims. File labels may encode outlet, topic, date, and writing-style differences. Grouping exact article bodies reduces direct duplicate leakage, but it does not prevent syndicated or near-duplicate stories from crossing folds. The model is not calibrated, does not use external evidence, and should not be used for consequential fact-checking. The local LIME artifact explains one old model example and is not an explanation of this new model.

## License

No project license file is present. Original project materials credit Shaik Aman. Dataset attribution: [Fake and Real News Dataset on Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset), account `clmentbisaillon`.
