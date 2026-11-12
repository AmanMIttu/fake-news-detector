# Dataset audit and evaluation report

## Dataset verification

The files `data/raw/Fake.csv` and `data/raw/True.csv` were found, read successfully with pandas, and are consistent with the source named in the original notebook: [Fake and Real News Dataset on Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset). The notebook's Kaggle CLI cell uses the same dataset slug. Both files have exactly the expected four columns: `title`, `text`, `subject`, and `date`. File-derived labels match the dataset's two-file convention (`Fake.csv` -> Fake; `True.csv` -> Real).

| File | Rows | Missing values | Exact full-row duplicates | Repeated `title` values | Repeated `text` values |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Fake.csv` | 23,481 | 0 | 3 | 5,578 | 6,026 |
| `True.csv` | 21,417 | 0 | 206 | 591 | 225 |
| **Total** | **44,898** | **0** | **209** | **6,169** | **6,251** |

The loader concatenates title and body with one space. Across the full inputs, 5,798 rows repeat an earlier normalized model input after case-folding and whitespace normalization; no repeated model input has contradictory labels. Those rows are removed, leaving 39,100 unique inputs (Fake 17,905; Real 21,195). The two CSVs are structurally and statistically consistent with the notebook's specified dataset. Since no trusted Kaggle checksum or signed manifest was supplied, their exact byte-level provenance cannot be independently authenticated.

SHA-256 checksums of the supplied files:

- `Fake.csv`: `BEBF8BCFE95678BF2C732BF413A2CE5F621AF0102C82BF08083B2E5D3C693D0C`
- `True.csv`: `BA0844414A65DC6AE7402B8EEE5306DA24B6B56488D6767135AF466C7DCB2775`

Attribution: Kaggle dataset **Fake and Real News Dataset**, account `clmentbisaillon`; source: <https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset>. Follow the dataset's terms for use and redistribution.

## Evaluation design

- Input features: `title + text`; `subject` and `date` are not used.
- Model: English-stopword unigram/bigram TF-IDF (`max_features=100000`, `min_df=2`, `sublinear_tf=True`) followed by balanced logistic regression (`max_iter=1000`).
- Seed: 42.
- Split: first fold of shuffled `StratifiedGroupKFold(n_splits=5)`, grouping rows by normalized article body. The 5,798 repeated normalized title/body inputs are removed before the split. The train/test article-body group overlap was measured as zero.
- Train fold: 31,726 rows; test fold: 7,374 rows. Group constraints make the fold smaller than an exact 80/20 row split.
- The model and vectorizer are fit only on the training fold. The test fold is not used for model selection or threshold tuning.

This is a single fixed baseline, not a tuned model comparison. The model was run on the original supplied files in this repository environment.

## Held-out test results

| Class / average | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Fake | 0.9933 | 0.9844 | 0.9888 | 3,330 |
| Real | 0.9872 | 0.9946 | 0.9909 | 4,044 |
| Macro average | 0.9903 | 0.9895 | 0.9899 | 7,374 |
| **Accuracy** |  |  | **0.9900** | **7,374** |

Fake-vs-Real ROC-AUC: **0.9994**.

Confusion matrix, rows actual and columns predicted, label order Fake then Real:

|  | Predicted Fake | Predicted Real |
| --- | ---: | ---: |
| Actual Fake | 3,278 | 52 |
| Actual Real | 22 | 4,022 |

Artifacts from this run are `results/reproduced/metrics.json`, `confusion_matrix.png`, `roc_curve.png`, and `precision_recall_curve.png`. The model is saved locally as ignored `models/fake_news_pipeline.joblib`.

## Historical notebook comparison

The historical README claims Random Forest F1 approximately 0.996, but it provides no split, artifacts, or reproducible command supporting that number. The saved notebook also contains a cell that reloads the full processed dataset, predicts those same 38,646 rows, and prints a report with 0.47 accuracy. That is a full-data prediction report, not an independent test result. Its output uses numeric labels and its model/label mapping cannot be reconstructed from the files in this checkout. The old F1 claim and that saved report disagree and neither can be compared as a valid holdout to the current results.

The new 0.9900 held-out accuracy comes from a different, explicit pipeline and a deduplicated, article-body-grouped test fold. Differences in preprocessing, label handling, model, row selection, and split could explain the gap, but the old processed dataset and source implementation are unavailable, so the cause cannot be isolated. No claim is made that this run reproduces the historical Random Forest result.

## Reproduction

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install pytest
$env:PYTHONPATH = "$PWD\src"
python -m fake_news.cli evaluate --data data/raw --output results/reproduced --model-output models/fake_news_pipeline.joblib
python -m fake_news.cli predict "Example article text" --model models/fake_news_pipeline.joblib
python -m pytest -q
```

The result scores describe dataset label prediction, not real-world truth detection. Source and time confounding, story syndication, and near duplicates remain; a time/source-held-out evaluation would be stronger evidence of deployment generalization.

## User interface

`app.py` serves a local Flask article-analysis workspace that loads the already-saved model. It displays the file-derived Fake/Real label, model scores, and signed TF-IDF/logistic-regression term contributions. Input validation covers empty and overlong articles. This interface does not retrain, change evaluation outputs, persist article history, fetch URLs, or establish article truth. Launch it with `python app.py` after installing requirements and generating the ignored model artifact.
