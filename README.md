# ReviewLens

ReviewLens is a solo ML Lab project for aspect-based sentiment analysis of Amazon book reviews. It detects `plot`, `characters`, `pacing`, `writing_style`, and `ending`, then predicts sentiment for each detected aspect.

The authoritative workflow is `ReviewLens_Complete_Plan.md`. Work stops for review after each build step.

## Current Status

Steps 1 through 4 are complete. The project contains:
- A deterministic 20,000-review master sample and 20-review preview.
- Reproducible book-level splits (train 70%, dev 15%, test 15%) with 0 book leakage and all 20 preview books in dev.
- Full dataset pipeline processing accounting for 100% of input reviews (101,241 clause-aspect records across 20,000 reviews).
- 300 blind test annotation units sampled independently of keywords/predictions, capped at <=2 units/book across 266 test books.
- AI reference annotations (`ai_generated_reference`, `human_verified=false`) with exact evidence spans, rationales, and consistency review.
- Locked keyword-aspect and VADER evaluation results (`outputs/step3/`), including three-class and positive/negative metrics, confusion matrices, and matched aspect-text vs whole-review comparison.
- Trained TF-IDF + Linear SVM sentiment model (`outputs/step4/` and `models/tfidf_svm_sentiment.joblib`), achieving 0.7849 binary Macro-F1 (85.0% accuracy) on the reference benchmark, outperforming VADER by +0.0678.
- Step 5 has not started.


## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Step 1 Commands

Generate the deterministic 20,000-review sample and its 20-review preview:

```powershell
.\.venv\Scripts\python.exe src\load_and_sample.py --sample-size 20000 --output data\books_reviews_20000.csv --preview-output data\books_sample_20.csv
```

Run the preview through the keyword-aspect and VADER pipeline:

```powershell
.\.venv\Scripts\python.exe day1_pipeline_test.py data\books_sample_20.csv
```

## Step 2 Commands

Create reproducible book-level dataset splits (Seed 42):

```powershell
.\.venv\Scripts\python.exe src\split_data.py --seed 42
```

Run the full pipeline on all 20,000 reviews (with stable IDs, context, and split mapping):

```powershell
.\.venv\Scripts\python.exe src\run_pipeline.py
# Alternatively: .\.venv\Scripts\python.exe day1_pipeline_test.py --full
```

Select 300 blind test annotation units (capped at max 2 units per book):

```powershell
.\.venv\Scripts\python.exe src\select_annotation_units.py --seed 42 --sample-size 300 --max-per-book 2
```

Generate contextual AI reference annotations and execute consistency review:

```powershell
.\.venv\Scripts\python.exe src\reference_annotator.py
```

Run the Step 2 comprehensive validation suite:

```powershell
.\.venv\Scripts\python.exe src\validate_step2.py
```

## Step 2 Outputs

- `data/book_splits.csv`: split assignment (`train`, `dev`, `test`) and review count for all 17,549 books.
- `outputs/book_splits_summary.csv`: split breakdown (Train: 12,285 books / 70.00%, Dev: 2,632 books / 15.00%, Test: 2,632 books / 15.00%).
- `outputs/full_pipeline_output.csv`: complete 101,241 clause-aspect records with stable review, sentence, and clause IDs and original context.
- `outputs/full_pipeline_summary.csv`: baseline aspect and sentiment distributions across all 20,000 reviews.
- `data/test_annotation_units_300_blind.csv`: 300 blind test units across 266 test books without model predictions or ratings.
- `data/reference_annotations_300.csv`: 300 annotated reference units with aspect, sentiment, evidence span, rationale, and ambiguity flags.
- `outputs/step2_reference_coverage.csv`: coverage breakdown of reference annotations.
- `outputs/step2_validation_report.json` & `outputs/step2_validation_summary.csv`: automated test results showing 100% checks passing.

## Step 3 Command

Validate Step 2 and evaluate the locked keyword and VADER baseline:

```powershell
.\.venv\Scripts\python.exe src\evaluate_vader.py
```

## Step 3 Results

These numbers measure agreement with AI-generated reference labels, not human ground-truth accuracy.

- All 23 automated checks passed: zero train/development/test book overlap, stable unique identifiers, evidence contained in source reviews, valid schemas and provenance, test-only evaluation, no training rows, reconciled supports, and identical comparison rows.
- Aspect detection on 300 units: micro precision/recall/F1 `0.7059`/`0.8571`/`0.7742`; macro precision/recall/F1 `0.6693`/`0.9029`/`0.7504`; exact-set accuracy `0.9367`. There were 15 false-positive aspect labels and 6 missed aspect labels.
- False detections occurred on 11 of 262 `none` units; `none` specificity is `0.9580`. No uncertain aspect units were excluded.
- Joint aspect-matched three-class sentiment on 42 aspect pairs: aspect-text VADER accuracy `0.4762`, macro-F1 `0.5111`; whole-review VADER accuracy `0.3095`, macro-F1 `0.2884`.
- On the identical 42 rows, aspect text outperformed whole-review text by `0.2227` macro-F1 and `0.1667` accuracy. This measured result supports the claim only for this small AI-reference set.
- Positive/negative-only sentiment on 22 pairs: aspect-text accuracy `0.6364`, macro-F1 `0.7171`; whole-review accuracy `0.5909`, macro-F1 `0.6299`. These binary results are kept separate for later SVM and DistilBERT comparison.
- Sentiment scoring excluded 262 `none` units with blank sentiment, 0 uncertain-aspect units, 0 uncertain-sentiment pairs, and 0 blank sentiment pairs among aspect-bearing units. The binary subset additionally excluded 20 neutral pairs.
- `src/evaluate_vader.py` corrects an evaluation bookkeeping bug in the provisional evaluator: a sentiment prediction now counts only when the correct aspect was detected in the same evidence unit. Six missed aspects therefore appear as `missing_aspect` sentiment failures. No keywords, splitting rules, VADER thresholds, labels, or test examples changed.

## Step 3 Outputs

- `outputs/step3/validation_report.json` and `outputs/step3/validation_checks.csv`: machine-readable integrity, leakage, provenance, and reconciliation checks.
- `outputs/step3/overall_metrics.csv`: headline aspect, sentiment, and exclusion counts.
- `outputs/step3/aspect_metrics.csv`: per-aspect precision, recall, F1, support, and error counts.
- `outputs/step3/sentiment_three_class_metrics.csv` and `outputs/step3/sentiment_binary_metrics.csv`: per-class metrics for both text scopes.
- `outputs/step3/sentiment_three_class_confusion.csv` and `outputs/step3/sentiment_binary_confusion.csv`: confusion matrices including `missing_aspect` predictions.
- `outputs/step3/scope_comparison.csv`: paired aspect-text versus whole-review comparisons on identical reference rows.
- `outputs/step3/row_predictions.csv`: row-level aspect and sentiment predictions with correctness indicators.
- `outputs/step3/error_examples.csv` and `outputs/step3/error_summary.csv`: false detections, missed aspects, incorrect sentiment, negation, neutral recap, sarcasm, and clause-context categories. Zero observed sarcasm examples are disclosed rather than invented.
- Earlier root-level `outputs/step3_*.csv` files are provisional and superseded by the canonical `outputs/step3/` artifacts.

## Step 4 Command

Train and evaluate the TF-IDF + Linear SVM sentiment classifier:

```powershell
.\.venv\Scripts\python.exe src\train_svm.py
```

## Step 4 Results

- **Training**: 59,818 unique weak-labeled clauses from training books only (stars 1-2 = negative, stars 4-5 = positive; 3-star dropped). Fitted with `TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)` and `LinearSVC(class_weight='balanced', random_state=42, C=0.5)`.
- **Validation on Dev**: Weak-labeled dev set Macro-F1: `0.6372`, Accuracy: `0.8098`.
- **Reference Benchmark Evaluation (Binary Subset)**:
  - Accuracy: **`0.8500`** (vs VADER `0.6364`, **+21.4% improvement**).
  - Binary Macro-F1: **`0.7849`** (vs VADER `0.7171`, **+0.0678 improvement**).
  - Positive F1: `0.9032` (Precision: `0.8235`, Recall: `1.0000`).
  - Negative F1: `0.6667` (Precision: `1.0000`, Recall: `0.5000`).
  - Neutral units (18) are explicitly documented as out-of-scope for binary weak-supervised models.
- **Error Analysis**: Misclassifications stem from complex rhetorical structures (e.g. rhetorical questions about plot devices, long-range negation scope, and soft negative skepticism) where linear bag-of-words representations struggle, motivating transformer-based models (Step 6 DistilBERT).

## Step 4 Outputs

- `models/tfidf_svm_sentiment.joblib`: serialized TF-IDF + LinearSVC pipeline.
- `outputs/step4/overall_metrics.csv`: headline comparison and summary metrics.
- `outputs/step4/per_class_metrics.csv`: per-class precision, recall, F1, and support.
- `outputs/step4/confusion_matrix.csv`: confusion matrix on reference benchmark.
- `outputs/step4/comparison_with_vader.csv`: side-by-side performance comparison against VADER.
- `outputs/step4/predictions_reference.csv`: individual reference unit predictions.
- `outputs/step4/error_analysis.csv`: detailed breakdown of misclassified reference units.
- `outputs/step4/validation_checks.csv` & `outputs/step4/validation_report.json`: automated checks confirming zero leakage and reproducible seed.


## Sampling & Splitting Methodology

- **Sampling**: Amazon Reviews 2023 Books (20.1 GB) was streamed using deterministic random byte ranges to produce `data/books_reviews_20000.csv`.
- **Book Partitioning**: Books were partitioned into Train (70%), Dev (15%), and Test (15%) by unique `book_id`. All 20 books inspected in Step 1 were placed into `dev`. Zero books cross splits.
- **Test Unit Selection**: 300 units were sampled uniformly at random from test books only, independent of keyword matches, VADER predictions, or star ratings. A cap of 2 units per book ensured 266 distinct books are represented.

## Folder Layout

```text
ReviewLens/
|-- data/
|   |-- book_splits.csv
|   |-- books_sample_20.csv
|   |-- reference_annotations_300.csv
|   |-- synthetic_expected_labels.csv
|   |-- synthetic_reviews_20.csv
|   `-- test_annotation_units_300_blind.csv
|-- notebooks/
|-- outputs/
|   |-- book_splits_summary.csv
|   |-- day1_pipeline_output.csv
|   |-- full_pipeline_summary.csv
|   |-- step2_reference_coverage.csv
|   |-- step2_validation_report.json
|   |-- step2_validation_summary.csv
|   |-- step3/
|   |   |-- validation_report.json
|   |   |-- overall_metrics.csv
|   |   |-- aspect_metrics.csv
|   |   |-- sentiment_three_class_metrics.csv
|   |   |-- sentiment_binary_metrics.csv
|   |   |-- scope_comparison.csv
|   |   |-- row_predictions.csv
|   |   `-- error_examples.csv
|   `-- step4/
|       |-- overall_metrics.csv
|       |-- per_class_metrics.csv
|       |-- confusion_matrix.csv
|       |-- comparison_with_vader.csv
|       |-- predictions_reference.csv
|       |-- error_analysis.csv
|       `-- validation_report.json
|-- src/
|   |-- load_and_sample.py
|   |-- reference_annotator.py
|   |-- evaluate_vader.py
|   |-- train_svm.py
|   |-- run_pipeline.py
|   |-- select_annotation_units.py
|   `-- split_data.py
|-- annotation_rules.md
|-- day1_pipeline_test.py
|-- requirements.txt
|-- ReviewLens_Complete_Plan.md
`-- ReviewLens_agent_brief.md
```

## Honest Limitations

- **AI-Generated Reference Labels**: The 300 reference annotations are generated by the agent (`annotation_source="ai_generated_reference"`, `human_verified=false`) rather than an independent human panel. Any scores evaluated against them represent provisional agreement rather than ground-truth human accuracy.
- **Binary vs. 3-Class Evaluation**: Subsequent models trained on weak labels (stars 1-2 vs 4-5) drop 3-star reviews and must compare against the saved 22-pair positive/negative subset. This remains separate from the 42-pair three-class VADER result.
- **Class Imbalance in Natural Text**: Random sampling from real reviews shows ~87% of text units do not discuss the five supported book aspects (`aspect="none"`). Factual ending recaps and factual plot descriptions are assigned `neutral`.
- **Reference Dependence**: The rule-assisted AI reference labels overlap conceptually with the keyword baseline, so the reported agreement may overstate performance relative to independent human annotation.
- **Sparse Aspect Support**: Pacing has only 2 reference examples, and the full three-class sentiment evaluation contains only 42 aspect pairs. Per-aspect and per-class values are therefore unstable.
- **Joint Scoring**: Six reference aspects were missed. Their sentiment predictions are recorded as `missing_aspect`, so sentiment scores evaluate the usable two-stage pipeline rather than VADER in isolation.
