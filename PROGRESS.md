# ReviewLens Progress

Status and reproducibility notes for the seven build steps in
[ReviewLens_Complete_Plan.md](ReviewLens_Complete_Plan.md). Commands below are
PowerShell commands run from the repository root. They may require the
dependencies in `requirements.txt` and, for Step 1, access to the source dataset.

## Step 1 — Loader and sampler

**Status: done**

**Files and artifacts**
- [src/load_and_sample.py](src/load_and_sample.py)
- [day1_pipeline_test.py](day1_pipeline_test.py)
- [data/books_reviews_20000.csv](data/books_reviews_20000.csv)
- [data/books_sample_20.csv](data/books_sample_20.csv)
- [outputs/books_reviews_20000_summary.csv](outputs/books_reviews_20000_summary.csv)
- [outputs/books_reviews_20000_rating_distribution.csv](outputs/books_reviews_20000_rating_distribution.csv)
- [outputs/books_reviews_20000_reviews_per_book.csv](outputs/books_reviews_20000_reviews_per_book.csv)
- [outputs/day1_pipeline_output.csv](outputs/day1_pipeline_output.csv)

**Reproduce**
```powershell
.\.venv\Scripts\python.exe src\load_and_sample.py --sample-size 20000 --output data\books_reviews_20000.csv --preview-output data\books_sample_20.csv
.\.venv\Scripts\python.exe day1_pipeline_test.py data\books_sample_20.csv
```

**Deviations / notes**
- The sampler reads deterministic random byte ranges rather than downloading and
  shuffling the whole source file. This is not a first-N-row sample, but the
  byte-range method can introduce text-length bias, as the plan notes.

## Step 2 — Full run and reference annotation

**Status: done**

**Files and artifacts**
- [src/split_data.py](src/split_data.py)
- [src/run_pipeline.py](src/run_pipeline.py)
- [src/select_annotation_units.py](src/select_annotation_units.py)
- [src/reference_annotator.py](src/reference_annotator.py)
- [src/validate_step2.py](src/validate_step2.py)
- [data/book_splits.csv](data/book_splits.csv)
- [outputs/book_splits_summary.csv](outputs/book_splits_summary.csv)
- [outputs/full_pipeline_output.csv](outputs/full_pipeline_output.csv)
- [outputs/full_pipeline_summary.csv](outputs/full_pipeline_summary.csv)
- [data/test_annotation_units_300_blind.csv](data/test_annotation_units_300_blind.csv)
- [data/reference_annotations_300.csv](data/reference_annotations_300.csv)
- [outputs/step2_reference_coverage.csv](outputs/step2_reference_coverage.csv)
- [outputs/step2_validation_report.json](outputs/step2_validation_report.json)
- [outputs/step2_validation_summary.csv](outputs/step2_validation_summary.csv)

**Reproduce**
```powershell
.\.venv\Scripts\python.exe src\split_data.py --seed 42
.\.venv\Scripts\python.exe src\run_pipeline.py
.\.venv\Scripts\python.exe src\select_annotation_units.py --seed 42 --sample-size 300 --max-per-book 2
.\.venv\Scripts\python.exe src\reference_annotator.py
.\.venv\Scripts\python.exe src\validate_step2.py
```

**Deviations / notes**
- Preview books are intentionally assigned to `dev`, as required by the plan.
- The reference labels are AI-generated and not human-verified. The annotation
  rules were finalized after Step 1 predictions had been viewed; the plan
  identifies this chronology limitation.

## Step 3 — Locked VADER evaluation

**Status: done**

**Files and artifacts**
- [src/evaluate_vader.py](src/evaluate_vader.py)
- [outputs/step3/](outputs/step3/), including validation, overall and per-aspect
  metrics, three-class and binary sentiment metrics/confusion matrices, paired
  text-scope comparisons, row predictions, and error examples.

**Reproduce**
```powershell
.\.venv\Scripts\python.exe src\evaluate_vader.py
```

**Deviations / notes**
- The scores are agreement with the AI-generated reference labels, not
  human-ground-truth accuracy. The evaluator documents that it does not tune
  keywords, clause splitting, VADER thresholds, annotations, or test rows.

## Step 4 — TF-IDF + SVM

**Status: done**

**Files and artifacts**
- [src/train_svm.py](src/train_svm.py)
- [models/tfidf_svm_sentiment.joblib](models/tfidf_svm_sentiment.joblib)
- [outputs/step4/](outputs/step4/), including overall and per-class metrics,
  confusion matrix, VADER comparison, reference predictions, error analysis,
  and validation outputs.

**Reproduce**
```powershell
.\.venv\Scripts\python.exe src\train_svm.py
```

**Deviations / notes**
- The classifier uses weak star-rating labels for training and reports results
  on the shared positive/negative reference subset. Neutral reference units are
  outside this binary model's scoring scope, as the plan specifies.

## Step 5 — Topic models

**Status: partly done**

**Files and artifacts**
- [src/topic_models.py](src/topic_models.py)
- [outputs/step5/lda_coherence_sweep.csv](outputs/step5/lda_coherence_sweep.csv)
- [outputs/step5/lda_topic_words.csv](outputs/step5/lda_topic_words.csv)
- [outputs/step5/lda_topic_distribution.csv](outputs/step5/lda_topic_distribution.csv)
- [outputs/step5/lda_topic_mean_probabilities.csv](outputs/step5/lda_topic_mean_probabilities.csv)
- [outputs/step5/step5_summary.json](outputs/step5/step5_summary.json)
- [outputs/step5/validation_checks.csv](outputs/step5/validation_checks.csv)

**Reproduce**
```powershell
.\.venv\Scripts\python.exe src\topic_models.py --lda-only
```

**Deviations / notes**
- LDA ran on train-book cleaned review text and selected 10 topics by c_v
  coherence (0.5343). BERTopic and pyLDAvis were skipped because the optional
  dependencies were unavailable in the current Windows/Python 3.14 environment.
- Topic names/aspect mappings have not been manually completed; the exported
  manual-review fields remain blank.
- The LDA vectorizer uses scikit-learn's `stop_words="english"`, which removes
  English stop words including negation terms such as `not`. This deviates from
  the plan's instruction never to strip negation words.

## Step 6 — DistilBERT

**Status: partly done**

**Files and artifacts**
- [src/train_distilbert.py](src/train_distilbert.py)
- [outputs/step6/](outputs/step6/), including overall metrics, baseline
  comparison, per-class metrics, confusion matrix, reference predictions, and
  validation outputs.
- [outputs/step6_smoke/](outputs/step6_smoke/) contains smoke-test artifacts.

**Reproduce**
```powershell
.\.venv\Scripts\python.exe src\train_distilbert.py --max-train 5000 --epochs 1 --batch-size 16 --output-dir outputs\step6
```

**Deviations / notes**
- The recorded run used 5,000 weak-labeled train rows and one epoch, but no
  Colab-ready notebook was created. The saved evaluation reports 20
  positive/negative reference units.
- For runs configured with more than one epoch, the script passes the reference
  set to the Trainer for per-epoch evaluation. Do not use those per-epoch gold
  metrics to choose settings or checkpoints; use the separate `dev` split for
  tuning, as required by the plan.

## Step 7 — Streamlit dashboard

**Status: not started**

**Files and artifacts**
- None. No Streamlit app or dashboard outputs are present.

**Reproduce**
- Not applicable; the dashboard has not been implemented.

**Deviations / notes**
- The planned book selector, aspect-sentiment chart, example sentences, and
  review-count display remain outstanding.

## Rule checks

| Rule | Yes / No | Evidence |
|---|---|---|
| 1. Random sample, not first N rows | **Yes** | [src/load_and_sample.py](src/load_and_sample.py) chooses random byte-range starts with a seeded RNG and then samples the deduplicated candidates; it does not take the first N rows. The byte-range approach has a possible text-length bias. |
| 2. Train/test split by book ID | **Yes** | [src/split_data.py](src/split_data.py) partitions unique `book_id` values and asserts that train, dev, and test book sets do not overlap. [outputs/step2_validation_report.json](outputs/step2_validation_report.json) records `book_leakage_count: 0`. |
| 3. Negation words not removed | **No** | The LDA vectorizer in [src/topic_models.py](src/topic_models.py) specifies `stop_words="english"`, which removes negation words such as `not`. Other pipeline text preparation preserves text, but this LDA path violates the rule. |
| 4. Labeling file has no predictions | **Yes** | [data/test_annotation_units_300_blind.csv](data/test_annotation_units_300_blind.csv) contains only IDs, book IDs, and sentence/unit text; it has no prediction columns or labels. [data/reference_annotations_300.csv](data/reference_annotations_300.csv) contains annotations and provenance but no model-prediction columns. |
| 5. No tuning on the gold set | **Yes, for the recorded runs** | [src/train_svm.py](src/train_svm.py) fits on train books, uses `dev` for validation, then evaluates on the reference set. [src/evaluate_vader.py](src/evaluate_vader.py) states it does not tune on test rows. The saved DistilBERT metrics report one epoch; with that setting, the reference set is predicted after training rather than evaluated each epoch. The multi-epoch caveat in Step 6 must be observed for future runs. |

No code was changed for this progress report.
