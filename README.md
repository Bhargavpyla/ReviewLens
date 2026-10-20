# ReviewLens

ReviewLens is a solo ML Lab project for aspect-based sentiment analysis of Amazon book reviews. It detects `plot`, `characters`, `pacing`, `writing_style`, and `ending`, then predicts sentiment for each detected aspect.

The authoritative workflow is `ReviewLens_Complete_Plan.md`. Work stops for review after each build step.

## Current Status

Step 1 is complete. The project contains a deterministic 20,000-review sample, a 20-review preview, finalized annotation rules, and the preview pipeline output. Step 2 has not started.

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

## Step 1 Outputs

- `data/books_reviews_20000.csv`: master sample with raw and cleaned text.
- `data/books_sample_20.csv`: first 20 rows of the deterministic master sample.
- `outputs/books_reviews_20000_summary.csv`: sampling and cleaning counts.
- `outputs/books_reviews_20000_rating_distribution.csv`: rating counts.
- `outputs/books_reviews_20000_reviews_per_book.csv`: review count by book.
- `outputs/day1_pipeline_output.csv`: sentence/clause aspect and VADER predictions.
- `outputs/day1_console_output.txt`: complete preview run log.

## Sampling Method

The official Amazon Reviews 2023 Books JSONL file is about 20.1 GB. The loader reads deterministic random byte ranges across the file, discards partial boundary records, removes empty and duplicate reviews, and samples from the resulting candidates. This avoids downloading the full source. It is an approximate random-row sample because byte-range selection gives a slight advantage to longer records; disclose this limitation in the report.

## Folder Layout

```text
ReviewLens/
|-- data/
|-- notebooks/
|-- outputs/
|-- src/
|-- annotation_rules.md
|-- day1_pipeline_test.py
|-- requirements.txt
|-- ReviewLens_Complete_Plan.md
`-- ReviewLens_agent_brief.md
```

## Important Rules

- Split train, development, and test data by `book_id`.
- Preserve raw text and use cleaned text without removing negation.
- Keep the gold-labeling file blind: no model predictions.
- Never train or tune on the gold set.
- Report macro-F1, per-class results, confusion matrices, and aspect detection separately.
- Treat the synthetic CSV files only as code-test fixtures; never report their scores as results.

## Known Step 1 Findings

- Most random Books-category text does not mention one of the five supported aspects.
- The Books category includes non-narrative products and reviews of illustrations, formatting, and physical condition.
- VADER mishandles sarcasm, negation scope, plot recaps, and phrases such as "mediocre at best" or "leave a lot to be desired".
- Regex clause splitting can separate text awkwardly.
- The annotation rules were finalized after predictions were viewed because the initial file was empty. Keep the later gold-labeling process prediction-free and disclose this chronology limitation.
