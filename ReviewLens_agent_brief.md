# ReviewLens: Agent Brief

`ReviewLens_Complete_Plan.md` is the authoritative plan. This brief summarizes the fixed constraints. If the two files conflict, follow the complete plan.

**Current status (2026-10-21):** Steps 1 and 2 are complete. Step 3 has not started.

Aspect-based sentiment analysis on Amazon book reviews. Solo student project (ML Lab, MIT Manipal), limited time. Work inside the existing ReviewLens folder and build on the existing files; do not rewrite from scratch.

**Existing files**
- `src/load_and_sample.py`: deterministic random-range sampler for the official Books JSONL source
- `src/split_data.py`: strict book-level partitioning (70% train, 15% dev, 15% test; preview books in dev)
- `src/run_pipeline.py`: full dataset pipeline processing 20,000 reviews with stable IDs and accounting
- `src/select_annotation_units.py`: sampling 300 test units with <=2 units/book cap into blind evaluation file
- `src/reference_annotator.py`: AI reference annotation engine with factual ending neutral rules and consistency review
- `src/validate_step2.py`: comprehensive verification suite for splits, IDs, and annotations
- `day1_pipeline_test.py`: review -> sentences (spaCy sentencizer) -> clauses -> keyword aspects -> VADER sentiment (supports preview and full run)
- `data/synthetic_reviews_20.csv` and `data/synthetic_expected_labels.csv`: development fixtures only
- `eval_synthetic.py` is not currently present; add it later only if the synthetic regression test is restored

## Aspect schema (fixed, do not add aspects on your own)

| Aspect | Seed keywords (extend after reading real output) |
|---|---|
| `plot` | plot, story, storyline, twist, premise, narrative, mystery, clue |
| `characters` | character(s), protagonist, hero, heroine, villain, cast, development |
| `pacing` | pacing, pace, slow, dragged, rushed, boring, page-turner, chapters, middle |
| `writing_style` | writing, prose, style, dialogue, voice, descriptions, sentences, narration |
| `ending` | ending, finale, conclusion, climax, epilogue, final chapter |
| `none` | no book-content aspect (shipping, seller, price, packaging, condition) |

Note: "story" and "plot" are the same aspect (`plot`). Sentiment labels are `positive`, `negative`, `neutral`. A plot recap or factual description with no opinion is `neutral`. A clause may carry more than one aspect only if it clearly discusses both.

## Rules that must hold

1. **Split train/test by book id**, never by review or sentence.
2. Keep a **raw text** copy and a **cleaned** copy. Never remove negation ("not", "never", "no") from sentiment input.
3. Weak training labels: stars 4-5 = positive, 1-2 = negative; drop 3-star for training only.
4. All methods are evaluated on the **same hand-labeled gold set / AI reference benchmark**. The reference set is never used for training or tuning.
5. Report **macro-F1** and a confusion matrix, not only accuracy. Report aspect detection separately from sentiment.
6. Fixed random seeds everywhere. Save every intermediate output as CSV.
7. Do not tune keywords or models on the gold set. Tune on a separate dev split.
8. Never show model predictions in the file used for labeling.
9. If something in these instructions looks wrong or risky for the evaluation, tell me before continuing.

## Data

- Source: Amazon Reviews 2023, Books category (follow the official data-loading guide). Do not download the whole dataset; stream it.
- **Randomly sample** about 20k reviews (not the first N lines). Keep review text, rating, book id. The current deterministic byte-range method has slight text-length bias; disclose it.
- Drop empty text and duplicates. Report counts before and after.
- Report rating distribution and reviews-per-book after sampling.

## Build order (stop and wait for me after each step)

1. [x] **Loader and sampler.** Produce the 20,000-review master CSV, derive a 20-review preview from it, run the existing pipeline, review every preview row, and finalize the Day 1 rules and keywords.
2. [x] **Full-sample run & reference annotation.** Full 20k reviews processed into train/dev/test splits with 0 book leakage; all 20 preview books in dev. Exported 300 test units with a 2-unit cap per book. Generated prediction-free blind file and AI-generated reference benchmark (`ai_generated_reference`, `human_verified=false`) with evidence spans and consistency review.

3. [ ] **Evaluation script.** Run VADER on the labeled gold set. Output: per-aspect table, confusion matrix, macro-F1, aspect-detection accuracy, and a comparison of aspect-level sentiment against whole-review sentiment.
4. [ ] **TF-IDF + SVM** (`class_weight='balanced'`), trained on weak labels, evaluated on the gold set.
5. [ ] **Topic models.** LDA (choose topic count by coherence, c_v, 5-10 topics) on cleaned text; BERTopic with default settings on raw sentences. Output top words per topic to a CSV for manual naming. Visualize LDA with pyLDAvis.
6. [ ] **DistilBERT** fine-tuned on a 5-10k sentence subset, as a Colab-ready notebook. Same train/test split and same gold set.
7. [ ] **Streamlit app**: one book dropdown, one aspect-sentiment bar chart, example sentences for each bar.

If time runs out, steps 1-4 already make a complete project. Cut from the bottom, not the top.

## Reproducibility

- A `README.md` with exact run commands in order.
- A `requirements.txt`.
- One folder layout: `data/`, `outputs/`, `src/`, `notebooks/`.
