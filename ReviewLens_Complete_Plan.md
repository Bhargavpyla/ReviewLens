# ReviewLens: Complete Plan (solo, fast track)

Two-stage NLP pipeline on Amazon book reviews: find aspects (plot, pacing, characters, writing style, ending) in each sentence, then classify sentiment per aspect. Compare VADER, TF-IDF+SVM and DistilBERT, and show that aspect-level sentiment is more informative than one score per review.

---

## 0. Working rules

- **One plan:** this file. Ignore other plans.
- **One folder, one codebase:** the ReviewLens folder. Codex/agent writes and edits code there. Claude is for planning, explanations and checking methodology.
- **Solo:** the gold set is labeled by one person. State this in the report.
- **Priority order:** if time runs out, cut from the bottom of the build order (Section 5). Steps 1-4 already make a complete project. If anything is dropped (BERTopic, DistilBERT), say so in the report.

## 1. Aspect schema (fixed)

| Aspect | Covers | Seed keywords |
|---|---|---|
| `plot` | story, premise, twists, mystery | plot, story, storyline, twist, premise, narrative, mystery, clue |
| `characters` | people and their development | character(s), protagonist, hero, villain, cast, development |
| `pacing` | speed, slow stretches, flow | pacing, pace, slow, dragged, rushed, boring, page-turner, chapters, middle |
| `writing_style` | prose, dialogue, descriptions | writing, prose, style, dialogue, voice, descriptions, sentences, narration |
| `ending` | conclusion | ending, finale, conclusion, climax, epilogue, final chapter |
| `none` | not about book content | shipping, seller, price, packaging, condition |

Sentiment labels: `positive`, `negative`, `neutral`.

## 2. Annotation rules (draft, edit and finalize on Day 1 BEFORE looking at model output)

1. **Neutral** = no opinion: plot recap, factual description, structure ("the final chapter is ten years later").
2. **Mixed sentence** ("great characters but slow ending") = split into two labels, one per clause.
3. **Overlap:** "the story dragged" = `pacing` (it is about speed), not `plot`.
4. **Vague praise** ("Loved this book!") = aspect `none`, sentiment ignored.
5. **Sarcasm:** label what the reviewer means, not the literal words.
6. **Negation:** "not boring" is positive; "not interesting at all" is negative.
7. **Shipping, seller, condition:** `none`.
8. **One sentence, one primary aspect** unless it clearly discusses two.
9. Write down every new rule you invent while labeling, and apply it consistently.

## 3. Rules the data and evaluation must follow

1. Split train/test **by book id**, never by review or sentence (prevents the model memorizing book-specific words).
2. Keep a **raw** copy (for BERTopic, DistilBERT) and a **cleaned** copy (for LDA, TF-IDF). Never strip negation words.
3. Weak training labels: stars 4-5 = positive, 1-2 = negative; drop 3-star for training only.
4. **Random** sample of about 20k reviews, not the first N lines.
5. All methods are evaluated on the **same hand-labeled gold set**, never used for training or tuning. Tune on a separate dev split.
6. Report **macro-F1**, per-class results and a confusion matrix, not only accuracy (reviews are mostly positive). Score aspect detection separately from sentiment.
7. Label the gold set **blind**: the labeling file contains no model predictions.
8. Draw the gold set randomly from test books, then check all 5 aspects, `none` and all 3 sentiments are represented. Do not hand-pick easy sentences.
9. Fixed random seeds. Save every intermediate output as CSV.
10. Keep the synthetic ChatGPT files (`synthetic_reviews_20.csv`, `synthetic_expected_labels.csv`) as a code unit test only. They are AI-written and too easy. Never report results from them.
11. LDA vs BERTopic fairness: each gets text prepared for its own needs (cleaned vs raw), the same topic count where possible. c_v coherence is only loosely comparable across the two, so also inspect topics manually. A topic is not automatically an aspect (it may be "fantasy genre" or "shipping"). Map by hand and discard junk.

## 4. Timeline (about 11 days)

| Day | Manual (you) | Agent (Codex) |
|---|---|---|
| 1 | Write `annotation_rules.md` (Section 2) | Step 1: loader, sampler, 20-review run |
| 2 | Read all 20 rows, note failures, finalize keywords | Step 2: full run, export labeling file |
| 3-5 | Hand-label 150-200 sentences (about 3-4 hours total) | Step 3: VADER evaluation script |
| 6 | Relabel 30 sentences without looking at old labels (self-agreement) | Step 4: TF-IDF + SVM |
| 6-7 | Name and map topics by hand | Step 5: LDA + BERTopic |
| 8 | Read errors, pick 8-10 examples | Step 6: DistilBERT on Colab |
| 9 | Test the dashboard | Step 7: Streamlit |
| 9-11 | Write the report | |

## 5. Agent build order (stop for review after each step)

1. **Loader and sampler:** Amazon Reviews 2023, Books category, following the official data-loading guide. Stream, do not download everything. Random sample about 20k; keep text, rating, book id; drop empty and duplicate rows; report counts, rating distribution, reviews per book. Run the existing pipeline on 20 reviews and output a CSV.
2. **Full run and labeling export:** run on the full sample. Export about 300 random test-book sentences with columns `sentence_id, review_id, sentence, gold_aspect, gold_sentiment` (last two empty, no predictions). Not dominated by one book or one aspect.
3. **Evaluation script:** VADER on the gold set. Per-aspect table, confusion matrix, macro-F1, aspect-detection accuracy, and **aspect-level vs whole-review sentiment** (the key result).
4. **TF-IDF + SVM:** `class_weight='balanced'`, trained on weak labels, tested on the gold set.
5. **Topic models:** LDA (5-10 topics, choose by c_v coherence, cleaned text) and BERTopic with defaults on raw sentences. Export top words per topic to CSV for manual naming. pyLDAvis for LDA.
6. **DistilBERT:** fine-tune on a 5-10k sentence subset, 1-2 epochs, Colab-ready notebook, same split and gold set.
7. **Streamlit:** one book dropdown, one aspect-sentiment bar chart, example sentences per bar, number of reviews analysed.

Existing files to build on: `day1_pipeline_test.py` (sentence, clause, keyword aspect, VADER), `eval_synthetic.py`. Do not rewrite from scratch. Agent should tell the user before continuing if any instruction looks risky for the evaluation. Include `README.md` with run commands, `requirements.txt`, and a folder layout of `data/`, `outputs/`, `src/`, `notebooks/`.

## 6. Known weaknesses to expect (use them in error analysis)

- VADER misses domain words ("letdown", "gripping", "flat") and sarcasm ("Wonderful, another page of clumsy dialogue").
- VADER reads plot recaps as sentiment ("the villain kills the king") and neutral text as negative ("a missing map").
- Keyword aspect detection misses unlisted words (mystery, chapters, sentences) and overlaps ("story dragged" hits both plot and pacing).
- The clause splitter is a regex hack and can cut sentences badly.
- Weak labels (star rating copied to sentences) are noisy; the gold set exposes this.

## 7. Report outline

1. Problem and motivation (one score per review hides what readers liked or disliked)
2. Data (source, sampling, split by book, sizes, rating distribution)
3. Method (clause splitting, aspect detection, topic models, three sentiment methods)
4. Gold set (how built, rules, size, single annotator, self-agreement on 30 items)
5. Results (macro-F1 table, per-aspect table, confusion matrices, topic coherence, aspect vs whole-review)
6. Error analysis (8-10 real examples: sarcasm, negation, recaps, overlaps)
7. Limitations (weak labels, small single-annotator gold set, keyword aspects, synthetic data used only for debugging)
8. Conclusion and future work (anything dropped goes here, stated plainly)

## 8. Checklist

- [ ] annotation_rules.md written before looking at output
- [ ] Agent step 1 output read row by row
- [ ] Keywords extended from real output
- [ ] Gold set labeled blind, 150-200 sentences, all classes present
- [ ] 30 sentences relabeled for self-agreement
- [ ] All methods scored on the same gold set with macro-F1
- [ ] Aspect-level vs whole-review comparison done
- [ ] Topics named by hand, junk topics discarded
- [ ] 8-10 error examples saved
- [ ] Anything dropped is stated in the report
