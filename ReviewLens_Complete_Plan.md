# ReviewLens: Complete Plan (solo, fast track)

Two-stage NLP pipeline on Amazon book reviews: find aspects (plot, pacing, characters, writing style, ending) in each sentence, then classify sentiment per aspect. Compare VADER, TF-IDF+SVM and DistilBERT, and show that aspect-level sentiment is more informative than one score per review.

## Current status

Steps 1 through 3 are complete as of 2026-10-21. The locked baseline was evaluated against 300 AI-generated reference units (`human_verified=false`). Aspect detection macro-F1 is 0.7504. Joint aspect-matched three-class VADER macro-F1 is 0.5111 on isolated aspect text versus 0.2884 on whole reviews; the measured difference is +0.2227. The separate positive/negative-only aspect-text macro-F1 is 0.7171 on 22 pairs. These are agreement scores against AI-generated references, not human-labelled gold accuracy. Step 4 has not started.

---

## 0. Working rules

- **One plan:** this file. Ignore other plans.
- **One folder, one codebase:** the ReviewLens folder. Codex/agent writes and edits code there. Claude is for planning, explanations and checking methodology.
- **Solo:** the current reference set is AI-generated and not human verified. If one-person verification is added, state that it used a single human annotator.
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

## 2. Annotation rules

The finalized rules are in `annotation_rules.md`. The initial file was empty and the rules were finalized after the Step 1 predictions were viewed. This chronology limitation must be disclosed in the report. Gold labeling must still be blind.

1. **Neutral** = no opinion: plot recap, factual description, structure ("the final chapter is ten years later").
2. **Mixed sentence** ("great characters but slow ending") = split into two labels, one per clause.
3. **Overlap:** "the story dragged" = `pacing` (it is about speed), not `plot`.
4. **Vague praise** ("Loved this book!") = aspect `none`; leave gold sentiment empty and exclude it from aspect-sentiment scoring.
5. **Sarcasm:** label what the reviewer means, not the literal words.
6. **Negation:** "not boring" is positive; "not interesting at all" is negative.
7. **Shipping, seller, condition:** `none`.
8. **One sentence, one primary aspect** unless it clearly discusses two.
9. Write down every new rule you invent while labeling, and apply it consistently.

## 3. Rules the data and evaluation must follow

1. Split train/test **by book id**, never by review or sentence (prevents the model memorizing book-specific words).
2. Keep a **raw** copy (for BERTopic, DistilBERT) and a **cleaned** copy (for LDA, TF-IDF). Never strip negation words.
3. Weak training labels: stars 4-5 = positive, 1-2 = negative; drop 3-star for training only.
4. **Random** sample of about 20k reviews, not the first N lines. The current loader uses deterministic random byte ranges to avoid downloading the 20.1 GB source. This introduces slight text-length bias and must be disclosed.
5. All methods are evaluated on the **same saved AI reference benchmark**, never used for training or tuning. Describe results as agreement with AI-generated labels until independent human verification exists. Tune only on the separate dev split.
6. Report **macro-F1**, per-class results and a confusion matrix, not only accuracy (reviews are mostly positive). Score aspect detection separately from sentiment.
7. Build the reference set **blind**: the annotation input contains no model predictions.
8. Draw the reference set randomly from test books, then check all 5 aspects, `none` and all 3 sentiments are represented. Do not hand-pick easy sentences.
9. Fixed random seeds. Save every intermediate output as CSV.
10. Keep the synthetic ChatGPT files (`synthetic_reviews_20.csv`, `synthetic_expected_labels.csv`) as a code unit test only. They are AI-written and too easy. Never report results from them.
11. LDA vs BERTopic fairness: each gets text prepared for its own needs (cleaned vs raw), the same topic count where possible. c_v coherence is only loosely comparable across the two, so also inspect topics manually. A topic is not automatically an aspect (it may be "fantasy genre" or "shipping"). Map by hand and discard junk.

## 4. Timeline (about 11 days)

| Day | Manual (you) | Agent (Codex) |
|---|---|---|
| 1 | Finalize `annotation_rules.md` (Section 2) | Step 1: loader, sampler, 20-review run |
| 2 | Read all 20 rows, note failures, finalize keywords | Step 2: full run, export labeling file |
| 3-5 | Optionally human-verify the AI reference labels | Step 3: VADER evaluation script |
| 6 | If human labels are added, relabel 30 units for self-agreement | Step 4: TF-IDF + SVM |
| 6-7 | Name and map topics by hand | Step 5: LDA + BERTopic |
| 8 | Read errors, pick 8-10 examples | Step 6: DistilBERT on Colab |
| 9 | Test the dashboard | Step 7: Streamlit |
| 9-11 | Write the report | |

## 5. Agent build order (stop for review after each step)

1. [x] **Loader and sampler:** Amazon Reviews 2023, Books category, following the official data-loading guide. Stream, do not download everything. Random sample about 20k; keep text, rating, book id; drop empty and duplicate rows; report counts, rating distribution, reviews per book. Run the existing pipeline on 20 reviews and output a CSV.
2. [x] **Full run and reference annotation:** run on the full 20,000 reviews with strict book-level splits (0 leakage; preview books in dev). Exported 300 blind test units (capped at max 2 units/book). Generated AI reference benchmark (`ai_generated_reference`, `human_verified=false`) with evidence spans and consistency review.
3. [x] **Evaluation script:** `src/evaluate_vader.py` passed 23 integrity and reconciliation checks; evaluated multilabel aspect detection plus joint aspect-matched three-class and binary VADER agreement; saved per-class metrics, confusion matrices, exclusions, row predictions, errors, and identical-row text-scope comparisons under `outputs/step3/`. No test-driven tuning occurred.
4. [ ] **TF-IDF + SVM:** `class_weight='balanced'`, trained on weak labels, tested on the saved reference benchmark.
5. [ ] **Topic models:** LDA (5-10 topics, choose by c_v coherence, cleaned text) and BERTopic with defaults on raw sentences. Export top words per topic to CSV for manual naming. pyLDAvis for LDA.
6. [ ] **DistilBERT:** fine-tune on a 5-10k sentence subset, 1-2 epochs, Colab-ready notebook, same split and reference benchmark.
7. [ ] **Streamlit:** one book dropdown, one aspect-sentiment bar chart, example sentences per bar, number of reviews analysed.

Existing files to build on: `day1_pipeline_test.py` (sentence, clause, keyword aspect, VADER), `src/load_and_sample.py`, and the synthetic CSV fixtures. `eval_synthetic.py` is referenced by the original brief but is not currently present. Do not rewrite from scratch. Agent should tell the user before continuing if any instruction looks risky for the evaluation. Keep `README.md`, `requirements.txt`, and the `data/`, `outputs/`, `src/`, `notebooks/` layout current.

## 6. Known weaknesses to expect (use them in error analysis)

- VADER misses domain words ("letdown", "gripping", "flat") and sarcasm ("Wonderful, another page of clumsy dialogue").
- VADER reads plot recaps as sentiment ("the villain kills the king") and neutral text as negative ("a missing map").
- Keyword aspect detection misses unlisted words (mystery, chapters, sentences) and overlaps ("story dragged" hits both plot and pacing).
- The clause splitter is a regex hack and can cut sentences badly.
- Weak labels (star rating copied to sentences) are noisy; the reference set exposes this.

## 7. Report outline

1. Problem and motivation (one score per review hides what readers liked or disliked)
2. Data (source, sampling, split by book, sizes, rating distribution)
3. Method (clause splitting, aspect detection, topic models, three sentiment methods)
4. Reference set (how built, rules, size, AI provenance, lack of human verification)
5. Results (macro-F1 table, per-aspect table, confusion matrices, topic coherence, aspect vs whole-review)
6. Error analysis (8-10 real examples: sarcasm, negation, recaps, overlaps)
7. Limitations (weak labels, AI-generated reference set without human verification, sparse aspect support, keyword aspects, synthetic data used only for debugging)
8. Conclusion and future work (anything dropped goes here, stated plainly)

## 8. Checklist

- [x] `annotation_rules.md` finalized; chronology limitation recorded because predictions were already viewed
- [x] Agent Step 1 output read row by row
- [x] Keywords extended from real output
- [x] Prediction-free reference set created with all three sentiment classes; labels are AI-generated and not human verified
- [ ] Optional human verification and 30-unit self-agreement study
- [ ] All methods scored on the same saved reference benchmark with macro-F1 (VADER complete; later models pending)
- [x] Aspect-level vs whole-review comparison done
- [ ] Topics named by hand, junk topics discarded
- [x] Error examples saved for missed aspects, false detections, sentiment errors, and lost splitting context
- [ ] Anything dropped is stated in the report
