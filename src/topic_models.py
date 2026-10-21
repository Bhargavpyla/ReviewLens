"""
ReviewLens - Step 5: Topic Models (LDA + BERTopic).

Uses scikit-learn's LatentDirichletAllocation for LDA and BERTopic on raw
sentences. Both models use training-book data only. Exports top words per topic
to CSV for manual naming.

The plan requires:
  - LDA: 5-10 topics, choose best by c_v coherence, cleaned review text
  - BERTopic: default settings on raw sentences (optional, skipped if unavailable)
  - Export top words per topic to CSV for manual naming
  - pyLDAvis for LDA

Usage:
    python src/topic_models.py
    python src/topic_models.py --lda-only   # skip BERTopic if deps unavailable
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

# ---------------------------------------------------------------
# Text Cleaning
# ---------------------------------------------------------------
def clean_for_lda(text: str) -> str:
    """Lowercase, remove non-alpha tokens, basic cleaning for LDA."""
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------
# ---------------------------------------------------------------
# LDA Pipeline (scikit-learn)
# ---------------------------------------------------------------
def run_lda(
    texts: list[str],
    output_dir: str,
    min_topics: int = 5,
    max_topics: int = 10,
) -> tuple:
    """Train LDA across topic counts and select the model with highest c_v."""
    print("\n=== LDA Topic Modeling (scikit-learn) ===")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    vectorizer = CountVectorizer(
        stop_words="english",
        min_df=10,
        max_df=0.5,
        max_features=15000,
        ngram_range=(1, 1),
        token_pattern=r"\b[a-z]{3,}\b",
    )
    dtm = vectorizer.fit_transform(texts)
    vocab = vectorizer.get_feature_names_out()
    if dtm.shape[0] == 0 or dtm.shape[1] == 0:
        raise ValueError("LDA requires documents and a non-empty vocabulary.")
    if min_topics < 2 or max_topics < min_topics:
        raise ValueError("Topic range must satisfy 2 <= min_topics <= max_topics.")

    import tomotopy as tp

    vocab_set = set(vocab)
    tokenized_texts = [
        [token for token in text.split() if token in vocab_set]
        for text in texts
    ]
    coherence_corpus = tp.utils.Corpus()
    for tokens in tokenized_texts:
        if tokens:
            coherence_corpus.add_doc(words=tokens)
    if not len(coherence_corpus):
        raise ValueError("No tokens remain in the LDA coherence dictionary.")

    print(f"Vocabulary: {len(vocab)} unique tokens after filtering.")
    print(f"Document-term matrix: {dtm.shape[0]} documents × {dtm.shape[1]} terms.")

    # Sweep topic counts
    coherence_scores = []
    all_models = {}
    top_n_words = 10

    for k in range(min_topics, max_topics + 1):
        print(f"  Training LDA with k={k} topics...", end=" ", flush=True)
        lda = LatentDirichletAllocation(
            n_components=k,
            random_state=SEED,
            max_iter=25,
            learning_method="online",
            batch_size=256,
            n_jobs=-1,
            doc_topic_prior=None,  # auto
            topic_word_prior=None,  # auto
        )
        lda.fit(dtm)

        topic_words = []
        for topic_dist in lda.components_:
            top_ids = topic_dist.argsort()[-top_n_words:][::-1].tolist()
            topic_words.append([vocab[i] for i in top_ids])

        coherence = tp.coherence.Coherence(
            coherence_corpus,
            coherence="c_v",
            targets=set(word for words in topic_words for word in words),
        )
        c_v = float(np.mean([
            coherence.get_score(words=words)
            for words in topic_words
        ]))
        coherence_scores.append({
            "k": k,
            "c_v_coherence": round(c_v, 4),
        })
        all_models[k] = lda
        print(f"c_v = {c_v:.4f}")

    # Save coherence sweep
    coh_df = pd.DataFrame(coherence_scores)
    coh_df.to_csv(output_path / "lda_coherence_sweep.csv", index=False)

    best_row = max(coherence_scores, key=lambda x: x["c_v_coherence"])
    best_k = best_row["k"]
    best_coherence = best_row["c_v_coherence"]
    best_model = all_models[best_k]
    print(f"\n  Best LDA: k={best_k}, c_v={best_coherence:.4f}")

    # Export top words per topic
    topic_words = []
    for t_idx in range(best_k):
        top_ids = best_model.components_[t_idx].argsort()[-15:][::-1]
        words = [vocab[i] for i in top_ids]
        weights = [round(best_model.components_[t_idx][i], 2) for i in top_ids]
        topic_words.append({
            "topic_id": t_idx,
            "top_words": ", ".join(words),
            "top_weights": ", ".join(map(str, weights)),
            "manual_aspect": "",
            "manual_topic_label": "",
            "is_junk": "",
        })
    tw_df = pd.DataFrame(topic_words)
    tw_df.to_csv(output_path / "lda_topic_words.csv", index=False)
    print(f"  Exported top-15 words per topic to {output_path / 'lda_topic_words.csv'}")

    return best_model, vectorizer, dtm, best_k, coherence_scores


def generate_pyldavis(model, dtm, vectorizer, output_dir: str) -> bool:
    """Generate pyLDAvis interactive HTML visualisation if available."""
    try:
        import pyLDAvis
        import pyLDAvis.lda_model as lda_vis
    except ImportError:
        print("  pyLDAvis not available; skipping HTML visualisation.")
        print("  Install with: pip install pyLDAvis")
        return False

    print("  Generating pyLDAvis visualisation...")
    vis_data = lda_vis.prepare(model, dtm, vectorizer)
    html_path = Path(output_dir) / "lda_pyldavis.html"
    pyLDAvis.save_html(vis_data, str(html_path))
    print(f"  Saved pyLDAvis to {html_path}")
    return True


# ---------------------------------------------------------------
# BERTopic Pipeline
# ---------------------------------------------------------------
def run_bertopic(
    raw_texts: list[str],
    output_dir: str,
) -> tuple:
    """
    Run BERTopic on raw sentence texts.
    Returns (model, topics, probs) or (None, None, None) if unavailable.
    """
    print("\n=== BERTopic Topic Modeling ===")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        from bertopic import BERTopic
        from umap import UMAP
    except ImportError:
        print("  BERTopic not installed. Skipping.")
        print("  To enable: pip install bertopic sentence-transformers umap-learn hdbscan")
        return None, None, None

    print(f"  Fitting BERTopic on {len(raw_texts):,} sentences...")
    model = BERTopic(umap_model=UMAP(random_state=SEED), verbose=True)
    topics, probs = model.fit_transform(raw_texts)

    # Topic info
    info_df = model.get_topic_info()
    info_df.to_csv(output_path / "bertopic_topic_info.csv", index=False)
    print(f"  Found {len(info_df) - 1} topics (excluding outlier topic -1).")

    # Top words per topic
    topic_words = []
    for tid in info_df["Topic"].unique():
        if tid == -1:
            continue
        top = model.get_topic(tid)
        if not top:
            continue
        words = [w for w, _ in top[:15]]
        weights = [round(s, 4) for _, s in top[:15]]
        topic_words.append({
            "topic_id": tid,
            "top_words": ", ".join(words),
            "top_weights": ", ".join(map(str, weights)),
            "manual_aspect": "",
            "manual_topic_label": "",
            "is_junk": "",
        })
    tw_df = pd.DataFrame(topic_words)
    tw_df.to_csv(output_path / "bertopic_topic_words.csv", index=False)
    print(f"  Exported top words to {output_path / 'bertopic_topic_words.csv'}")

    return model, topics, probs


# ---------------------------------------------------------------
# Document-Topic Distribution Analysis
# ---------------------------------------------------------------
def analyse_topic_distribution(
    model, dtm, output_dir: str, prefix: str = "lda"
):
    """Compute and save per-document topic distributions and statistics."""
    output_path = Path(output_dir)
    doc_topics = model.transform(dtm)

    # Dominant topic per document
    dominant = np.argmax(doc_topics, axis=1)
    dominant_counts = pd.Series(dominant).value_counts().sort_index()

    dist_df = pd.DataFrame({
        "topic_id": dominant_counts.index,
        "document_count": dominant_counts.values,
        "percentage": (dominant_counts.values / len(dominant) * 100).round(2),
    })
    dist_df.to_csv(output_path / f"{prefix}_topic_distribution.csv", index=False)
    print(f"  Topic distribution saved to {prefix}_topic_distribution.csv")

    # Mean topic probabilities
    mean_probs = doc_topics.mean(axis=0)
    prob_df = pd.DataFrame({
        "topic_id": range(len(mean_probs)),
        "mean_probability": mean_probs.round(4),
    })
    prob_df.to_csv(output_path / f"{prefix}_topic_mean_probabilities.csv", index=False)

    return doc_topics


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def run_topic_models(
    reviews_input: str = "data/books_reviews_20000.csv",
    pipeline_output: str = "outputs/full_pipeline_output.csv",
    splits_path: str = "data/book_splits.csv",
    output_dir: str = "outputs/step5",
    lda_only: bool = False,
    min_topics: int = 5,
    max_topics: int = 10,
):
    """Run LDA on cleaned reviews and BERTopic on raw training sentences."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("--- Loading book splits ---")
    df_splits = pd.read_csv(splits_path, keep_default_na=False)
    book_to_split = dict(zip(df_splits["book_id"].astype(str), df_splits["split"]))

    # --- LDA (cleaned text) ---
    print("--- Loading cleaned reviews for LDA ---")
    reviews = pd.read_csv(reviews_input, keep_default_na=False)
    required_review_columns = {"book_id", "cleaned_text"}
    if missing := required_review_columns - set(reviews.columns):
        raise ValueError(f"Review input is missing required columns: {sorted(missing)}")
    reviews["split"] = reviews["book_id"].astype(str).map(book_to_split)
    if reviews["split"].isna().any():
        raise ValueError("Some reviews have no book-level split assignment.")
    train_reviews = reviews[reviews["split"] == "train"].drop_duplicates("review_id")
    if train_reviews.empty:
        raise ValueError("No training-split reviews are available for LDA.")

    print("\nCleaning text for LDA...")
    cleaned_texts = [
        clean_for_lda(text)
        for text in train_reviews["cleaned_text"]
    ]
    cleaned_texts = [text for text in cleaned_texts if len(text.split()) >= 3]
    print(f"Training review documents with >=3 tokens: {len(cleaned_texts):,}")

    lda_model, vectorizer, dtm, best_k, coh_scores = run_lda(
        cleaned_texts, output_dir, min_topics, max_topics
    )

    # Document-topic distribution analysis
    analyse_topic_distribution(lda_model, dtm, output_dir, prefix="lda")

    # pyLDAvis
    pyldavis_ok = generate_pyldavis(lda_model, dtm, vectorizer, output_dir)

    # --- BERTopic (raw training sentences) ---
    bt_model = None
    bt_topic_count = 0
    raw_sentences = []
    if not lda_only:
        print("--- Loading raw sentences for BERTopic ---")
        sentences = pd.read_csv(pipeline_output, keep_default_na=False)
        required_sentence_columns = {"book_id", "split", "sentence_id", "sentence_text"}
        if missing := required_sentence_columns - set(sentences.columns):
            raise ValueError(
                f"Pipeline output is missing required columns: {sorted(missing)}"
            )
        sentences["split"] = sentences["book_id"].astype(str).map(book_to_split)
        if sentences["split"].isna().any():
            raise ValueError("Some sentences have no book-level split assignment.")
        train_sentences = sentences[sentences["split"] == "train"].drop_duplicates(
            subset=["sentence_id"]
        )
        raw_sentences = [
            text.strip()
            for text in train_sentences["sentence_text"]
            if len(text.split()) >= 3
        ]
        if not raw_sentences:
            raise ValueError("No training-split raw sentences are available for BERTopic.")

        bt_model, bt_topics, bt_probs = run_bertopic(raw_sentences, output_dir)

        if bt_model is not None:
            bt_topic_count = int(pd.Series(bt_topics).nunique() - (-1 in bt_topics))
            print("\n  Review topic-word CSVs and enter topic names/aspects manually.")
    else:
        print("\n  BERTopic skipped (--lda-only flag).")

    # --- Validation checks ---
    val_checks = [
        {
            "check": "lda_coherence_sweep",
            "status": "PASS",
            "detail": f"Tested k={min_topics}-{max_topics}, best k={best_k}",
        },
        {
            "check": "lda_topic_words_exported",
            "status": "PASS",
            "detail": f"Top-15 words exported for {best_k} topics",
        },
        {
            "check": "manual_topic_naming",
            "status": "INFO",
            "detail": "Topic aspect/name/junk fields are blank for manual review",
        },
        {
            "check": "training_books_only",
            "status": "PASS",
            "detail": "LDA reviews and BERTopic sentences were filtered to train books",
        },
        {
            "check": "pyldavis_generated",
            "status": "PASS" if pyldavis_ok else "SKIP",
            "detail": "pyLDAvis HTML saved" if pyldavis_ok else "pyLDAvis not available",
        },
        {
            "check": "bertopic_run",
            "status": "PASS" if bt_model else "SKIP",
            "detail": f"BERTopic found {bt_topic_count} topics" if bt_model else "BERTopic skipped or unavailable",
        },
    ]
    val_df = pd.DataFrame(val_checks)
    val_df.to_csv(output_path / "validation_checks.csv", index=False)

    # --- Summary ---
    summary = {
        "status": "PASS",
        "lda_best_k": best_k,
        "lda_best_c_v": max(item["c_v_coherence"] for item in coh_scores),
        "lda_review_count": len(cleaned_texts),
        "bertopic_available": bt_model is not None,
        "bertopic_topic_count": bt_topic_count,
        "bertopic_sentence_count": len(raw_sentences),
        "pyldavis_generated": pyldavis_ok,
        "note": "Name topics and map/discard them manually after reviewing the CSVs.",
    }
    with open(output_path / "step5_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n--- Step 5 Summary saved to {output_path / 'step5_summary.json'} ---")
    print(json.dumps(summary, indent=2))

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReviewLens Step 5: Topic Models")
    parser.add_argument("--reviews-input", default="data/books_reviews_20000.csv")
    parser.add_argument("--pipeline-output", default="outputs/full_pipeline_output.csv")
    parser.add_argument("--splits", default="data/book_splits.csv")
    parser.add_argument("--output-dir", default="outputs/step5")
    parser.add_argument("--lda-only", action="store_true",
                        help="Skip BERTopic (useful if deps unavailable)")
    parser.add_argument("--min-topics", type=int, default=5)
    parser.add_argument("--max-topics", type=int, default=10)

    args = parser.parse_args()
    run_topic_models(
        reviews_input=args.reviews_input,
        pipeline_output=args.pipeline_output,
        splits_path=args.splits,
        output_dir=args.output_dir,
        lda_only=args.lda_only,
        min_topics=args.min_topics,
        max_topics=args.max_topics,
    )
