"""
ReviewLens - Step 2 Full Pipeline Execution.

Processes all reviews (up to 20,000) through sentence splitting, safe clause splitting,
keyword aspect detection, and VADER sentiment analysis.
Integrates book-level split membership (train, dev, test) and produces stable, reproducible
review, sentence, and clause identifiers with full sentence context preserved.
Guarantees that 100% of input reviews are accounted for.

Usage:
    python src/run_pipeline.py
    python src/run_pipeline.py --input data/books_reviews_20000.csv --splits data/book_splits.csv --output outputs/full_pipeline_output.csv
"""

import argparse
import html
import re
from pathlib import Path
import pandas as pd
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ---------------------------------------------------------------
# Aspect Vocabulary (preserves and extends Step 1 findings)
# ---------------------------------------------------------------
ASPECTS = {
    "plot": [
        "plot", "story", "storyline", "twist", "twists", "premise", "narrative",
        "mystery", "clue", "kept me guessing"
    ],
    "characters": [
        "character", "characters", "protagonist", "hero", "heroine",
        "villain", "cast", "development"
    ],
    "pacing": [
        "pacing", "pace", "paced", "slow", "dragged", "drags", "fast-paced",
        "rushed", "boring", "page-turner", "chapters", "middle", "put down",
        "what happens next", "quick read"
    ],
    "writing_style": [
        "writing", "prose", "style", "written", "language", "dialogue", "voice",
        "descriptions", "sentences", "narration", "grammar", "typo", "typos",
        "accent", "accents", "proofread", "proofreading", "technique"
    ],
    "ending": [
        "end", "ending", "ended", "finale", "conclusion", "climax", "epilogue",
        "final chapter"
    ],
}

HTML_TAG = re.compile(r"<[^>]+>")
PACING_OVER_PLOT = re.compile(
    r"\b(?:story|plot|narrative)\b.{0,25}\b(?:dragged|drags|slow|rushed|boring)\b",
    re.I,
)
READ_TO_END = re.compile(r"\b(?:read|reading).{0,12}\b(?:to|until)\s+the\s+end\b", re.I)
CLAUSE_SPLIT_REGEX = re.compile(
    r"\s*(?:;|,?\s+but\s+|,?\s+however,?\s+|,?\s+though\s*$|\s+yet\s+)",
    re.I,
)

# Setup NLP and VADER
nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")
vader = SentimentIntensityAnalyzer()


def prepare_text(text: str) -> str:
    if text is None:
        return ""
    text_str = str(text)
    if text_str.strip() in ("", "None", "null"):
        return ""
    unescaped = html.unescape(text_str).replace("\xa0", " ")
    return HTML_TAG.sub(" ", unescaped).strip()


def split_clauses(sentence: str) -> list[str]:
    """
    Split mixed sentences into clauses while protecting correlatives, idioms,
    and structures where splitting alters the reviewer's intended assessment.
    """
    if not sentence or not sentence.strip():
        return []

    text = sentence
    # Protect correlative "not only ... but", "not just ... but"
    # and idioms like "nothing but", "anything but", "all but", "help but"
    replacements = [
        (re.compile(r"\bnot\s+only\b([^;,\n]+?)\bbut\b", re.I), r"not only\1__BUT_TOKEN__"),
        (re.compile(r"\bnot\s+just\b([^;,\n]+?)\bbut\b", re.I), r"not just\1__BUT_TOKEN__"),
        (re.compile(r"\bnot\s+simply\b([^;,\n]+?)\bbut\b", re.I), r"not simply\1__BUT_TOKEN__"),
        (
            re.compile(r"\b(?:nothing|anything|all|help)\s+but\b", re.I),
            lambda m: m.group(0).replace("but", "__BUT_TOKEN__"),
        ),
    ]
    for pattern, repl in replacements:
        text = pattern.sub(repl, text)

    parts = CLAUSE_SPLIT_REGEX.split(text)
    clauses = [
        p.replace("__BUT_TOKEN__", "but").strip()
        for p in parts
        if p and p.strip()
    ]
    return clauses or [sentence.strip()]


def find_aspects(text: str) -> list[str]:
    normalized = text.lower()
    tokens = set(re.findall(r"[a-z\-]+", normalized))
    aspects = [
        aspect
        for aspect, keywords in ASPECTS.items()
        if any(
            keyword in tokens if " " not in keyword else keyword in normalized
            for keyword in keywords
        )
    ]
    # Filter out false positive "story book"
    if "plot" in aspects and re.search(r"\bstory books?\b", normalized):
        plot_keywords = set(ASPECTS["plot"]) - {"story"}
        if not any(
            keyword in tokens if " " not in keyword else keyword in normalized
            for keyword in plot_keywords
        ):
            aspects.remove("plot")
    # Disambiguation: pacing language attached to story/plot
    if "plot" in aspects and "pacing" in aspects and PACING_OVER_PLOT.search(normalized):
        aspects.remove("plot")
    # Disambiguation: "read to the end" is reading activity, not ending evaluation
    if "ending" in aspects and READ_TO_END.search(normalized):
        strong_ending_terms = set(ASPECTS["ending"]) - {"end"}
        if not any(
            keyword in tokens if " " not in keyword else keyword in normalized
            for keyword in strong_ending_terms
        ):
            aspects.remove("ending")
    return aspects


def vader_label(compound: float, thr: float = 0.05) -> str:
    if compound >= thr:
        return "positive"
    if compound <= -thr:
        return "negative"
    return "neutral"


def run_pipeline_on_dataframe(
    df_reviews: pd.DataFrame,
    df_splits: pd.DataFrame | None = None,
) -> pd.DataFrame:
    book_to_split = {}
    if df_splits is not None and not df_splits.empty:
        book_to_split = dict(zip(df_splits["book_id"].astype(str), df_splits["split"]))

    rows = []
    text_column = "cleaned_text" if "cleaned_text" in df_reviews.columns else "raw_text"
    if text_column not in df_reviews.columns:
        text_column = "text"

    for idx, r in df_reviews.iterrows():
        review_id = str(r.get("review_id", f"review_{idx+1:05d}"))
        book_id = str(r.get("book_id", r.get("parent_asin", "")))
        split_name = book_to_split.get(book_id, "unknown")
        raw_val = r.get(text_column, "")
        rating = float(r["rating"]) if pd.notna(r.get("rating")) else None
        cleaned = prepare_text(raw_val)

        # Handle empty/null review text gracefully so review is accounted for
        if not cleaned:
            rows.append({
                "review_id": review_id,
                "book_id": book_id,
                "split": split_name,
                "stars": rating,
                "whole_review_vader": 0.0,
                "sentence_id": f"{review_id}_s01",
                "sentence_text": "[empty review]",
                "clause_id": f"{review_id}_s01_c01",
                "clause_text": "[empty review]",
                "aspect": "none",
                "clause_vader": 0.0,
                "pred_sentiment": "neutral",
            })
            continue

        whole_score = round(vader.polarity_scores(cleaned)["compound"], 2)
        doc = nlp(cleaned)
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        if not sents:
            sents = [cleaned]

        for s_idx, sent in enumerate(sents, start=1):
            sentence_id = f"{review_id}_s{s_idx:02d}"
            clauses = split_clauses(sent)
            for c_idx, clause in enumerate(clauses, start=1):
                clause_id = f"{sentence_id}_c{c_idx:02d}"
                aspects = find_aspects(clause)
                c_score = round(vader.polarity_scores(clause)["compound"], 2)
                pred_sent = vader_label(c_score)

                for asp in aspects or ["none"]:
                    rows.append({
                        "review_id": review_id,
                        "book_id": book_id,
                        "split": split_name,
                        "stars": rating,
                        "whole_review_vader": whole_score,
                        "sentence_id": sentence_id,
                        "sentence_text": sent,
                        "clause_id": clause_id,
                        "clause_text": clause,
                        "aspect": asp,
                        "clause_vader": c_score,
                        "pred_sentiment": pred_sent,
                    })

    df_out = pd.DataFrame(rows)

    # Verification: Every input review MUST be present in output
    input_reviews = set(df_reviews["review_id"].astype(str)) if "review_id" in df_reviews.columns else set()
    output_reviews = set(df_out["review_id"].astype(str))
    if input_reviews:
        missing = input_reviews - output_reviews
        if missing:
            raise RuntimeError(f"Pipeline dropped {len(missing)} reviews! Missing: {list(missing)[:5]}")

    return df_out


def run_pipeline(
    input_path: str = "data/books_reviews_20000.csv",
    splits_path: str = "data/book_splits.csv",
    output_path: str = "outputs/full_pipeline_output.csv",
    summary_path: str = "outputs/full_pipeline_summary.csv",
    limit: int | None = None,
) -> pd.DataFrame:
    print(f"Loading reviews from: {input_path}")
    df_reviews = pd.read_csv(input_path, keep_default_na=False)
    if limit:
        df_reviews = df_reviews.head(limit)
    print(f"Loaded {len(df_reviews):,} reviews.")

    df_splits = None
    if splits_path and Path(splits_path).exists():
        print(f"Loading split assignments from: {splits_path}")
        df_splits = pd.read_csv(splits_path, keep_default_na=False)

    print("Executing sentence splitting, clause segmentation, aspect detection, and VADER...")
    df_out = run_pipeline_on_dataframe(df_reviews, df_splits)
    print(f"Pipeline generated {len(df_out):,} clause-aspect records across {df_out['review_id'].nunique():,} unique reviews.")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"Saved pipeline output to: {output_path}")

    # Summary report
    aspect_counts = df_out["aspect"].value_counts().to_dict()
    total_clauses = len(df_out)
    summary_data = []
    for asp in ["plot", "characters", "pacing", "writing_style", "ending", "none"]:
        cnt = aspect_counts.get(asp, 0)
        asp_df = df_out[df_out["aspect"] == asp]
        pos_cnt = (asp_df["pred_sentiment"] == "positive").sum()
        neg_cnt = (asp_df["pred_sentiment"] == "negative").sum()
        neu_cnt = (asp_df["pred_sentiment"] == "neutral").sum()
        summary_data.append({
            "aspect": asp,
            "clause_count": cnt,
            "percentage": round(cnt / total_clauses * 100, 2),
            "pred_positive": pos_cnt,
            "pred_negative": neg_cnt,
            "pred_neutral": neu_cnt,
        })
    df_summary = pd.DataFrame(summary_data)
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(summary_path, index=False)
    print(f"Saved pipeline summary to: {summary_path}")

    print("\n--- Pipeline Aspect & Sentiment Distribution ---")
    print(df_summary.to_string(index=False))

    if "split" in df_out.columns:
        print("\n--- Records by Split ---")
        split_dist = df_out.groupby("split")["review_id"].nunique()
        print(split_dist.to_string())

    return df_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ReviewLens pipeline across reviews.")
    parser.add_argument("--input", default="data/books_reviews_20000.csv", help="Input reviews CSV")
    parser.add_argument("--splits", default="data/book_splits.csv", help="Book splits CSV")
    parser.add_argument("--output", default="outputs/full_pipeline_output.csv", help="Output CSV")
    parser.add_argument("--summary", default="outputs/full_pipeline_summary.csv", help="Summary CSV")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for quick test")

    args = parser.parse_args()
    run_pipeline(
        input_path=args.input,
        splits_path=args.splits,
        output_path=args.output,
        summary_path=args.summary,
        limit=args.limit,
    )
