"""
ReviewLens - Step 1 preview: run 20 master-sample reviews through the full baseline.

Chain:  review -> sentences -> clauses -> aspect keywords -> VADER sentiment
Goal:   Review baseline failures before creating the blind Step 2 labeling file.

Setup:  pip install vaderSentiment spacy pandas
Run:    python day1_pipeline_test.py                      (uses built-in samples)
        python day1_pipeline_test.py data/books_sample_20.csv
"""
import html
import re
import sys
import pandas as pd
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------------------------------------------
# 1. Aspect vocabulary (start small, grow it after reading output)
# ---------------------------------------------------------------
ASPECTS = {
    "plot":      ["plot", "story", "storyline", "twist", "twists", "premise", "narrative", "mystery", "clue",
                  "kept me guessing"],
    "characters": ["character", "characters", "protagonist", "hero", "heroine",
                   "villain", "cast", "development"],
    "pacing":    ["pacing", "pace", "paced", "slow", "dragged", "drags", "fast-paced",
                  "rushed", "boring", "page-turner", "chapters", "middle", "put down",
                  "what happens next", "quick read"],
    "writing_style": ["writing", "prose", "style", "written", "language", "dialogue", "voice",
                      "descriptions", "sentences", "narration", "grammar", "typo", "typos",
                      "accent", "accents", "proofread", "proofreading", "technique"],
    "ending":    ["end", "ending", "ended", "finale", "conclusion", "climax", "epilogue",
                  "final chapter"],
}

# ---------------------------------------------------------------
# 2. Built-in test reviews (replace with real ones once this works)
#    Includes the hard cases on purpose: mixed, negation, sarcasm, spoiler recap.
# ---------------------------------------------------------------
SAMPLES = [
    ("The characters were excellent, but the story dragged in the middle.", 3),
    ("Loved the writing style. The ending was a huge letdown though.", 3),
    ("Not bad at all. The plot twists kept me guessing till the last page.", 5),
    ("Great premise but the pacing was painfully slow and the ending felt rushed.", 2),
    ("A true page-turner! The protagonist is wonderfully written.", 5),
    ("The villain kills the king, then the hero flees to the north.", 3),  # recap, not opinion
    ("Oh great, another predictable plot. Just what I needed.", 1),       # sarcasm
    ("Beautiful prose, but flat characters and no real development.", 3),
    ("I couldn't put it down. The ending was perfect.", 5),
    ("Terrible book. Boring, slow, and the writing is clumsy.", 1),
    ("The dialogue is sharp and the story is gripping from the start.", 5),
    ("The cover is nice and it arrived quickly.", 4),                     # no aspect at all
]


# ---------------------------------------------------------------
# 3. Helpers
# ---------------------------------------------------------------
nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")          # rule-based sentence splitter, no model download needed
vader = SentimentIntensityAnalyzer()

# Split mixed sentences into clauses so "X was great, but Y dragged" becomes 2 pieces.
CLAUSE_SPLIT = re.compile(r"\s*(?:;|,?\s+but\s+|,?\s+however,?\s+|,?\s+though\s*$|\s+yet\s+)", re.I)
HTML_TAG = re.compile(r"<[^>]+>")
PACING_OVER_PLOT = re.compile(
    r"\b(?:story|plot|narrative)\b.{0,25}\b(?:dragged|drags|slow|rushed|boring)\b",
    re.I,
)
READ_TO_END = re.compile(r"\b(?:read|reading).{0,12}\b(?:to|until)\s+the\s+end\b", re.I)


def prepare_text(text: str) -> str:
    return HTML_TAG.sub(" ", html.unescape(text)).replace("\xa0", " ")


def split_clauses(sentence: str):
    if not sentence or not sentence.strip():
        return []
    text = sentence
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

    parts = CLAUSE_SPLIT.split(text)
    clauses = [
        p.replace("__BUT_TOKEN__", "but").strip()
        for p in parts
        if p and p.strip()
    ]
    return clauses or [sentence.strip()]


def find_aspects(text: str):
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
    if "plot" in aspects and re.search(r"\bstory books?\b", normalized):
        plot_keywords = set(ASPECTS["plot"]) - {"story"}
        if not any(
            keyword in tokens if " " not in keyword else keyword in normalized
            for keyword in plot_keywords
        ):
            aspects.remove("plot")
    if "plot" in aspects and "pacing" in aspects and PACING_OVER_PLOT.search(normalized):
        aspects.remove("plot")
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


def load_reviews(path=None, full_dataset: bool = False):
    if path is None:
        return [
            {"review_id": str(index), "book_id": "built_in", "raw_text": text, "rating": rating}
            for index, (text, rating) in enumerate(SAMPLES)
        ]
    if str(path).lower().endswith(".csv"):
        df = pd.read_csv(path, keep_default_na=False)
    else:
        df = pd.read_json(path, lines=True)
    text_column = "cleaned_text" if "cleaned_text" in df.columns else ("raw_text" if "raw_text" in df.columns else "text")
    if not full_dataset and "20000" not in str(path) and len(df) > 20 and str(path) != "--full":
        df = df.head(20)
    reviews = []
    for index, row in df.iterrows():
        reviews.append({
            "review_id": row.get("review_id", f"review_{index+1:05d}"),
            "book_id": row.get("book_id", row.get("parent_asin", "")),
            "raw_text": row.get(text_column, ""),
            "rating": row.get("rating"),
        })
    return reviews


# ---------------------------------------------------------------
# 4. Run the chain
# ---------------------------------------------------------------
def run(reviews, splits_path: str = "data/book_splits.csv"):
    book_to_split = {}
    from pathlib import Path
    if splits_path and Path(splits_path).exists():
        df_sp = pd.read_csv(splits_path, keep_default_na=False)
        book_to_split = dict(zip(df_sp["book_id"].astype(str), df_sp["split"]))

    rows = []
    for review in reviews:
        review_id = str(review["review_id"])
        book_id = str(review["book_id"])
        split_name = book_to_split.get(book_id, "unknown")
        text = prepare_text(review["raw_text"])
        rating = review["rating"]
        if not text:
            rows.append({
                "review_id": review_id,
                "book_id": book_id,
                "split": split_name,
                "stars": rating,
                "whole_review_vader": 0.0,
                "sentence_id": f"{review_id}_s01",
                "sentence_text": "[empty review]",
                "clause_id": f"{review_id}_s01_c01",
                "clause": "[empty review]",
                "aspect": "none",
                "clause_vader": 0.0,
                "pred_sentiment": "neutral",
            })
            continue

        whole = vader.polarity_scores(text)["compound"]
        doc = nlp(text)
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        if not sents:
            sents = [text]

        for s_idx, sent in enumerate(sents, start=1):
            sentence_id = f"{review_id}_s{s_idx:02d}"
            clauses = split_clauses(sent)
            for c_idx, clause in enumerate(clauses, start=1):
                clause_id = f"{sentence_id}_c{c_idx:02d}"
                aspects = find_aspects(clause)
                score = vader.polarity_scores(clause)["compound"]
                for asp in aspects or ["none"]:
                    rows.append({
                        "review_id": review_id,
                        "book_id": book_id,
                        "split": split_name,
                        "stars": rating,
                        "whole_review_vader": round(whole, 2),
                        "sentence_id": sentence_id,
                        "sentence_text": sent,
                        "clause_id": clause_id,
                        "clause": clause,
                        "aspect": asp,
                        "clause_vader": round(score, 2),
                        "pred_sentiment": vader_label(score),
                    })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from pathlib import Path
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    is_full = (arg == "--full") or (arg and "20000" in str(arg))

    if is_full:
        data_path = "data/books_reviews_20000.csv" if arg == "--full" else arg
        output_path = "outputs/full_pipeline_output.csv"
        reviews = load_reviews(data_path, full_dataset=True)
        print(f"Running full pipeline on {len(reviews):,} reviews from {data_path}...")
        df = run(reviews)
        print(f"Processed {len(df):,} clauses across {df['review_id'].nunique():,} reviews.")
    else:
        reviews = load_reviews(arg, full_dataset=False)
        df = run(reviews)
        pd.set_option("display.max_colwidth", 60, "display.width", 200)
        print(df[["review_id", "stars", "whole_review_vader", "clause",
                  "aspect", "clause_vader", "pred_sentiment"]].to_string(index=False))
        output_path = "outputs/day1_pipeline_output.csv"

    print("\nClauses with NO aspect found:", (df.aspect == "none").sum(), "of", len(df))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {output_path}")

