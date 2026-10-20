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
    return [c.strip() for c in CLAUSE_SPLIT.split(sentence) if c and c.strip()]


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


def load_reviews(path=None):
    if path is None:
        return [
            {"review_id": str(index), "book_id": "built_in", "raw_text": text, "rating": rating}
            for index, (text, rating) in enumerate(SAMPLES)
        ]
    if str(path).lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_json(path, lines=True)
    text_column = "raw_text" if "raw_text" in df.columns else "text"
    df = df.dropna(subset=[text_column, "rating"]).head(20)
    reviews = []
    for index, row in df.iterrows():
        reviews.append({
            "review_id": row.get("review_id", str(index)),
            "book_id": row.get("book_id", row.get("parent_asin", "")),
            "raw_text": row[text_column],
            "rating": row["rating"],
        })
    return reviews


# ---------------------------------------------------------------
# 4. Run the chain
# ---------------------------------------------------------------
def run(reviews):
    rows = []
    for review in reviews:
        review_id = review["review_id"]
        book_id = review["book_id"]
        text = prepare_text(review["raw_text"])
        rating = review["rating"]
        whole = vader.polarity_scores(text)["compound"]
        for sent in (s.text.strip() for s in nlp(text).sents):
            for clause in split_clauses(sent):
                aspects = find_aspects(clause)
                score = vader.polarity_scores(clause)["compound"]
                for asp in aspects or ["none"]:
                    rows.append({
                        "review_id": review_id,
                        "book_id": book_id,
                        "stars": rating,
                        "whole_review_vader": round(whole, 2),
                        "clause": clause,
                        "aspect": asp,
                        "clause_vader": round(score, 2),
                        "pred_sentiment": vader_label(score),
                    })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    reviews = load_reviews(sys.argv[1] if len(sys.argv) > 1 else None)
    df = run(reviews)

    pd.set_option("display.max_colwidth", 60, "display.width", 200)
    print(df[["review_id", "stars", "whole_review_vader", "clause",
              "aspect", "clause_vader", "pred_sentiment"]].to_string(index=False))

    print("\nClauses with NO aspect found:", (df.aspect == "none").sum(), "of", len(df))
    output_path = "outputs/day1_pipeline_output.csv"
    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {output_path}")
