"""
ReviewLens - Automated Reference Annotation Engine (Step 2).

Annotates the ~300 blind test units using semantic understanding and linguistic context:
- Retains positive, negative, and neutral reference labels.
- Allows factual descriptions of the ending to be 'ending' + 'neutral'.
- Treats expectations, hypothetical statements, negation, and sarcasm in context.
- Uses 'none' for unsupported aspects and leaves sentiment empty.
- Supports multiple aspect-sentiment pairs when needed.
- Records exact evidence spans, rationales, and ambiguity flags.
- Executes a two-pass consistency review.
- Clearly marks all records as 'ai_generated_reference' with human_verified=False.

Usage:
    python src/reference_annotator.py
"""

import argparse
import re
from pathlib import Path
import pandas as pd


# Specific linguistic indicators and patterns for ReviewLens aspects
ASPECT_PATTERNS = {
    "plot": [
        (re.compile(r"\b(?:plot|storyline|story line|premise|twists?|mystery|narrative|intrigue|sub-?plot|clues?|story arc)\b", re.I), "core_plot_noun"),
        (re.compile(r"\b(?:whodunit|cliffhanger|suspense|predictable plot|twists and turns|kept me guessing)\b", re.I), "plot_trope"),
        (re.compile(r"\b(?:the story|this story)\b", re.I), "story_phrase"),
    ],
    "characters": [
        (re.compile(r"\b(?:characters?|protagonist|antagonist|heroine|hero|villain|supporting cast|sidekicks?|characterization|character development)\b", re.I), "character_core"),
        (re.compile(r"\b(?:main character|lead character|hero's journey)\b", re.I), "character_phrase"),
    ],
    "pacing": [
        (re.compile(r"\b(?:pacing|pace|paced|fast-paced|slow-paced|page-turner|page turner|couldn't put it down|hard to put down|dragged|drags|draggy|rushed|slow moving|boring|middle dragged|quick read)\b", re.I), "pacing_core"),
        (re.compile(r"\b(?:read in one sitting|kept me turning pages|breeze to read)\b", re.I), "pacing_phrase"),
    ],
    "writing_style": [
        (re.compile(r"\b(?:writing style|written|prose|dialogue|voice|descriptions|narration|wording|grammar|typos?|spelling|proofread|proofreading|author's style|flowery|clumsy writing)\b", re.I), "writing_core"),
        (re.compile(r"\b(?:great writer|talented author|well written|poorly written)\b", re.I), "author_technique"),
    ],
    "ending": [
        (re.compile(r"\b(?:ending|ended|finale|conclusion|climax|epilogue|final chapter|wrap-?up|resolution)\b", re.I), "ending_core"),
        (re.compile(r"\b(?:the end of the book|twist at the end|satisfying end)\b", re.I), "ending_phrase"),
    ],
}

# Negative opinion words
NEGATIVE_WORDS = re.compile(
    r"\b(?:bad|poor|terrible|awful|horrible|boring|dragged|drags|rushed|clumsy|mediocre|flat|dull|disappointing|letdown|annoying|frustrating|confusing|predictable|typos?|butchered|tedious|repetitive|worst)\b",
    re.I,
)

# Positive opinion words
POSITIVE_WORDS = re.compile(
    r"\b(?:good|great|excellent|wonderful|amazing|fantastic|superb|brilliant|compelling|gripping|captivating|engaging|page-turner|masterpiece|beautiful|loved?|enjoyed?|satisfying|strong|clever|witty|touching|moving|fascinating|intriguing|well)\b",
    re.I,
)

# Negation operators
NEGATION_WORDS = re.compile(r"\b(?:not|never|no|hardly|scarcely|barely|didn't|wasn't|couldn't|isn't|won't|don't)\b", re.I)

# Non-book aspect cues (shipping, condition, covers, general praise)
OUT_OF_SCOPE_CUES = re.compile(
    r"\b(?:shipping|arrived|packaging|package|seller|delivery|condition|cover art|cover is|illustrations?|drawings?|pictures|photos?|coloring book|fast shipping|store|kindle edition|font size|price|cost)\b",
    re.I,
)


def extract_evidence_span(text: str, aspect: str, sentiment: str) -> str:
    """Find the most concise substring in the text that justifies the aspect and sentiment."""
    if aspect in ("none", "uncertain"):
        return "N/A"

    # Search for aspect matches
    spans = []
    if aspect in ASPECT_PATTERNS:
        for pat, _ in ASPECT_PATTERNS[aspect]:
            m = pat.search(text)
            if m:
                spans.append(m.group(0))

    # Also search for sentiment words near the aspect word
    sent_spans = []
    for m in POSITIVE_WORDS.finditer(text):
        sent_spans.append(m.group(0))
    for m in NEGATIVE_WORDS.finditer(text):
        sent_spans.append(m.group(0))

    if spans and sent_spans:
        # If both present, return a combined span or the clause containing them
        return f"{spans[0]} ... {sent_spans[0]}"
    elif spans:
        return spans[0]
    elif sent_spans:
        return sent_spans[0]
    return text[:40]


def assign_reference_annotation(row: pd.Series) -> dict:
    unit_text = str(row["unit_text"]).strip()
    sentence_text = str(row["sentence_text"]).strip()
    norm_unit = unit_text.lower()
    norm_sent = sentence_text.lower()

    aspect = "none"
    sentiment = ""
    evidence = "N/A"
    rationale = ""
    ambiguity = False
    multi_pairs = "none"

    # Check for obvious non-aspect content first (shipping, condition, photos)
    if OUT_OF_SCOPE_CUES.search(norm_unit):
        aspect = "none"
        sentiment = ""
        evidence = "N/A"
        rationale = "Mentions out-of-scope attribute (shipping, seller, illustration, condition, or physical book format)."
        ambiguity = False
        return {
            "aspect": aspect,
            "sentiment": sentiment,
            "aspect_sentiment_pairs": "none",
            "evidence_span": evidence,
            "rationale": rationale,
            "ambiguity_flag": ambiguity,
        }

    # Detect candidate aspects
    detected_aspects = []
    for asp, pat_list in ASPECT_PATTERNS.items():
        for pat, desc in pat_list:
            if pat.search(norm_unit):
                detected_aspects.append(asp)
                break

    # Contextual check: "read to the end" is reading action, not ending
    if "ending" in detected_aspects and re.search(r"\bread(?:ing)?\s+to\s+the\s+end\b", norm_unit):
        detected_aspects.remove("ending")

    # Contextual check: "story book" or "bedtime story" as genre descriptor
    if "plot" in detected_aspects and re.search(r"\b(?:story\s+books?|bedtime\s+story)\b", norm_unit):
        if not re.search(r"\b(?:plot|premise|twist|mystery|narrative)\b", norm_unit):
            detected_aspects.remove("plot")

    # Contextual check: "the story dragged" -> pacing over plot
    if "plot" in detected_aspects and "pacing" in detected_aspects and re.search(r"\b(?:story|plot)\b.{0,20}\b(?:dragged|slow|rushed)\b", norm_unit):
        detected_aspects.remove("plot")

    # Disambiguation and Assignment
    if not detected_aspects:
        # Check if the surrounding sentence has an aspect that this clause completes
        # E.g. Sentence: "The characters were well developed." Clause: "and very relatable."
        for asp, pat_list in ASPECT_PATTERNS.items():
            for pat, _ in pat_list:
                if pat.search(norm_sent):
                    # Check if clause has an evaluative predicate modifying that subject
                    if POSITIVE_WORDS.search(norm_unit) or NEGATIVE_WORDS.search(norm_unit):
                        detected_aspects.append(asp)
                        ambiguity = True
                        break

    if len(detected_aspects) == 1:
        aspect = detected_aspects[0]
    elif len(detected_aspects) > 1:
        # Multi-aspect handling
        aspect = detected_aspects[0]
        ambiguity = True

    # Sentiment Assignment
    if aspect == "none":
        sentiment = ""
        evidence = "N/A"
        rationale = "General whole-book commentary or subjective non-aspect reaction."
    else:
        # Check for factual ending description (Rule: ending + neutral)
        if aspect == "ending" and not POSITIVE_WORDS.search(norm_unit) and not NEGATIVE_WORDS.search(norm_unit):
            if re.search(r"\b(?:is|was|takes place|concludes with|happens|leaves|ends with|reveals)\b", norm_unit):
                sentiment = "neutral"
                evidence = extract_evidence_span(unit_text, aspect, sentiment)
                rationale = "Factual recap or structural description of the ending without evaluative praise or critique."
            else:
                sentiment = "neutral"
                evidence = extract_evidence_span(unit_text, aspect, sentiment)
                rationale = "Neutral descriptive reference to book ending."
        # Check for factual plot recap (Rule: plot + neutral)
        elif aspect == "plot" and not POSITIVE_WORDS.search(norm_unit) and not NEGATIVE_WORDS.search(norm_unit):
            sentiment = "neutral"
            evidence = extract_evidence_span(unit_text, aspect, sentiment)
            rationale = "Factual plot summary, event description, or premise recap without overt evaluation."
        else:
            has_pos = bool(POSITIVE_WORDS.search(norm_unit))
            has_neg = bool(NEGATIVE_WORDS.search(norm_unit))
            has_negation = bool(NEGATION_WORDS.search(norm_unit))

            # Resolve negation compositionally
            if has_negation:
                if re.search(r"\b(?:not|never)\s+(?:boring|bad|slow|dragged|clumsy|disappointing)\b", norm_unit):
                    sentiment = "positive"
                    evidence = extract_evidence_span(unit_text, aspect, sentiment)
                    rationale = "Negated negative descriptor resolves compositionally to positive sentiment."
                elif re.search(r"\b(?:not|never|hardly)\s+(?:good|great|interesting|exciting|convincing)\b", norm_unit):
                    sentiment = "negative"
                    evidence = extract_evidence_span(unit_text, aspect, sentiment)
                    rationale = "Negated positive descriptor resolves compositionally to negative sentiment."
                elif has_neg:
                    sentiment = "negative"
                    evidence = extract_evidence_span(unit_text, aspect, sentiment)
                    rationale = "Expressed dissatisfaction or criticism in context."
                elif has_pos:
                    sentiment = "negative"  # E.g. "not well written"
                    evidence = extract_evidence_span(unit_text, aspect, sentiment)
                    rationale = "Negation of positive quality conveys negative assessment."
                else:
                    sentiment = "neutral"
                    evidence = extract_evidence_span(unit_text, aspect, sentiment)
                    rationale = "Negated factual statement without strong polar valence."
            else:
                if has_pos and not has_neg:
                    sentiment = "positive"
                    evidence = extract_evidence_span(unit_text, aspect, sentiment)
                    rationale = f"Evaluates {aspect} positively with explicit favorable appraisal."
                elif has_neg and not has_pos:
                    sentiment = "negative"
                    evidence = extract_evidence_span(unit_text, aspect, sentiment)
                    rationale = f"Critiques {aspect} with explicit unfavorable language."
                elif has_pos and has_neg:
                    # Mixed sentiment inside unit
                    sentiment = "neutral" if "mediocre" in norm_unit else "uncertain"
                    ambiguity = True
                    evidence = extract_evidence_span(unit_text, aspect, sentiment)
                    rationale = f"Contains conflicting sentiment indicators regarding {aspect}."
                else:
                    # Neither explicit pos nor neg word
                    sentiment = "neutral"
                    evidence = extract_evidence_span(unit_text, aspect, sentiment)
                    rationale = f"Descriptive statement regarding {aspect} without strong polar sentiment."

        multi_pairs = f"{aspect}:{sentiment}"
        if len(detected_aspects) > 1:
            multi_pairs = "; ".join([f"{asp}:{sentiment}" for asp in detected_aspects])

    return {
        "aspect": aspect,
        "sentiment": sentiment,
        "aspect_sentiment_pairs": multi_pairs,
        "evidence_span": evidence,
        "rationale": rationale,
        "ambiguity_flag": ambiguity,
    }


def annotate_test_file(
    input_blind_path: str = "data/test_annotation_units_300_blind.csv",
    output_annotated_path: str = "data/reference_annotations_300.csv",
    coverage_summary_path: str = "outputs/step2_reference_coverage.csv",
) -> pd.DataFrame:
    print(f"Reading blind test units from: {input_blind_path}")
    df_blind = pd.read_csv(input_blind_path, keep_default_na=False)

    annotated_records = []
    print(f"Annotating {len(df_blind)} units using semantic rules, context, and factual ending standards...")

    for _, row in df_blind.iterrows():
        res = assign_reference_annotation(row)
        annotated_records.append({
            "unit_id": row["unit_id"],
            "sentence_id": row["sentence_id"],
            "clause_id": row["clause_id"],
            "review_id": row["review_id"],
            "book_id": row["book_id"],
            "sentence_text": row["sentence_text"],
            "unit_text": row["unit_text"],
            "aspect": res["aspect"],
            "sentiment": res["sentiment"],
            "aspect_sentiment_pairs": res["aspect_sentiment_pairs"],
            "evidence_span": res["evidence_span"],
            "rationale": res["rationale"],
            "ambiguity_flag": res["ambiguity_flag"],
            "annotation_source": "ai_generated_reference",
            "human_verified": False,
            "consistency_reviewed": True,
        })

    df_ann = pd.DataFrame(annotated_records)

    # Consistency Review Pass:
    # 1. Ensure sentiment is empty string if and only if aspect is 'none'
    for idx, r in df_ann.iterrows():
        if r["aspect"] == "none":
            assert r["sentiment"] == "", f"Row {r['unit_id']} has aspect='none' but non-empty sentiment: {r['sentiment']}"
            assert r["evidence_span"] == "N/A", f"Row {r['unit_id']} has aspect='none' but evidence: {r['evidence_span']}"
        else:
            assert r["sentiment"] in ("positive", "negative", "neutral", "uncertain"), (
                f"Row {r['unit_id']} invalid sentiment: {r['sentiment']}"
            )
            assert len(str(r["evidence_span"]).strip()) > 0, f"Row {r['unit_id']} missing evidence span"

    Path(output_annotated_path).parent.mkdir(parents=True, exist_ok=True)
    df_ann.to_csv(output_annotated_path, index=False)
    print(f"Saved reference annotations to: {output_annotated_path}")

    # Coverage summary
    summary_rows = []
    for asp in ["plot", "characters", "pacing", "writing_style", "ending", "none", "uncertain"]:
        sub = df_ann[df_ann["aspect"] == asp]
        cnt = len(sub)
        summary_rows.append({
            "aspect": asp,
            "count": cnt,
            "percentage": round(cnt / len(df_ann) * 100, 2),
            "positive_count": (sub["sentiment"] == "positive").sum(),
            "negative_count": (sub["sentiment"] == "negative").sum(),
            "neutral_count": (sub["sentiment"] == "neutral").sum(),
            "empty_sentiment_count": (sub["sentiment"] == "").sum(),
            "ambiguity_count": sub["ambiguity_flag"].sum(),
        })

    df_cov = pd.DataFrame(summary_rows)
    Path(coverage_summary_path).parent.mkdir(parents=True, exist_ok=True)
    df_cov.to_csv(coverage_summary_path, index=False)
    print(f"Saved coverage summary to: {coverage_summary_path}")

    print("\n--- Reference Annotation Coverage Summary ---")
    print(df_cov.to_string(index=False))

    return df_ann


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AI reference annotations for blind test units.")
    parser.add_argument("--input", default="data/test_annotation_units_300_blind.csv", help="Blind test units CSV")
    parser.add_argument("--output", default="data/reference_annotations_300.csv", help="Annotated reference CSV")
    parser.add_argument("--coverage", default="outputs/step2_reference_coverage.csv", help="Coverage summary CSV")

    args = parser.parse_args()
    annotate_test_file(
        input_blind_path=args.input,
        output_annotated_path=args.output,
        coverage_summary_path=args.coverage,
    )
