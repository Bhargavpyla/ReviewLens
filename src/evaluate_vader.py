"""Evaluate the locked ReviewLens keyword and VADER baseline.

This script validates the Step 2 evaluation inputs before scoring. It does not
tune keywords, clause splitting, VADER thresholds, annotations, or test rows.

Usage:
    python src/evaluate_vader.py
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from run_pipeline import find_aspects, vader, vader_label


SUPPORTED_ASPECTS = ["plot", "characters", "pacing", "writing_style", "ending"]
SENTIMENTS = ["positive", "negative", "neutral"]
ALLOWED_ASPECTS = set(SUPPORTED_ASPECTS + ["none", "uncertain"])
ALLOWED_SENTIMENTS = set(SENTIMENTS + ["", "uncertain"])
ID_PATTERNS = {
    "review_id": re.compile(r"^review_\d{5}$"),
    "sentence_id": re.compile(r"^review_\d{5}_s\d{2,}$"),
    "clause_id": re.compile(r"^review_\d{5}_s\d{2,}_c\d{2,}$"),
    "unit_id": re.compile(r"^unit_\d{3}$"),
}
NEGATION_PATTERN = re.compile(
    r"\b(?:no|not|never|neither|nor|nothing|hardly|barely|didn't|doesn't|isn't|wasn't|weren't|can't|couldn't|wouldn't|won't)\b",
    re.I,
)
SARCASM_PATTERN = re.compile(
    r"\b(?:yeah right|oh great|just what i needed|as if|thanks a lot|wonderful, another)\b",
    re.I,
)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def parse_pairs(row: pd.Series) -> list[tuple[str, str]]:
    raw = str(row["aspect_sentiment_pairs"]).strip()
    if raw in {"", "none"} or row["aspect"] == "none":
        return []
    pairs = []
    for item in raw.split(";"):
        parts = [part.strip() for part in item.split(":", maxsplit=1)]
        if len(parts) != 2:
            raise ValueError(f"Malformed aspect_sentiment_pairs for {row['unit_id']}: {raw}")
        pairs.append((parts[0], parts[1]))
    return pairs


def add_check(checks: list[dict], name: str, passed: bool, detail: str) -> None:
    checks.append({"check": name, "status": "PASS" if passed else "FAIL", "detail": detail})
    if not passed:
        raise ValueError(f"{name}: {detail}")


def validate_inputs(
    master: pd.DataFrame,
    splits: pd.DataFrame,
    pipeline: pd.DataFrame,
    blind: pd.DataFrame,
    references: pd.DataFrame,
) -> tuple[list[dict], dict]:
    checks: list[dict] = []
    split_books = {
        name: set(splits.loc[splits["split"] == name, "book_id"])
        for name in ["train", "dev", "test"]
    }
    overlaps = {
        "train_dev": len(split_books["train"] & split_books["dev"]),
        "train_test": len(split_books["train"] & split_books["test"]),
        "dev_test": len(split_books["dev"] & split_books["test"]),
    }
    add_check(checks, "Book splits do not overlap", not any(overlaps.values()), str(overlaps))
    add_check(
        checks,
        "Book splits cover the master data",
        set(master["book_id"]) == set().union(*split_books.values()),
        f"master_books={master['book_id'].nunique()}, split_books={len(set().union(*split_books.values()))}",
    )

    required_reference_columns = {
        "unit_id", "sentence_id", "clause_id", "review_id", "book_id",
        "sentence_text", "unit_text", "aspect", "sentiment",
        "aspect_sentiment_pairs", "evidence_span", "rationale",
        "ambiguity_flag", "annotation_source", "human_verified",
        "consistency_reviewed",
    }
    missing_columns = required_reference_columns - set(references.columns)
    add_check(checks, "Reference schema is complete", not missing_columns, f"missing={sorted(missing_columns)}")
    add_check(
        checks,
        "Evaluation identifiers are unique",
        references["unit_id"].is_unique and references["clause_id"].is_unique,
        f"rows={len(references)}, unique_units={references['unit_id'].nunique()}, unique_clauses={references['clause_id'].nunique()}",
    )

    stable_ids = (
        references["unit_id"].map(lambda value: bool(ID_PATTERNS["unit_id"].match(value))).all()
        and references["review_id"].map(lambda value: bool(ID_PATTERNS["review_id"].match(value))).all()
        and references["sentence_id"].map(lambda value: bool(ID_PATTERNS["sentence_id"].match(value))).all()
        and references["clause_id"].map(lambda value: bool(ID_PATTERNS["clause_id"].match(value))).all()
        and (references["sentence_id"].str.rsplit("_s", n=1).str[0] == references["review_id"]).all()
        and (references["clause_id"].str.rsplit("_c", n=1).str[0] == references["sentence_id"]).all()
    )
    add_check(checks, "Review sentence and clause identifiers are stable", stable_ids, "all IDs match their documented hierarchy")

    blind_indexed = blind.set_index("unit_id")
    reference_indexed = references.set_index("unit_id")
    source_fields = ["sentence_id", "clause_id", "review_id", "book_id", "sentence_text", "unit_text"]
    same_units = set(blind_indexed.index) == set(reference_indexed.index)
    same_source = same_units and all(
        blind_indexed.loc[unit_id, field] == reference_indexed.loc[unit_id, field]
        for unit_id in reference_indexed.index
        for field in source_fields
    )
    add_check(checks, "Blind and reference source fields match", same_source, f"units={len(references)}")

    pipeline_units = pipeline[pipeline["clause_id"].isin(references["clause_id"])].copy()
    pipeline_source = pipeline_units.drop_duplicates("clause_id").set_index("clause_id")
    pipeline_match = set(references["clause_id"]) == set(pipeline_source.index) and all(
        pipeline_source.loc[row.clause_id, "review_id"] == row.review_id
        and pipeline_source.loc[row.clause_id, "sentence_id"] == row.sentence_id
        and pipeline_source.loc[row.clause_id, "book_id"] == row.book_id
        and pipeline_source.loc[row.clause_id, "sentence_text"] == row.sentence_text
        and pipeline_source.loc[row.clause_id, "clause_text"] == row.unit_text
        for row in references.itertuples()
    )
    add_check(checks, "Reference units match locked pipeline units", pipeline_match, f"matched_clauses={len(pipeline_source)}")

    master_by_review = master.set_index("review_id")
    evidence_failures = []
    source_text_failures = []
    for row in references.itertuples():
        review_text = normalize_text(master_by_review.loc[row.review_id, "cleaned_text"])
        if normalize_text(row.sentence_text) not in review_text or normalize_text(row.unit_text) not in review_text:
            source_text_failures.append(row.unit_id)
        if row.evidence_span != "N/A":
            fragments = [
                normalize_text(fragment).strip(" .")
                for fragment in re.split(r"\s+\.\.\.\s+", row.evidence_span)
                if normalize_text(fragment).strip(" .")
            ]
            if not fragments or any(fragment not in review_text for fragment in fragments):
                evidence_failures.append(row.unit_id)
    add_check(checks, "Sentence and clause text belongs to source review", not source_text_failures, f"failures={source_text_failures}")
    add_check(checks, "Evidence text belongs to source review", not evidence_failures, f"failures={evidence_failures}")

    invalid_aspects = sorted(set(references["aspect"]) - ALLOWED_ASPECTS)
    invalid_sentiments = sorted(set(references["sentiment"]) - ALLOWED_SENTIMENTS)
    pair_schema_errors = []
    for _, row in references.iterrows():
        try:
            pairs = parse_pairs(row)
        except ValueError:
            pair_schema_errors.append(row["unit_id"])
            continue
        if any(aspect not in ALLOWED_ASPECTS - {"none"} or sentiment not in ALLOWED_SENTIMENTS - {""} for aspect, sentiment in pairs):
            pair_schema_errors.append(row["unit_id"])
        if row["aspect"] == "none" and (row["sentiment"] != "" or pairs or row["evidence_span"] != "N/A"):
            pair_schema_errors.append(row["unit_id"])
        if row["aspect"] in SUPPORTED_ASPECTS and (not pairs or row["sentiment"] not in SENTIMENTS + ["uncertain"] or row["evidence_span"] in {"", "N/A"}):
            pair_schema_errors.append(row["unit_id"])
    add_check(checks, "Aspect label schema is valid", not invalid_aspects, f"invalid={invalid_aspects}")
    add_check(checks, "Sentiment label schema is valid", not invalid_sentiments, f"invalid={invalid_sentiments}")
    add_check(checks, "Aspect sentiment pairs are valid", not pair_schema_errors, f"failures={sorted(set(pair_schema_errors))}")

    incomplete = references[
        references["unit_id"].eq("")
        | references["sentence_id"].eq("")
        | references["clause_id"].eq("")
        | references["review_id"].eq("")
        | references["rationale"].str.strip().eq("")
        | ~references["consistency_reviewed"].map(as_bool)
    ]
    add_check(checks, "Annotations are complete", incomplete.empty, f"incomplete_rows={len(incomplete)}")

    provenance_ok = (
        set(references["annotation_source"]) == {"ai_generated_reference"}
        and set(references["human_verified"].map(as_bool)) == {False}
    )
    add_check(checks, "Reference provenance is AI generated and unverified", provenance_ok, "annotation_source=ai_generated_reference; human_verified=false")

    reference_books = set(references["book_id"])
    evaluation_pipeline_splits = set(pipeline_units["split"])
    test_only = reference_books.issubset(split_books["test"]) and evaluation_pipeline_splits == {"test"}
    add_check(checks, "Evaluation rows are test-only", test_only, f"reference_books={len(reference_books)}, pipeline_splits={sorted(evaluation_pipeline_splits)}")
    add_check(
        checks,
        "Evaluation uses no training rows",
        not (reference_books & split_books["train"]) and "train" not in evaluation_pipeline_splits,
        f"training_book_overlap={len(reference_books & split_books['train'])}",
    )

    forbidden_blind = {"aspect", "sentiment", "pred_sentiment", "clause_vader", "whole_review_vader", "stars"} & set(blind.columns)
    add_check(checks, "Blind file contains no labels or predictions", not forbidden_blind, f"forbidden={sorted(forbidden_blind)}")

    prediction_consistency = all(
        vader_label(float(row.clause_vader)) == row.pred_sentiment
        for row in pipeline_units.itertuples()
    )
    add_check(checks, "Saved VADER labels use the locked threshold", prediction_consistency, "thresholds remain +0.05 and -0.05")

    summary = {
        "train_books": len(split_books["train"]),
        "development_books": len(split_books["dev"]),
        "test_books": len(split_books["test"]),
        "split_overlap_count": sum(overlaps.values()),
        "reference_rows": len(references),
        "reference_books": len(reference_books),
        "evaluation_training_rows": 0,
    }
    return checks, summary


def class_metrics(y_true: list[str], y_pred: list[str], labels: list[str], scope: str, task: str) -> tuple[pd.DataFrame, dict]:
    rows = []
    for label in labels:
        true_positive = sum(actual == label and predicted == label for actual, predicted in zip(y_true, y_pred))
        false_positive = sum(actual != label and predicted == label for actual, predicted in zip(y_true, y_pred))
        false_negative = sum(actual == label and predicted != label for actual, predicted in zip(y_true, y_pred))
        precision = safe_divide(true_positive, true_positive + false_positive)
        recall = safe_divide(true_positive, true_positive + false_negative)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        rows.append({
            "task": task,
            "scope": scope,
            "class": label,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": sum(actual == label for actual in y_true),
            "predicted_count": sum(predicted == label for predicted in y_pred),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
        })
    accuracy = safe_divide(sum(actual == predicted for actual, predicted in zip(y_true, y_pred)), len(y_true))
    return pd.DataFrame(rows), {
        "task": task,
        "scope": scope,
        "eligible_rows": len(y_true),
        "accuracy": round(accuracy, 6),
        "macro_f1": round(sum(row["f1"] for row in rows) / len(labels), 6),
    }


def confusion_matrix(y_true: list[str], y_pred: list[str], actual_labels: list[str], predicted_labels: list[str], scope: str, task: str) -> pd.DataFrame:
    rows = []
    for actual in actual_labels:
        row = {"task": task, "scope": scope, "actual_label": actual}
        for predicted in predicted_labels:
            row[f"predicted_{predicted}"] = sum(
                truth == actual and guess == predicted
                for truth, guess in zip(y_true, y_pred)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_aspects(references: pd.DataFrame, pipeline: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict[str, set[str]]]:
    pipeline_subset = pipeline[pipeline["clause_id"].isin(references["clause_id"])]
    predicted_by_clause = {
        clause_id: set(group.loc[group["aspect"].isin(SUPPORTED_ASPECTS), "aspect"])
        for clause_id, group in pipeline_subset.groupby("clause_id", sort=False)
    }
    uncertain_units = set()
    gold_by_unit: dict[str, set[str]] = {}
    rows = []
    for _, reference in references.iterrows():
        pairs = parse_pairs(reference)
        if reference["aspect"] == "uncertain" or any(aspect == "uncertain" for aspect, _ in pairs):
            uncertain_units.add(reference["unit_id"])
            continue
        gold_aspects = {aspect for aspect, _ in pairs if aspect in SUPPORTED_ASPECTS}
        predicted_aspects = predicted_by_clause.get(reference["clause_id"], set())
        gold_by_unit[reference["unit_id"]] = gold_aspects
        rows.append({
            "unit_id": reference["unit_id"],
            "review_id": reference["review_id"],
            "book_id": reference["book_id"],
            "sentence_id": reference["sentence_id"],
            "clause_id": reference["clause_id"],
            "unit_text": reference["unit_text"],
            "reference_aspects": ";".join(sorted(gold_aspects)) if gold_aspects else "none",
            "predicted_aspects": ";".join(sorted(predicted_aspects)) if predicted_aspects else "none",
            "exact_match": gold_aspects == predicted_aspects,
            "false_positive_aspects": ";".join(sorted(predicted_aspects - gold_aspects)),
            "false_negative_aspects": ";".join(sorted(gold_aspects - predicted_aspects)),
            "reference_none": not gold_aspects,
            "false_detection_on_none": not gold_aspects and bool(predicted_aspects),
        })
    predictions = pd.DataFrame(rows).sort_values("unit_id").reset_index(drop=True)

    metric_rows = []
    totals = Counter()
    for aspect in SUPPORTED_ASPECTS:
        true_positive = false_positive = false_negative = true_negative = 0
        false_positive_on_none = 0
        for row in predictions.itertuples():
            gold = aspect in row.reference_aspects.split(";")
            predicted = aspect in row.predicted_aspects.split(";")
            if gold and predicted:
                true_positive += 1
            elif not gold and predicted:
                false_positive += 1
                false_positive_on_none += int(row.reference_none)
            elif gold and not predicted:
                false_negative += 1
            else:
                true_negative += 1
        precision = safe_divide(true_positive, true_positive + false_positive)
        recall = safe_divide(true_positive, true_positive + false_negative)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        metric_rows.append({
            "aspect": aspect,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": true_positive + false_negative,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
            "false_positive_on_none": false_positive_on_none,
        })
        totals.update(tp=true_positive, fp=false_positive, fn=false_negative)
    metrics = pd.DataFrame(metric_rows)
    micro_precision = safe_divide(totals["tp"], totals["tp"] + totals["fp"])
    micro_recall = safe_divide(totals["tp"], totals["tp"] + totals["fn"])
    micro_f1 = safe_divide(2 * micro_precision * micro_recall, micro_precision + micro_recall)
    none_rows = predictions[predictions["reference_none"]]
    summary = {
        "evaluated_units": len(predictions),
        "uncertain_units_excluded": len(uncertain_units),
        "reference_aspect_pairs": int(metrics["support"].sum()),
        "micro_precision": round(micro_precision, 6),
        "micro_recall": round(micro_recall, 6),
        "micro_f1": round(micro_f1, 6),
        "macro_precision": round(metrics["precision"].mean(), 6),
        "macro_recall": round(metrics["recall"].mean(), 6),
        "macro_f1": round(metrics["f1"].mean(), 6),
        "exact_match_accuracy": round(predictions["exact_match"].mean(), 6),
        "false_positive_count": int(totals["fp"]),
        "false_negative_count": int(totals["fn"]),
        "none_units": len(none_rows),
        "none_correct_units": int((~none_rows["false_detection_on_none"]).sum()),
        "none_false_detection_units": int(none_rows["false_detection_on_none"].sum()),
        "none_specificity": round((~none_rows["false_detection_on_none"]).mean(), 6),
    }
    return metrics, predictions, summary, gold_by_unit


def build_sentiment_rows(references: pd.DataFrame, pipeline: pd.DataFrame, gold_by_unit: dict[str, set[str]]) -> tuple[pd.DataFrame, dict]:
    pipeline_subset = pipeline[pipeline["clause_id"].isin(references["clause_id"])]
    clause_rows = pipeline_subset.drop_duplicates("clause_id").set_index("clause_id")
    detected = {
        clause_id: set(group.loc[group["aspect"].isin(SUPPORTED_ASPECTS), "aspect"])
        for clause_id, group in pipeline_subset.groupby("clause_id", sort=False)
    }
    rows = []
    excluded = Counter()
    for _, reference in references.iterrows():
        pairs = parse_pairs(reference)
        if reference["aspect"] == "none":
            excluded["none_units"] += 1
            if reference["sentiment"] == "":
                excluded["blank_sentiment_units"] += 1
            continue
        if reference["aspect"] == "uncertain" or any(aspect == "uncertain" for aspect, _ in pairs):
            excluded["uncertain_aspect_units"] += 1
            continue
        for aspect, sentiment in pairs:
            if sentiment == "uncertain":
                excluded["uncertain_sentiment_pairs"] += 1
                continue
            if sentiment == "":
                excluded["blank_sentiment_pairs"] += 1
                continue
            if aspect not in SUPPORTED_ASPECTS or sentiment not in SENTIMENTS:
                continue
            saved = clause_rows.loc[reference["clause_id"]]
            aspect_detected = aspect in detected.get(reference["clause_id"], set())
            aspect_prediction = saved["pred_sentiment"] if aspect_detected else "missing_aspect"
            whole_prediction = vader_label(float(saved["whole_review_vader"])) if aspect_detected else "missing_aspect"
            sentence_prediction = vader_label(vader.polarity_scores(reference["sentence_text"])["compound"])
            pair_id = f"{reference['unit_id']}__{aspect}"
            rows.append({
                "pair_id": pair_id,
                "unit_id": reference["unit_id"],
                "review_id": reference["review_id"],
                "book_id": reference["book_id"],
                "sentence_id": reference["sentence_id"],
                "clause_id": reference["clause_id"],
                "reference_aspect": aspect,
                "reference_sentiment": sentiment,
                "aspect_detected": aspect_detected,
                "aspect_text_prediction": aspect_prediction,
                "whole_review_prediction": whole_prediction,
                "sentence_text_prediction": sentence_prediction,
                "aspect_text_correct": aspect_prediction == sentiment,
                "whole_review_correct": whole_prediction == sentiment,
                "unit_text": reference["unit_text"],
                "sentence_text": reference["sentence_text"],
                "whole_review_text": "",
            })
    sentiment_rows = pd.DataFrame(rows).sort_values("pair_id").reset_index(drop=True)
    master_exclusions = {
        "none_units": excluded["none_units"],
        "blank_sentiment_units": excluded["blank_sentiment_units"],
        "uncertain_aspect_units": excluded["uncertain_aspect_units"],
        "uncertain_sentiment_pairs": excluded["uncertain_sentiment_pairs"],
        "blank_sentiment_pairs": excluded["blank_sentiment_pairs"],
        "neutral_pairs_excluded_from_binary": int((sentiment_rows["reference_sentiment"] == "neutral").sum()),
    }
    return sentiment_rows, master_exclusions


def evaluate_sentiment(sentiment_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    all_metric_tables = []
    three_confusions = []
    binary_confusions = []
    comparison_rows = []
    summaries = []
    prediction_labels = SENTIMENTS + ["missing_aspect"]
    for task, subset, labels in [
        ("three_class", sentiment_rows, SENTIMENTS),
        ("binary_positive_negative", sentiment_rows[sentiment_rows["reference_sentiment"].isin(["positive", "negative"])], ["positive", "negative"]),
    ]:
        pair_ids_by_scope = {}
        task_summaries = {}
        for scope, prediction_column in [
            ("aspect_text", "aspect_text_prediction"),
            ("whole_review", "whole_review_prediction"),
        ]:
            y_true = subset["reference_sentiment"].tolist()
            y_pred = subset[prediction_column].tolist()
            metrics, summary = class_metrics(y_true, y_pred, labels, scope, task)
            all_metric_tables.append(metrics)
            summaries.append(summary)
            task_summaries[scope] = summary
            pair_ids_by_scope[scope] = subset["pair_id"].tolist()
            matrix = confusion_matrix(y_true, y_pred, labels, prediction_labels, scope, task)
            if task == "three_class":
                three_confusions.append(matrix)
            else:
                binary_confusions.append(matrix)
            comparison_rows.append({
                "task": task,
                "scope": scope,
                "eligible_rows": len(subset),
                "accuracy": summary["accuracy"],
                "macro_f1": summary["macro_f1"],
                "identical_reference_rows": True,
            })
        if pair_ids_by_scope["aspect_text"] != pair_ids_by_scope["whole_review"]:
            raise ValueError(f"{task} comparison scopes do not use identical rows")
        comparison_rows.append({
            "task": task,
            "scope": "aspect_text_minus_whole_review",
            "eligible_rows": len(subset),
            "accuracy": round(task_summaries["aspect_text"]["accuracy"] - task_summaries["whole_review"]["accuracy"], 6),
            "macro_f1": round(task_summaries["aspect_text"]["macro_f1"] - task_summaries["whole_review"]["macro_f1"], 6),
            "identical_reference_rows": True,
        })
    metrics = pd.concat(all_metric_tables, ignore_index=True)
    three_metrics = metrics[metrics["task"] == "three_class"].reset_index(drop=True)
    binary_metrics = metrics[metrics["task"] == "binary_positive_negative"].reset_index(drop=True)
    return (
        three_metrics,
        binary_metrics,
        pd.concat(three_confusions, ignore_index=True),
        pd.concat(binary_confusions, ignore_index=True),
        pd.DataFrame(comparison_rows),
        summaries,
    )


def build_row_predictions(aspect_predictions: pd.DataFrame, sentiment_rows: pd.DataFrame) -> pd.DataFrame:
    sentiment_by_unit = {unit_id: group for unit_id, group in sentiment_rows.groupby("unit_id")}
    rows = []
    for aspect_row in aspect_predictions.itertuples():
        unit_sentiments = sentiment_by_unit.get(aspect_row.unit_id)
        if unit_sentiments is None:
            rows.append({
                **aspect_row._asdict(),
                "pair_id": "",
                "reference_aspect": "none",
                "reference_sentiment": "",
                "aspect_detected": "",
                "aspect_text_prediction": "",
                "whole_review_prediction": "",
                "aspect_text_correct": "",
                "whole_review_correct": "",
            })
            continue
        for sentiment_row in unit_sentiments.itertuples():
            rows.append({
                **aspect_row._asdict(),
                "pair_id": sentiment_row.pair_id,
                "reference_aspect": sentiment_row.reference_aspect,
                "reference_sentiment": sentiment_row.reference_sentiment,
                "aspect_detected": sentiment_row.aspect_detected,
                "aspect_text_prediction": sentiment_row.aspect_text_prediction,
                "whole_review_prediction": sentiment_row.whole_review_prediction,
                "aspect_text_correct": sentiment_row.aspect_text_correct,
                "whole_review_correct": sentiment_row.whole_review_correct,
            })
    return pd.DataFrame(rows).sort_values(["unit_id", "reference_aspect"]).reset_index(drop=True)


def build_errors(aspect_predictions: pd.DataFrame, sentiment_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    errors = []
    for row in aspect_predictions.itertuples():
        for aspect in filter(None, row.false_positive_aspects.split(";")):
            errors.append({"error_type": "false_aspect_detection", "unit_id": row.unit_id, "pair_id": "", "aspect": aspect, "reference_sentiment": "", "prediction": aspect, "scope": "aspect_detection", "unit_text": row.unit_text, "sentence_text": "", "note": "Predicted aspect absent from the AI reference."})
        for aspect in filter(None, row.false_negative_aspects.split(";")):
            errors.append({"error_type": "missed_aspect", "unit_id": row.unit_id, "pair_id": f"{row.unit_id}__{aspect}", "aspect": aspect, "reference_sentiment": "", "prediction": "none", "scope": "aspect_detection", "unit_text": row.unit_text, "sentence_text": "", "note": "AI reference aspect was not detected."})
    for row in sentiment_rows.itertuples():
        for scope, prediction in [("aspect_text", row.aspect_text_prediction), ("whole_review", row.whole_review_prediction)]:
            if prediction != row.reference_sentiment:
                errors.append({"error_type": "incorrect_sentiment", "unit_id": row.unit_id, "pair_id": row.pair_id, "aspect": row.reference_aspect, "reference_sentiment": row.reference_sentiment, "prediction": prediction, "scope": scope, "unit_text": row.unit_text, "sentence_text": row.sentence_text, "note": "Prediction includes aspect matching; missing_aspect is incorrect."})
        sentence_aspects = set(find_aspects(row.sentence_text))
        context_failure = (
            (not row.aspect_detected and row.reference_aspect in sentence_aspects)
            or (row.aspect_text_prediction != row.reference_sentiment and row.sentence_text_prediction == row.reference_sentiment)
        )
        if context_failure:
            errors.append({"error_type": "clause_context_failure", "unit_id": row.unit_id, "pair_id": row.pair_id, "aspect": row.reference_aspect, "reference_sentiment": row.reference_sentiment, "prediction": row.aspect_text_prediction, "scope": "aspect_text", "unit_text": row.unit_text, "sentence_text": row.sentence_text, "note": "Sentence context restores the aspect or the reference sentiment."})
        if row.reference_sentiment == "neutral" and row.reference_aspect in {"plot", "ending"} and row.aspect_text_prediction != "neutral":
            errors.append({"error_type": "neutral_recap_error", "unit_id": row.unit_id, "pair_id": row.pair_id, "aspect": row.reference_aspect, "reference_sentiment": row.reference_sentiment, "prediction": row.aspect_text_prediction, "scope": "aspect_text", "unit_text": row.unit_text, "sentence_text": row.sentence_text, "note": "Neutral plot or ending recap received non-neutral sentiment."})
        if NEGATION_PATTERN.search(row.unit_text) and row.aspect_text_prediction != row.reference_sentiment:
            errors.append({"error_type": "negation_error", "unit_id": row.unit_id, "pair_id": row.pair_id, "aspect": row.reference_aspect, "reference_sentiment": row.reference_sentiment, "prediction": row.aspect_text_prediction, "scope": "aspect_text", "unit_text": row.unit_text, "sentence_text": row.sentence_text, "note": "Incorrect aspect-text sentiment on a unit containing negation."})
        if SARCASM_PATTERN.search(row.unit_text) and row.aspect_text_prediction != row.reference_sentiment:
            errors.append({"error_type": "sarcasm_error", "unit_id": row.unit_id, "pair_id": row.pair_id, "aspect": row.reference_aspect, "reference_sentiment": row.reference_sentiment, "prediction": row.aspect_text_prediction, "scope": "aspect_text", "unit_text": row.unit_text, "sentence_text": row.sentence_text, "note": "Potential sarcasm pattern with incorrect sentiment."})
    columns = ["error_type", "unit_id", "pair_id", "aspect", "reference_sentiment", "prediction", "scope", "unit_text", "sentence_text", "note"]
    error_frame = pd.DataFrame(errors, columns=columns).drop_duplicates().sort_values(["error_type", "unit_id", "pair_id", "scope"]).reset_index(drop=True)
    expected_types = ["false_aspect_detection", "missed_aspect", "incorrect_sentiment", "sarcasm_error", "negation_error", "neutral_recap_error", "clause_context_failure"]
    summary = pd.DataFrame([
        {"error_type": error_type, "example_count": int((error_frame["error_type"] == error_type).sum())}
        for error_type in expected_types
    ])
    return error_frame, summary


def overall_metrics(aspect_summary: dict, sentiment_summaries: list[dict], exclusions: dict) -> pd.DataFrame:
    rows = []
    for metric, value in aspect_summary.items():
        rows.append({"section": "aspect_detection", "scope": "all", "metric": metric, "value": value})
    for summary in sentiment_summaries:
        for metric in ["eligible_rows", "accuracy", "macro_f1"]:
            rows.append({"section": summary["task"], "scope": summary["scope"], "metric": metric, "value": summary[metric]})
    for metric, value in exclusions.items():
        rows.append({"section": "exclusions", "scope": "reference", "metric": metric, "value": value})
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    master = read_csv(Path(args.master))
    splits = read_csv(Path(args.splits))
    pipeline = read_csv(Path(args.pipeline))
    blind = read_csv(Path(args.blind))
    references = read_csv(Path(args.references))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checks, validation_summary = validate_inputs(master, splits, pipeline, blind, references)
    aspect_metrics, aspect_predictions, aspect_summary, gold_by_unit = evaluate_aspects(references, pipeline)
    sentiment_rows, exclusions = build_sentiment_rows(references, pipeline, gold_by_unit)
    three_metrics, binary_metrics, three_confusion, binary_confusion, scope_comparison, sentiment_summaries = evaluate_sentiment(sentiment_rows)
    row_predictions = build_row_predictions(aspect_predictions, sentiment_rows)
    errors, error_summary = build_errors(aspect_predictions, sentiment_rows)

    add_check(checks, "Aspect supports reconcile", int(aspect_metrics["support"].sum()) == aspect_summary["reference_aspect_pairs"], f"support_sum={int(aspect_metrics['support'].sum())}, reference_pairs={aspect_summary['reference_aspect_pairs']}")
    add_check(checks, "Sentiment supports reconcile", all(int(group["support"].sum()) == len(sentiment_rows) for _, group in three_metrics.groupby("scope")), f"eligible_pairs={len(sentiment_rows)}")
    binary_count = int(sentiment_rows["reference_sentiment"].isin(["positive", "negative"]).sum())
    add_check(checks, "Binary supports reconcile", all(int(group["support"].sum()) == binary_count for _, group in binary_metrics.groupby("scope")), f"eligible_binary_pairs={binary_count}")
    add_check(checks, "Scope comparisons use identical eligible rows", scope_comparison["identical_reference_rows"].all(), f"three_class_rows={len(sentiment_rows)}, binary_rows={binary_count}")
    add_check(checks, "Sentiment pair identifiers are unique", sentiment_rows["pair_id"].is_unique, f"pairs={len(sentiment_rows)}, unique={sentiment_rows['pair_id'].nunique()}")

    validation_report = {
        "status": "PASS",
        "reference_provenance": "ai_generated_reference",
        "human_verified": False,
        "interpretation": "Metrics measure agreement with AI-generated reference annotations, not human-labelled gold accuracy.",
        "locked_test_policy": "No keywords, clause splitting, VADER thresholds, labels, or test examples were changed.",
        "summary": {**validation_summary, **exclusions},
        "checks": checks,
    }
    with (output_dir / "validation_report.json").open("w", encoding="utf-8") as handle:
        json.dump(validation_report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    pd.DataFrame(checks).to_csv(output_dir / "validation_checks.csv", index=False)
    overall_metrics(aspect_summary, sentiment_summaries, exclusions).to_csv(output_dir / "overall_metrics.csv", index=False)
    aspect_metrics.to_csv(output_dir / "aspect_metrics.csv", index=False)
    three_metrics.to_csv(output_dir / "sentiment_three_class_metrics.csv", index=False)
    binary_metrics.to_csv(output_dir / "sentiment_binary_metrics.csv", index=False)
    three_confusion.to_csv(output_dir / "sentiment_three_class_confusion.csv", index=False)
    binary_confusion.to_csv(output_dir / "sentiment_binary_confusion.csv", index=False)
    scope_comparison.to_csv(output_dir / "scope_comparison.csv", index=False)
    row_predictions.to_csv(output_dir / "row_predictions.csv", index=False)
    errors.to_csv(output_dir / "error_examples.csv", index=False)
    error_summary.to_csv(output_dir / "error_summary.csv", index=False)

    three_compare = scope_comparison[scope_comparison["task"] == "three_class"].set_index("scope")
    binary_compare = scope_comparison[scope_comparison["task"] == "binary_positive_negative"].set_index("scope")
    print("STEP 3 EVALUATION PASSED")
    print(f"Validation checks: {len(checks)}/{len(checks)} PASS")
    print(f"Aspect micro-F1: {aspect_summary['micro_f1']:.6f}; macro-F1: {aspect_summary['macro_f1']:.6f}")
    print(f"Three-class aspect-text macro-F1: {three_compare.loc['aspect_text', 'macro_f1']:.6f}")
    print(f"Three-class whole-review macro-F1: {three_compare.loc['whole_review', 'macro_f1']:.6f}")
    print(f"Three-class macro-F1 difference: {three_compare.loc['aspect_text_minus_whole_review', 'macro_f1']:.6f}")
    print(f"Binary aspect-text macro-F1: {binary_compare.loc['aspect_text', 'macro_f1']:.6f}")
    print(f"Outputs: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Step 2 and evaluate the locked ReviewLens VADER baseline")
    parser.add_argument("--master", default="data/books_reviews_20000.csv")
    parser.add_argument("--splits", default="data/book_splits.csv")
    parser.add_argument("--pipeline", default="outputs/full_pipeline_output.csv")
    parser.add_argument("--blind", default="data/test_annotation_units_300_blind.csv")
    parser.add_argument("--references", default="data/reference_annotations_300.csv")
    parser.add_argument("--output-dir", default="outputs/step3")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
