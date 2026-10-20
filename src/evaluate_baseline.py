"""Evaluate the locked ReviewLens keyword + VADER baseline on Step 2 references."""

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from run_pipeline import find_aspects, vader, vader_label


ASPECTS = ["plot", "characters", "pacing", "writing_style", "ending"]
SENTIMENTS = ["positive", "negative", "neutral"]
REFERENCE_COLUMNS = [
    "unit_id",
    "sentence_id",
    "clause_id",
    "review_id",
    "book_id",
    "sentence_text",
    "unit_text",
    "aspect",
    "sentiment",
    "aspect_sentiment_pairs",
    "evidence_span",
    "rationale",
    "ambiguity_flag",
    "annotation_source",
    "human_verified",
    "consistency_reviewed",
]


def safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def parse_reference_pairs(row: pd.Series) -> list[tuple[str, str]]:
    raw_pairs = str(row["aspect_sentiment_pairs"]).strip()
    if row["aspect"] == "none" or raw_pairs in {"", "none"}:
        return []

    pairs = []
    for raw_pair in raw_pairs.split(";"):
        parts = raw_pair.strip().split(":", maxsplit=1)
        if len(parts) != 2:
            continue
        aspect, sentiment = (part.strip() for part in parts)
        pairs.append((aspect, sentiment))

    if not pairs and row["aspect"] in ASPECTS:
        pairs.append((str(row["aspect"]), str(row["sentiment"])))
    return pairs


def classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
    scope: str,
) -> tuple[pd.DataFrame, dict]:
    rows = []
    for label in labels:
        true_positive = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        false_positive = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        false_negative = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        support = sum(t == label for t in y_true)
        predicted_count = sum(p == label for p in y_pred)
        precision = safe_divide(true_positive, true_positive + false_positive)
        recall = safe_divide(true_positive, true_positive + false_negative)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        rows.append(
            {
                "scope": scope,
                "class": label,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "support": support,
                "predicted_count": predicted_count,
            }
        )

    accuracy = safe_divide(sum(t == p for t, p in zip(y_true, y_pred)), len(y_true))
    macro_f1 = sum(row["f1"] for row in rows) / len(labels)
    weighted_f1 = safe_divide(
        sum(row["f1"] * row["support"] for row in rows),
        len(y_true),
    )
    summary = {
        "scope": scope,
        "example_count": len(y_true),
        "accuracy": round(accuracy, 6),
        "macro_f1": round(macro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
    }
    return pd.DataFrame(rows), summary


def confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
    scope: str,
) -> pd.DataFrame:
    rows = []
    for actual in labels:
        row = {"scope": scope, "actual_label": actual}
        for predicted in labels:
            row[f"predicted_{predicted}"] = sum(
                t == actual and p == predicted for t, p in zip(y_true, y_pred)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def validate_step2(
    master: pd.DataFrame,
    splits: pd.DataFrame,
    pipeline: pd.DataFrame,
    blind: pd.DataFrame,
    references: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    checks = []

    def add_check(check: str, passed: bool, detail: str) -> None:
        checks.append({"check": check, "status": "PASS" if passed else "FAIL", "detail": detail})
        if not passed:
            raise ValueError(f"{check}: {detail}")

    split_books = {
        split: set(splits.loc[splits["split"] == split, "book_id"].astype(str))
        for split in ["train", "dev", "test"]
    }
    overlap_count = sum(
        len(split_books[left] & split_books[right])
        for left, right in [("train", "dev"), ("train", "test"), ("dev", "test")]
    )
    add_check("Book split overlap", overlap_count == 0, f"overlap_count={overlap_count}")

    all_split_books = set().union(*split_books.values())
    master_books = set(master["book_id"].astype(str))
    add_check(
        "Master book split coverage",
        all_split_books == master_books,
        f"master_books={len(master_books)}, split_books={len(all_split_books)}",
    )

    required_columns_present = set(REFERENCE_COLUMNS).issubset(references.columns)
    add_check(
        "Reference schema",
        required_columns_present,
        f"required_columns_present={required_columns_present}",
    )
    add_check(
        "Reference row completeness",
        (
            len(references) == 300
            and references["unit_id"].nunique() == 300
            and references["clause_id"].nunique() == 300
        ),
        (
            f"rows={len(references)}, unique_units={references['unit_id'].nunique()}, "
            f"unique_clauses={references['clause_id'].nunique()}"
        ),
    )

    forbidden_blind_columns = {
        "aspect",
        "sentiment",
        "pred_sentiment",
        "clause_vader",
        "whole_review_vader",
        "stars",
    }
    present_forbidden_columns = forbidden_blind_columns & set(blind.columns)
    add_check(
        "Blind file contains no labels or predictions",
        not present_forbidden_columns,
        f"forbidden_columns_present={sorted(present_forbidden_columns)}",
    )

    blind_by_id = blind.set_index("unit_id")
    ref_by_id = references.set_index("unit_id")
    same_units = set(blind_by_id.index) == set(ref_by_id.index)
    same_source_fields = same_units and all(
        blind_by_id.loc[unit_id, field] == ref_by_id.loc[unit_id, field]
        for unit_id in ref_by_id.index
        for field in ["sentence_id", "clause_id", "review_id", "book_id", "sentence_text", "unit_text"]
    )
    add_check(
        "Blind/reference identity",
        same_source_fields,
        "all reference source fields match the prediction-free blind file",
    )

    reference_books = set(references["book_id"].astype(str))
    add_check(
        "Reference books restricted to test",
        reference_books.issubset(split_books["test"]),
        f"reference_books={len(reference_books)}, non_test_books={len(reference_books - split_books['test'])}",
    )

    valid_aspects = set(ASPECTS + ["none", "uncertain"])
    invalid_aspects = set(references["aspect"]) - valid_aspects
    invalid_sentiments = set(references["sentiment"]) - set(SENTIMENTS + ["", "uncertain"])
    none_sentiment_errors = (
        (references["aspect"] == "none") & (references["sentiment"] != "")
    ).sum()
    aspect_sentiment_errors = (
        references["aspect"].isin(ASPECTS) & ~references["sentiment"].isin(SENTIMENTS + ["uncertain"])
    ).sum()
    none_evidence_errors = (
        (references["aspect"] == "none") & (references["evidence_span"] != "N/A")
    ).sum()
    aspect_evidence_errors = (
        references["aspect"].isin(ASPECTS)
        & references["evidence_span"].isin(["", "N/A"])
    ).sum()
    missing_rationales = references["rationale"].astype(str).str.strip().eq("").sum()
    unreviewed_annotations = (~references["consistency_reviewed"].map(as_bool)).sum()
    annotations_complete = (
        not invalid_aspects
        and not invalid_sentiments
        and none_sentiment_errors == 0
        and aspect_sentiment_errors == 0
        and none_evidence_errors == 0
        and aspect_evidence_errors == 0
        and missing_rationales == 0
        and unreviewed_annotations == 0
        and (references["aspect"] != "").all()
    )
    add_check(
        "Reference annotation completeness",
        annotations_complete,
        (
            f"invalid_aspects={sorted(invalid_aspects)}, invalid_sentiments={sorted(invalid_sentiments)}, "
            f"sentiment_errors={none_sentiment_errors + aspect_sentiment_errors}, "
            f"evidence_errors={none_evidence_errors + aspect_evidence_errors}, "
            f"missing_rationales={missing_rationales}, unreviewed_annotations={unreviewed_annotations}"
        ),
    )

    reference_sources = set(references["annotation_source"])
    human_verified_values = {as_bool(value) for value in references["human_verified"]}
    add_check(
        "Reference provenance",
        reference_sources == {"ai_generated_reference"} and human_verified_values == {False},
        f"annotation_source={sorted(reference_sources)}, human_verified={sorted(human_verified_values)}",
    )

    pipeline_clause_ids = set(pipeline["clause_id"].astype(str))
    missing_pipeline_units = set(references["clause_id"].astype(str)) - pipeline_clause_ids
    reference_pipeline_splits = set(
        pipeline.loc[pipeline["clause_id"].isin(references["clause_id"]), "split"]
    )
    add_check(
        "Reference units available in locked test pipeline",
        not missing_pipeline_units and reference_pipeline_splits == {"test"},
        f"missing_units={len(missing_pipeline_units)}, pipeline_splits={sorted(reference_pipeline_splits)}",
    )

    split_map = dict(zip(splits["book_id"].astype(str), splits["split"]))
    master_with_split = master.copy()
    master_with_split["split"] = master_with_split["book_id"].astype(str).map(split_map)
    ratings = pd.to_numeric(master_with_split["rating"], errors="coerce")
    weak_train = master_with_split[
        (master_with_split["split"] == "train") & ratings.isin([1.0, 2.0, 4.0, 5.0])
    ]
    training_reference_overlap = set(weak_train["book_id"].astype(str)) & reference_books
    add_check(
        "Test books excluded from weak-label training candidates",
        not training_reference_overlap,
        (
            f"eligible_train_reviews={len(weak_train)}, "
            f"reference_book_overlap={len(training_reference_overlap)}"
        ),
    )

    unresolved_aspect_count = int((references["aspect"] == "uncertain").sum())
    unresolved_sentiment_count = int((references["sentiment"] == "uncertain").sum())
    validation_meta = {
        "train_book_count": len(split_books["train"]),
        "dev_book_count": len(split_books["dev"]),
        "test_book_count": len(split_books["test"]),
        "reference_unit_count": len(references),
        "reference_book_count": len(reference_books),
        "weak_train_candidate_review_count": len(weak_train),
        "unresolved_aspect_count": unresolved_aspect_count,
        "unresolved_sentiment_count": unresolved_sentiment_count,
        "annotation_source": "ai_generated_reference",
        "human_verified": False,
    }
    return pd.DataFrame(checks), validation_meta


def build_predictions(
    master: pd.DataFrame,
    pipeline: pd.DataFrame,
    references: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference_clause_ids = set(references["clause_id"].astype(str))
    pipeline_subset = pipeline[pipeline["clause_id"].astype(str).isin(reference_clause_ids)].copy()
    pipeline_groups = {clause_id: group for clause_id, group in pipeline_subset.groupby("clause_id")}
    review_text_map = dict(zip(master["review_id"].astype(str), master["cleaned_text"].astype(str)))

    unit_rows = []
    pair_rows = []
    for _, reference in references.iterrows():
        clause_id = str(reference["clause_id"])
        group = pipeline_groups[clause_id]
        predicted_aspects = sorted(set(group["aspect"]) - {"none"})
        gold_pairs = parse_reference_pairs(reference)
        gold_aspects = sorted({aspect for aspect, _ in gold_pairs if aspect in ASPECTS})

        unit_score = float(group.iloc[0]["clause_vader"])
        unit_label = str(group.iloc[0]["pred_sentiment"])
        sentence_score = vader.polarity_scores(str(reference["sentence_text"]))["compound"]
        sentence_label = vader_label(sentence_score)
        review_text = review_text_map[str(reference["review_id"])]
        whole_score = vader.polarity_scores(review_text)["compound"]
        whole_label = vader_label(whole_score)

        unit_rows.append(
            {
                "unit_id": reference["unit_id"],
                "review_id": reference["review_id"],
                "book_id": reference["book_id"],
                "sentence_id": reference["sentence_id"],
                "clause_id": clause_id,
                "sentence_text": reference["sentence_text"],
                "unit_text": reference["unit_text"],
                "gold_aspects": ";".join(gold_aspects) if gold_aspects else "none",
                "predicted_aspects": ";".join(predicted_aspects) if predicted_aspects else "none",
                "aspect_exact_match": set(gold_aspects) == set(predicted_aspects),
                "aspect_text_vader_score": round(unit_score, 6),
                "aspect_text_vader_label": unit_label,
                "sentence_vader_score": round(sentence_score, 6),
                "sentence_vader_label": sentence_label,
                "whole_review_vader_score": round(whole_score, 6),
                "whole_review_vader_label": whole_label,
                "ambiguity_flag": as_bool(reference["ambiguity_flag"]),
                "annotation_source": reference["annotation_source"],
                "human_verified": as_bool(reference["human_verified"]),
            }
        )

        for gold_aspect, gold_sentiment in gold_pairs:
            pair_rows.append(
                {
                    "unit_id": reference["unit_id"],
                    "review_id": reference["review_id"],
                    "book_id": reference["book_id"],
                    "gold_aspect": gold_aspect,
                    "gold_sentiment": gold_sentiment,
                    "aspect_text_vader_score": round(unit_score, 6),
                    "aspect_text_vader_label": unit_label,
                    "sentence_vader_score": round(sentence_score, 6),
                    "sentence_vader_label": sentence_label,
                    "whole_review_vader_score": round(whole_score, 6),
                    "whole_review_vader_label": whole_label,
                    "ambiguity_flag": as_bool(reference["ambiguity_flag"]),
                }
            )

    return pd.DataFrame(unit_rows), pd.DataFrame(pair_rows)


def aspect_detection_metrics(unit_predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    total_true_positive = 0
    total_false_positive = 0
    total_false_negative = 0
    gold_none_mask = unit_predictions["gold_aspects"] == "none"

    for aspect in ASPECTS:
        gold_has = unit_predictions["gold_aspects"].str.split(";").apply(lambda values: aspect in values)
        pred_has = unit_predictions["predicted_aspects"].str.split(";").apply(lambda values: aspect in values)
        true_positive = int((gold_has & pred_has).sum())
        false_positive = int((~gold_has & pred_has).sum())
        false_negative = int((gold_has & ~pred_has).sum())
        true_negative = int((~gold_has & ~pred_has).sum())
        precision = safe_divide(true_positive, true_positive + false_positive)
        recall = safe_divide(true_positive, true_positive + false_negative)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        rows.append(
            {
                "aspect": aspect,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "support": int(gold_has.sum()),
                "predicted_count": int(pred_has.sum()),
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "true_negative": true_negative,
                "false_detections_on_none": int((gold_none_mask & pred_has).sum()),
            }
        )
        total_true_positive += true_positive
        total_false_positive += false_positive
        total_false_negative += false_negative

    exact_accuracy = unit_predictions["aspect_exact_match"].mean()
    macro_f1 = sum(row["f1"] for row in rows) / len(rows)
    micro_precision = safe_divide(total_true_positive, total_true_positive + total_false_positive)
    micro_recall = safe_divide(total_true_positive, total_true_positive + total_false_negative)
    micro_f1 = safe_divide(2 * micro_precision * micro_recall, micro_precision + micro_recall)
    none_false_detection_units = int(
        (gold_none_mask & (unit_predictions["predicted_aspects"] != "none")).sum()
    )
    summary = {
        "unit_count": len(unit_predictions),
        "gold_none_unit_count": int(gold_none_mask.sum()),
        "none_false_detection_unit_count": none_false_detection_units,
        "none_specificity": round(
            safe_divide(int(gold_none_mask.sum()) - none_false_detection_units, int(gold_none_mask.sum())),
            6,
        ),
        "exact_set_accuracy": round(float(exact_accuracy), 6),
        "macro_f1": round(macro_f1, 6),
        "micro_precision": round(micro_precision, 6),
        "micro_recall": round(micro_recall, 6),
        "micro_f1": round(micro_f1, 6),
    }
    return pd.DataFrame(rows), summary


def build_error_examples(
    unit_predictions: pd.DataFrame,
    pair_predictions: pd.DataFrame,
) -> pd.DataFrame:
    errors = []
    unit_by_id = unit_predictions.set_index("unit_id")

    def append_error(unit_id: str, error_type: str, focus_aspect: str, detail: str) -> None:
        unit = unit_by_id.loc[unit_id]
        errors.append(
            {
                "error_id": f"error_{len(errors) + 1:04d}",
                "error_type": error_type,
                "unit_id": unit_id,
                "review_id": unit["review_id"],
                "book_id": unit["book_id"],
                "focus_aspect": focus_aspect,
                "gold_aspects": unit["gold_aspects"],
                "predicted_aspects": unit["predicted_aspects"],
                "gold_sentiment": "",
                "aspect_text_vader": unit["aspect_text_vader_label"],
                "sentence_vader": unit["sentence_vader_label"],
                "whole_review_vader": unit["whole_review_vader_label"],
                "unit_text": unit["unit_text"],
                "sentence_text": unit["sentence_text"],
                "detail": detail,
            }
        )

    for _, unit in unit_predictions.iterrows():
        gold_aspects = set() if unit["gold_aspects"] == "none" else set(unit["gold_aspects"].split(";"))
        predicted_aspects = (
            set() if unit["predicted_aspects"] == "none" else set(unit["predicted_aspects"].split(";"))
        )
        for aspect in sorted(gold_aspects - predicted_aspects):
            append_error(
                unit["unit_id"],
                "missed_aspect",
                aspect,
                f"Reference aspect '{aspect}' was not detected by the locked keyword baseline.",
            )
        for aspect in sorted(predicted_aspects - gold_aspects):
            error_type = "false_detection_on_none" if not gold_aspects else "incorrect_extra_aspect"
            append_error(
                unit["unit_id"],
                error_type,
                aspect,
                f"Baseline detected '{aspect}' outside the saved reference aspect set.",
            )

        sentence_aspects = set(find_aspects(str(unit["sentence_text"])))
        if str(unit["unit_text"]) != str(unit["sentence_text"]):
            for aspect in sorted((gold_aspects - predicted_aspects) & sentence_aspects):
                append_error(
                    unit["unit_id"],
                    "context_lost_during_splitting",
                    aspect,
                    "The clause missed the reference aspect, while the surrounding sentence contains a matching keyword signal.",
                )

    for _, pair in pair_predictions.iterrows():
        if pair["gold_sentiment"] not in SENTIMENTS:
            continue
        unit_id = pair["unit_id"]
        unit = unit_by_id.loc[unit_id]
        if pair["aspect_text_vader_label"] != pair["gold_sentiment"]:
            row = {
                "error_id": f"error_{len(errors) + 1:04d}",
                "error_type": "incorrect_sentiment_aspect_text",
                "unit_id": unit_id,
                "review_id": pair["review_id"],
                "book_id": pair["book_id"],
                "focus_aspect": pair["gold_aspect"],
                "gold_aspects": unit["gold_aspects"],
                "predicted_aspects": unit["predicted_aspects"],
                "gold_sentiment": pair["gold_sentiment"],
                "aspect_text_vader": pair["aspect_text_vader_label"],
                "sentence_vader": pair["sentence_vader_label"],
                "whole_review_vader": pair["whole_review_vader_label"],
                "unit_text": unit["unit_text"],
                "sentence_text": unit["sentence_text"],
                "detail": "Three-class VADER on the isolated annotation unit disagreed with the saved reference sentiment.",
            }
            errors.append(row)
        if pair["whole_review_vader_label"] != pair["gold_sentiment"]:
            row = {
                "error_id": f"error_{len(errors) + 1:04d}",
                "error_type": "incorrect_sentiment_whole_review",
                "unit_id": unit_id,
                "review_id": pair["review_id"],
                "book_id": pair["book_id"],
                "focus_aspect": pair["gold_aspect"],
                "gold_aspects": unit["gold_aspects"],
                "predicted_aspects": unit["predicted_aspects"],
                "gold_sentiment": pair["gold_sentiment"],
                "aspect_text_vader": pair["aspect_text_vader_label"],
                "sentence_vader": pair["sentence_vader_label"],
                "whole_review_vader": pair["whole_review_vader_label"],
                "unit_text": unit["unit_text"],
                "sentence_text": unit["sentence_text"],
                "detail": "Three-class VADER on the whole review disagreed with the same reference sentiment.",
            }
            errors.append(row)
        if (
            str(unit["unit_text"]) != str(unit["sentence_text"])
            and pair["aspect_text_vader_label"] != pair["gold_sentiment"]
            and pair["sentence_vader_label"] == pair["gold_sentiment"]
        ):
            row = {
                "error_id": f"error_{len(errors) + 1:04d}",
                "error_type": "context_lost_during_splitting",
                "unit_id": unit_id,
                "review_id": pair["review_id"],
                "book_id": pair["book_id"],
                "focus_aspect": pair["gold_aspect"],
                "gold_aspects": unit["gold_aspects"],
                "predicted_aspects": unit["predicted_aspects"],
                "gold_sentiment": pair["gold_sentiment"],
                "aspect_text_vader": pair["aspect_text_vader_label"],
                "sentence_vader": pair["sentence_vader_label"],
                "whole_review_vader": pair["whole_review_vader_label"],
                "unit_text": unit["unit_text"],
                "sentence_text": unit["sentence_text"],
                "detail": "The isolated clause was wrong but VADER on the original sentence matched the reference sentiment.",
            }
            errors.append(row)

    return pd.DataFrame(errors)


def evaluate(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(args.master, keep_default_na=False)
    splits = pd.read_csv(args.splits, keep_default_na=False)
    pipeline = pd.read_csv(args.pipeline, keep_default_na=False)
    blind = pd.read_csv(args.blind, keep_default_na=False)
    references = pd.read_csv(args.references, keep_default_na=False)

    validation, validation_meta = validate_step2(master, splits, pipeline, blind, references)
    validation.to_csv(output_dir / "step3_validation_summary.csv", index=False)

    unit_predictions, pair_predictions = build_predictions(master, pipeline, references)
    unit_predictions.to_csv(output_dir / "step3_unit_predictions.csv", index=False)

    aspect_metrics, aspect_summary = aspect_detection_metrics(unit_predictions)
    aspect_metrics.to_csv(output_dir / "step3_aspect_detection_metrics.csv", index=False)

    evaluable_pairs = pair_predictions[
        pair_predictions["gold_aspect"].isin(ASPECTS)
        & pair_predictions["gold_sentiment"].isin(SENTIMENTS)
    ].copy()
    y_true = evaluable_pairs["gold_sentiment"].tolist()
    aspect_y_pred = evaluable_pairs["aspect_text_vader_label"].tolist()
    whole_y_pred = evaluable_pairs["whole_review_vader_label"].tolist()

    aspect_class_metrics, aspect_sentiment_summary = classification_metrics(
        y_true, aspect_y_pred, SENTIMENTS, "aspect_text_vader_3class"
    )
    whole_class_metrics, whole_sentiment_summary = classification_metrics(
        y_true, whole_y_pred, SENTIMENTS, "whole_review_vader_3class"
    )
    pd.concat([aspect_class_metrics, whole_class_metrics], ignore_index=True).to_csv(
        output_dir / "step3_vader_per_class_metrics.csv", index=False
    )

    pd.concat(
        [
            confusion_matrix(y_true, aspect_y_pred, SENTIMENTS, "aspect_text_vader_3class"),
            confusion_matrix(y_true, whole_y_pred, SENTIMENTS, "whole_review_vader_3class"),
        ],
        ignore_index=True,
    ).to_csv(output_dir / "step3_vader_confusion_matrices.csv", index=False)

    paired_outcomes = Counter()
    for gold, aspect_pred, whole_pred in zip(y_true, aspect_y_pred, whole_y_pred):
        aspect_correct = aspect_pred == gold
        whole_correct = whole_pred == gold
        if aspect_correct and not whole_correct:
            paired_outcomes["aspect_text_only_correct"] += 1
        elif whole_correct and not aspect_correct:
            paired_outcomes["whole_review_only_correct"] += 1
        elif aspect_correct and whole_correct:
            paired_outcomes["both_correct"] += 1
        else:
            paired_outcomes["both_incorrect"] += 1

    comparison = pd.DataFrame(
        [
            {
                **aspect_sentiment_summary,
                **paired_outcomes,
                "macro_f1_difference_vs_whole_review": round(
                    aspect_sentiment_summary["macro_f1"] - whole_sentiment_summary["macro_f1"], 6
                ),
                "accuracy_difference_vs_whole_review": round(
                    aspect_sentiment_summary["accuracy"] - whole_sentiment_summary["accuracy"], 6
                ),
            },
            {
                **whole_sentiment_summary,
                **paired_outcomes,
                "macro_f1_difference_vs_whole_review": 0.0,
                "accuracy_difference_vs_whole_review": 0.0,
            },
        ]
    )
    comparison.to_csv(output_dir / "step3_vader_scope_comparison.csv", index=False)

    none_excluded = int((references["aspect"] == "none").sum())
    unresolved_excluded = len(pair_predictions) - len(evaluable_pairs)
    ambiguous_evaluable = int(evaluable_pairs["ambiguity_flag"].sum())
    evaluable_unit_count = int(evaluable_pairs["unit_id"].nunique())
    additional_multi_aspect_pairs = len(evaluable_pairs) - evaluable_unit_count
    summary_rows = [
        {"section": "provenance", "metric": "annotation_source", "value": validation_meta["annotation_source"]},
        {"section": "provenance", "metric": "human_verified", "value": validation_meta["human_verified"]},
        {"section": "validation", "metric": "book_split_overlap_count", "value": 0},
        {
            "section": "validation",
            "metric": "test_book_overlap_with_weak_train_candidates",
            "value": 0,
        },
    ]
    summary_rows.extend(
        {"section": "aspect_detection", "metric": key, "value": value}
        for key, value in aspect_summary.items()
    )
    summary_rows.extend(
        [
            {"section": "sentiment_exclusions", "metric": "none_units_excluded", "value": none_excluded},
            {
                "section": "sentiment_exclusions",
                "metric": "unresolved_pairs_excluded",
                "value": unresolved_excluded,
            },
            {
                "section": "sentiment_exclusions",
                "metric": "ambiguous_pairs_included_with_resolved_labels",
                "value": ambiguous_evaluable,
            },
            {
                "section": "sentiment_3class",
                "metric": "evaluable_unit_count",
                "value": evaluable_unit_count,
            },
            {
                "section": "sentiment_3class",
                "metric": "evaluable_aspect_pair_count",
                "value": aspect_sentiment_summary["example_count"],
            },
            {
                "section": "sentiment_3class",
                "metric": "additional_multi_aspect_pairs",
                "value": additional_multi_aspect_pairs,
            },
            {
                "section": "sentiment_3class",
                "metric": "aspect_text_macro_f1",
                "value": aspect_sentiment_summary["macro_f1"],
            },
            {
                "section": "sentiment_3class",
                "metric": "aspect_text_accuracy",
                "value": aspect_sentiment_summary["accuracy"],
            },
            {
                "section": "sentiment_3class",
                "metric": "whole_review_macro_f1",
                "value": whole_sentiment_summary["macro_f1"],
            },
            {
                "section": "sentiment_3class",
                "metric": "whole_review_accuracy",
                "value": whole_sentiment_summary["accuracy"],
            },
        ]
    )
    results_summary = pd.DataFrame(summary_rows)
    results_summary.to_csv(output_dir / "step3_results_summary.csv", index=False)

    error_examples = build_error_examples(unit_predictions, evaluable_pairs)
    error_examples.to_csv(output_dir / "step3_error_examples.csv", index=False)
    error_summary = (
        error_examples.groupby("error_type", as_index=False)
        .size()
        .rename(columns={"size": "error_count"})
        .sort_values("error_type")
    )
    error_summary.to_csv(output_dir / "step3_error_summary.csv", index=False)

    print("Step 3 baseline evaluation complete.")
    print(f"Reference provenance: {validation_meta['annotation_source']} (human_verified=False)")
    print(f"Aspect detection macro-F1: {aspect_summary['macro_f1']:.4f}")
    print(f"Aspect-text VADER 3-class macro-F1: {aspect_sentiment_summary['macro_f1']:.4f}")
    print(f"Whole-review VADER 3-class macro-F1: {whole_sentiment_summary['macro_f1']:.4f}")
    print(f"Sentiment examples scored: {len(evaluable_pairs)} pairs from {(references['aspect'] != 'none').sum()} units")
    print(f"Excluded from sentiment: {none_excluded} none units, {unresolved_excluded} unresolved pairs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the locked keyword + VADER baseline.")
    parser.add_argument("--master", default="data/books_reviews_20000.csv")
    parser.add_argument("--splits", default="data/book_splits.csv")
    parser.add_argument("--pipeline", default="outputs/full_pipeline_output.csv")
    parser.add_argument("--blind", default="data/test_annotation_units_300_blind.csv")
    parser.add_argument("--references", default="data/reference_annotations_300.csv")
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
