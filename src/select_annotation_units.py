"""
ReviewLens - Select Test Annotation Units (Step 2).

Samples ~300 annotation units from test books independently of keyword matches,
predicted sentiment, and star ratings.
Enforces a documented per-book cap (max 2 units per book) to ensure wide diversity.
Retains original sentence context and outputs a strictly blind annotation file.

Usage:
    python src/select_annotation_units.py
    python src/select_annotation_units.py --input outputs/full_pipeline_output.csv --output data/test_annotation_units_300_blind.csv --seed 42 --sample-size 300 --max-per-book 2
"""

import argparse
import random
from pathlib import Path
import pandas as pd


def select_test_units(
    input_path: str = "outputs/full_pipeline_output.csv",
    output_path: str = "data/test_annotation_units_300_blind.csv",
    sample_size: int = 300,
    seed: int = 42,
    max_per_book: int = 2,
) -> pd.DataFrame:
    print(f"Loading pipeline records from: {input_path}")
    df = pd.read_csv(input_path, keep_default_na=False)

    # Filter to test split and deduplicate by clause_id so each unique unit appears once
    test_df = df[df["split"] == "test"].drop_duplicates(subset=["clause_id"]).copy()
    print(f"Candidate test units available: {len(test_df):,} unique clauses across {test_df['book_id'].nunique():,} test books.")

    # Sort deterministically by clause_id before shuffling
    test_df_sorted = test_df.sort_values(by=["clause_id"]).reset_index(drop=True)
    indices = list(test_df_sorted.index)
    rng = random.Random(seed)
    rng.shuffle(indices)

    selected_rows = []
    book_counts = {}

    for idx in indices:
        row = test_df_sorted.iloc[idx]
        book_id = str(row["book_id"])
        unit_text = str(row["clause_text"]).strip()

        # Skip degenerate empty clauses
        if len(unit_text) < 3 or unit_text == "[empty review]":
            continue

        if book_counts.get(book_id, 0) < max_per_book:
            book_counts[book_id] = book_counts.get(book_id, 0) + 1
            selected_rows.append(row)
            if len(selected_rows) == sample_size:
                break

    if len(selected_rows) < sample_size:
        raise RuntimeError(
            f"Could only sample {len(selected_rows)} units with max_per_book={max_per_book}. Increase cap or candidate pool."
        )

    # Build clean, blind evaluation dataframe (NO predictions, NO VADER, NO stars)
    blind_records = []
    for u_idx, r in enumerate(selected_rows, start=1):
        blind_records.append({
            "unit_id": f"unit_{u_idx:03d}",
            "sentence_id": r["sentence_id"],
            "clause_id": r["clause_id"],
            "review_id": r["review_id"],
            "book_id": r["book_id"],
            "sentence_text": r["sentence_text"],
            "unit_text": r["clause_text"],
        })

    df_blind = pd.DataFrame(blind_records)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_blind.to_csv(output_path, index=False)
    print(f"Saved {len(df_blind)} blind test annotation units to: {output_path}")

    # Validation checks
    assert len(df_blind) == sample_size
    assert df_blind["unit_id"].nunique() == sample_size
    assert df_blind["clause_id"].nunique() == sample_size
    assert df_blind["book_id"].value_counts().max() <= max_per_book

    print("\n--- Test Annotation Units Sampling Report ---")
    print(f"Total units selected: {len(df_blind)}")
    print(f"Unique books represented: {df_blind['book_id'].nunique()} (max cap: {max_per_book})")
    print(f"Fixed Seed: {seed}")
    print(f"Verified blind: No predictions, VADER scores, or star ratings included.")

    return df_blind


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select blind test annotation units.")
    parser.add_argument("--input", default="outputs/full_pipeline_output.csv", help="Full pipeline output CSV")
    parser.add_argument("--output", default="data/test_annotation_units_300_blind.csv", help="Blind output CSV")
    parser.add_argument("--sample-size", type=int, default=300, help="Number of annotation units")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--max-per-book", type=int, default=2, help="Maximum units per book")

    args = parser.parse_args()
    select_test_units(
        input_path=args.input,
        output_path=args.output,
        sample_size=args.sample_size,
        seed=args.seed,
        max_per_book=args.max_per_book,
    )
