"""
ReviewLens - Reproducible Book-Level Dataset Splitting (Step 2).

Partitions books into train (70%), dev (15%), and test (15%) by unique book_id count.
Enforces strict book-level isolation: no book crosses splits.
Guarantees that all 20 books inspected in the Step 1 preview are placed into development.

Usage:
    python src/split_data.py
    python src/split_data.py --seed 42 --input data/books_reviews_20000.csv --preview data/books_sample_20.csv
"""

import argparse
import random
from pathlib import Path
import pandas as pd


def create_book_splits(
    input_path: str = "data/books_reviews_20000.csv",
    preview_path: str = "data/books_sample_20.csv",
    output_path: str = "data/book_splits.csv",
    summary_path: str = "outputs/book_splits_summary.csv",
    seed: int = 42,
    train_ratio: float = 0.70,
    dev_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # 1. Load data safely (preserving literal strings like 'None')
    df_full = pd.read_csv(input_path, keep_default_na=False)
    df_preview = pd.read_csv(preview_path, keep_default_na=False)

    preview_books = set(df_preview["book_id"].dropna().astype(str).unique())
    all_books = sorted(df_full["book_id"].dropna().astype(str).unique())
    total_books = len(all_books)

    # 2. Verify all preview books are in the master dataset
    missing_preview = preview_books - set(all_books)
    if missing_preview:
        raise ValueError(f"Preview books not in master dataset: {missing_preview}")

    # 3. Calculate target book counts
    n_dev = round(total_books * dev_ratio)
    n_test = round(total_books * test_ratio)
    n_train = total_books - n_dev - n_test

    if len(preview_books) > n_dev:
        raise ValueError(
            f"Preview book count ({len(preview_books)}) exceeds target dev count ({n_dev})"
        )

    # 4. Partition remaining books deterministically
    remaining_books = [b for b in all_books if b not in preview_books]
    rng = random.Random(seed)
    rng.shuffle(remaining_books)

    n_dev_additional = n_dev - len(preview_books)
    dev_books = set(preview_books) | set(remaining_books[:n_dev_additional])
    test_books = set(remaining_books[n_dev_additional : n_dev_additional + n_test])
    train_books = set(remaining_books[n_dev_additional + n_test :])

    # 5. Strict verification assertions
    assert len(dev_books) == n_dev, f"Dev count mismatch: {len(dev_books)} != {n_dev}"
    assert len(test_books) == n_test, f"Test count mismatch: {len(test_books)} != {n_test}"
    assert len(train_books) == n_train, f"Train count mismatch: {len(train_books)} != {n_train}"
    assert preview_books.issubset(dev_books), "Not all preview books were placed into dev!"
    assert len(dev_books & test_books) == 0, "Book leakage: dev and test overlap!"
    assert len(dev_books & train_books) == 0, "Book leakage: dev and train overlap!"
    assert len(test_books & train_books) == 0, "Book leakage: test and train overlap!"
    assert len(dev_books | test_books | train_books) == total_books, "Books missing from splits!"

    # 6. Map splits to review counts
    book_to_split = {}
    for b in train_books:
        book_to_split[b] = "train"
    for b in dev_books:
        book_to_split[b] = "dev"
    for b in test_books:
        book_to_split[b] = "test"

    review_counts_per_book = df_full["book_id"].astype(str).value_counts().to_dict()

    split_rows = [
        {
            "book_id": b,
            "split": book_to_split[b],
            "review_count": review_counts_per_book.get(b, 0),
        }
        for b in all_books
    ]
    df_splits = pd.DataFrame(split_rows)

    # 7. Summary dataframe
    df_full["split"] = df_full["book_id"].astype(str).map(book_to_split)
    total_reviews = len(df_full)

    summary_rows = []
    for split_name in ["train", "dev", "test"]:
        b_cnt = (df_splits["split"] == split_name).sum()
        r_cnt = (df_full["split"] == split_name).sum()
        summary_rows.append({
            "split": split_name,
            "book_count": b_cnt,
            "book_percentage": round(b_cnt / total_books * 100, 3),
            "review_count": r_cnt,
            "review_percentage": round(r_cnt / total_reviews * 100, 3),
        })
    df_summary = pd.DataFrame(summary_rows)

    # 8. Save outputs
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_splits.to_csv(output_path, index=False)
    print(f"Saved book splits to: {output_path}")

    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(summary_path, index=False)
    print(f"Saved split summary to: {summary_path}")

    print("\n--- Book Splits Summary ---")
    print(df_summary.to_string(index=False))
    print(f"\nAll {len(preview_books)} Step 1 preview books confirmed in 'dev'.")
    print(f"Strict isolation verified: 0 books cross splits. Seed: {seed}")

    return df_splits, df_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create reproducible book-level dataset splits.")
    parser.add_argument("--input", default="data/books_reviews_20000.csv", help="Master reviews CSV")
    parser.add_argument("--preview", default="data/books_sample_20.csv", help="Preview reviews CSV")
    parser.add_argument("--output", default="data/book_splits.csv", help="Output book splits CSV")
    parser.add_argument("--summary", default="outputs/book_splits_summary.csv", help="Summary CSV")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train ratio by book count")
    parser.add_argument("--dev-ratio", type=float, default=0.15, help="Dev ratio by book count")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test ratio by book count")

    args = parser.parse_args()
    create_book_splits(
        input_path=args.input,
        preview_path=args.preview,
        output_path=args.output,
        summary_path=args.summary,
        seed=args.seed,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
    )
