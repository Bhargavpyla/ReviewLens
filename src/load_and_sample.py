"""Randomly sample Amazon Reviews 2023 Books without downloading the full file."""

import argparse
import csv
import json
import math
import random
import re
import statistics
from collections import Counter
from pathlib import Path
from urllib.request import Request, urlopen


DEFAULT_URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/"
    "resolve/main/raw/review_categories/Books.jsonl"
)
WHITESPACE = re.compile(r"\s+")


def get_file_size(url: str) -> int:
    request = Request(url, method="HEAD")
    with urlopen(request) as response:
        linked_size = response.headers.get("X-Linked-Size")
        content_length = response.headers.get("Content-Length")
    return int(linked_size or content_length)


def fetch_complete_lines(url: str, start: int, chunk_bytes: int) -> list[bytes]:
    end = start + chunk_bytes - 1
    request = Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urlopen(request) as response:
        if response.status != 206 or not response.headers.get("Content-Range"):
            raise RuntimeError("Dataset server did not honor the byte-range request")
        payload = response.read()

    lines = payload.splitlines()
    if start > 0 and lines:
        lines = lines[1:]
    if payload and not payload.endswith((b"\n", b"\r")) and lines:
        lines = lines[:-1]
    return lines


def clean_text(text: str) -> str:
    return WHITESPACE.sub(" ", text).strip()


def sample_reviews(
    url: str,
    sample_size: int,
    seed: int,
    chunk_count: int,
    chunk_bytes: int,
) -> tuple[list[dict], dict]:
    file_size = get_file_size(url)
    random_generator = random.Random(seed)
    max_start = max(0, file_size - chunk_bytes)
    starts = random_generator.sample(range(max_start + 1), chunk_count)

    candidates = []
    empty_count = 0
    invalid_count = 0
    for start in starts:
        for line in fetch_complete_lines(url, start, chunk_bytes):
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                invalid_count += 1
                continue

            raw_text = str(record.get("text") or "")
            cleaned_text = clean_text(raw_text)
            if not cleaned_text:
                empty_count += 1
                continue

            candidates.append(
                {
                    "book_id": record.get("parent_asin") or record.get("asin") or "",
                    "rating": record.get("rating"),
                    "title": clean_text(str(record.get("title") or "")),
                    "raw_text": raw_text,
                    "cleaned_text": cleaned_text,
                }
            )

    unique_candidates = {}
    for record in candidates:
        duplicate_key = (record["book_id"], record["cleaned_text"].casefold())
        unique_candidates.setdefault(duplicate_key, record)

    unique_records = list(unique_candidates.values())
    if len(unique_records) < sample_size:
        raise RuntimeError(
            f"Only {len(unique_records)} usable unique reviews found; "
            "increase --chunks or --chunk-bytes"
        )

    sampled = random_generator.sample(unique_records, sample_size)
    for review_number, record in enumerate(sampled, start=1):
        record["review_id"] = f"review_{review_number:05d}"

    stats = {
        "file_size": file_size,
        "chunks": chunk_count,
        "bytes_requested": chunk_count * chunk_bytes,
        "parsed_nonempty": len(candidates),
        "empty_dropped": empty_count,
        "invalid_dropped": invalid_count,
        "duplicates_dropped": len(candidates) - len(unique_records),
    }
    return sampled, stats


def write_csv(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "review_id",
        "book_id",
        "rating",
        "title",
        "raw_text",
        "cleaned_text",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_stats(records: list[dict], stats: dict, output_path: Path, stats_dir: Path) -> None:
    rating_counts = Counter(record["rating"] for record in records)
    book_counts = Counter(record["book_id"] for record in records)
    stats_dir.mkdir(parents=True, exist_ok=True)
    stem = output_path.stem

    summary_path = stats_dir / f"{stem}_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(
            {"metric": metric, "value": value}
            for metric, value in {
                **stats,
                "final_sample": len(records),
                "unique_books": len(book_counts),
            }.items()
        )

    rating_path = stats_dir / f"{stem}_rating_distribution.csv"
    with rating_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["rating", "review_count"])
        writer.writeheader()
        writer.writerows(
            {"rating": rating, "review_count": count}
            for rating, count in sorted(rating_counts.items())
        )

    books_path = stats_dir / f"{stem}_reviews_per_book.csv"
    with books_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["book_id", "review_count"])
        writer.writeheader()
        writer.writerows(
            {"book_id": book_id, "review_count": count}
            for book_id, count in sorted(book_counts.items())
        )


def print_report(records: list[dict], stats: dict, output_path: Path) -> None:
    rating_counts = Counter(record["rating"] for record in records)
    book_counts = Counter(record["book_id"] for record in records)
    print(f"Official file size: {stats['file_size']:,} bytes")
    print(
        f"Random chunks read: {stats['chunks']} "
        f"({stats['bytes_requested']:,} bytes requested)"
    )
    print(f"Parsed non-empty reviews before deduplication: {stats['parsed_nonempty']}")
    print(f"Dropped empty reviews: {stats['empty_dropped']}")
    print(f"Dropped invalid records: {stats['invalid_dropped']}")
    print(f"Dropped duplicate reviews: {stats['duplicates_dropped']}")
    print(f"Final random sample: {len(records)}")
    print(f"Rating distribution: {dict(sorted(rating_counts.items()))}")
    print(f"Unique books: {len(book_counts)}")
    print(f"Maximum reviews for one book: {max(book_counts.values())}")
    print(f"Median reviews per book: {statistics.median(book_counts.values()):g}")
    print(f"Saved: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--chunks", type=int)
    parser.add_argument("--chunk-bytes", type=int)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, default=Path("data/books_sample_20.csv"))
    parser.add_argument("--preview-output", type=Path)
    parser.add_argument("--preview-size", type=int, default=20)
    parser.add_argument("--stats-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunk_bytes = args.chunk_bytes or (65_536 if args.sample_size <= 100 else 1_048_576)
    chunk_count = args.chunks or max(8, math.ceil(args.sample_size / 750))
    records, stats = sample_reviews(
        url=args.url,
        sample_size=args.sample_size,
        seed=args.seed,
        chunk_count=chunk_count,
        chunk_bytes=chunk_bytes,
    )
    write_csv(records, args.output)
    write_stats(records, stats, args.output, args.stats_dir)
    if args.preview_output:
        write_csv(records[: args.preview_size], args.preview_output)
        print(f"Saved preview: {args.preview_output}")
    print_report(records, stats, args.output)


if __name__ == "__main__":
    main()
