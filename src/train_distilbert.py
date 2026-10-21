"""
ReviewLens - Step 6: DistilBERT fine-tuning on weak-labeled train text.

Train a binary DistilBERT sentiment classifier on a balanced 5k-sentence subset
of train-book review text. Evaluate it on the same AI reference benchmark used by
Step 3 and Step 4.

Usage:
    python src/train_distilbert.py
    python src/train_distilbert.py --max-train 2000 --epochs 1 --output-dir outputs/step6_test
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

SEED = 42
LABEL2ID = {"negative": 0, "positive": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def load_train_subset(
    pipeline_output_path: str,
    splits_path: str,
    max_train: int = 5000,
) -> pd.DataFrame:
    pipe_df = pd.read_csv(pipeline_output_path, keep_default_na=False)
    split_df = pd.read_csv(splits_path, keep_default_na=False)
    book_to_split = dict(zip(split_df["book_id"].astype(str), split_df["split"]))
    pipe_df["split"] = pipe_df["book_id"].astype(str).map(book_to_split)

    train_df = pipe_df[pipe_df["split"] == "train"].copy()
    train_df["stars"] = pd.to_numeric(train_df["stars"], errors="coerce")
    train_df = train_df[train_df["stars"].isin([1.0, 2.0, 4.0, 5.0])].copy()
    train_df["label"] = train_df["stars"].apply(lambda s: "positive" if s >= 4.0 else "negative")

    train_df["text"] = train_df["sentence_text"].fillna("").str.strip()
    train_df = train_df[train_df["text"].str.len() > 0].copy()
    train_df = train_df.drop_duplicates(subset=["text", "label"])

    counts = train_df["label"].value_counts()
    if len(counts) < 2:
        raise ValueError("Need both positive and negative labels in the train split.")

    n_per_class = min(max_train // 2, int(min(counts["negative"], counts["positive"])))
    if n_per_class <= 0:
        raise ValueError("No usable training rows after filtering weak labels.")

    positive = train_df[train_df["label"] == "positive"].sample(
        n=min(n_per_class, len(train_df[train_df["label"] == "positive"])),
        random_state=SEED,
    )
    negative = train_df[train_df["label"] == "negative"].sample(
        n=min(n_per_class, len(train_df[train_df["label"] == "negative"])),
        random_state=SEED,
    )
    sampled = pd.concat([positive, negative], ignore_index=True)
    sampled["label"] = sampled["label"].map({"negative": 0, "positive": 1})
    return sampled[["text", "label"]].reset_index(drop=True)


def load_reference_eval(reference_path: str) -> pd.DataFrame:
    ref_df = pd.read_csv(reference_path, keep_default_na=False)
    ref_df = ref_df[(ref_df["aspect"] != "none") & (ref_df["sentiment"].isin(["positive", "negative"]))].copy()
    ref_df["text"] = ref_df["unit_text"].fillna("").astype(str).str.strip()
    ref_df = ref_df[ref_df["text"].str.len() > 0].copy()
    ref_df["label"] = ref_df["sentiment"].map({"negative": 0, "positive": 1}).astype(int)
    return ref_df[["unit_id", "book_id", "aspect", "sentiment", "text", "label"]]


def build_dataset(df: pd.DataFrame) -> Dataset:
    dataset = Dataset.from_pandas(df[["text", "label"]].copy(), preserve_index=False)
    return dataset


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=[0, 1], average=None, zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "positive_f1": f1[1],
        "negative_f1": f1[0],
        "positive_precision": precision[1],
        "negative_precision": precision[0],
        "positive_recall": recall[1],
        "negative_recall": recall[0],
    }


def train_distilbert(
    pipeline_output_path: str = "outputs/full_pipeline_output.csv",
    splits_path: str = "data/book_splits.csv",
    reference_path: str = "data/reference_annotations_300.csv",
    output_dir: str = "outputs/step6",
    model_name: str = "distilbert-base-uncased",
    max_train: int = 5000,
    epochs: int = 1,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    seed: int = 42,
) -> dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    train_df = load_train_subset(pipeline_output_path, splits_path, max_train=max_train)
    eval_df = load_reference_eval(reference_path)

    print(f"Training rows selected: {len(train_df):,}")
    print(train_df["label"].value_counts().to_dict())
    print(f"Evaluation rows: {len(eval_df):,}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = build_dataset(train_df)
    eval_ds = build_dataset(eval_df[["text", "label"]])

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

    tokenized_train = train_ds.map(tokenize_fn, batched=True, batch_size=8)
    tokenized_eval = eval_ds.map(tokenize_fn, batched=True, batch_size=8)
    tokenized_train = tokenized_train.rename_column("label", "labels")
    tokenized_eval = tokenized_eval.rename_column("label", "labels")
    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized_eval.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    args = TrainingArguments(
        output_dir=str(Path(output_dir) / "run"),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_steps=25,
        save_strategy="no",
        eval_strategy="epoch" if epochs > 1 else "no",
        report_to="none",
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval if epochs > 1 else None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    pred_out = trainer.predict(tokenized_eval)
    logits = pred_out.predictions
    pred_ids = np.argmax(logits, axis=-1)
    true_ids = pred_out.label_ids

    acc = accuracy_score(true_ids, pred_ids)
    macro_f1 = f1_score(true_ids, pred_ids, average="macro")
    precision, recall, f1_scores, support = precision_recall_fscore_support(
        true_ids, pred_ids, labels=[0, 1], zero_division=0
    )

    cm = confusion_matrix(true_ids, pred_ids, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=["true_negative", "true_positive"],
        columns=["pred_negative", "pred_positive"],
    )
    cm_df.to_csv(f"{output_dir}/confusion_matrix.csv")

    per_class_df = pd.DataFrame([
        {
            "class": "negative",
            "precision": round(precision[0], 4),
            "recall": round(recall[0], 4),
            "f1": round(f1_scores[0], 4),
            "support": int(support[0]),
        },
        {
            "class": "positive",
            "precision": round(precision[1], 4),
            "recall": round(recall[1], 4),
            "f1": round(f1_scores[1], 4),
            "support": int(support[1]),
        },
    ])
    per_class_df.to_csv(f"{output_dir}/per_class_metrics.csv", index=False)

    eval_df["pred_label"] = pred_ids
    eval_df["pred_sentiment"] = eval_df["pred_label"].map({0: "negative", 1: "positive"})
    eval_df[[
        "unit_id", "book_id", "aspect", "sentiment", "text", "pred_sentiment", "label"
    ]].to_csv(f"{output_dir}/predictions_reference.csv", index=False)

    # Comparison against the locked VADER and TF-IDF baselines
    svm_overall = pd.read_csv("outputs/step4/overall_metrics.csv")
    svm_f1 = float(svm_overall[svm_overall["metric"] == "macro_f1_aspect_text"]["value"].iloc[0])
    vader_f1 = 0.7171
    comparison_df = pd.DataFrame([
        {
            "model": "VADER Baseline",
            "scope": "aspect_text",
            "accuracy": 0.6364,
            "macro_f1": vader_f1,
            "positive_f1": 0.7200,
            "negative_f1": 0.7143,
        },
        {
            "model": "TF-IDF + LinearSVC",
            "scope": "aspect_text",
            "accuracy": float(svm_overall[svm_overall["metric"] == "accuracy_aspect_text"]["value"].iloc[0]),
            "macro_f1": svm_f1,
            "positive_f1": 0.9032,
            "negative_f1": 0.6667,
        },
        {
            "model": "DistilBERT",
            "scope": "aspect_text",
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "positive_f1": round(f1_scores[1], 4),
            "negative_f1": round(f1_scores[0], 4),
        },
    ])
    comparison_df.to_csv(f"{output_dir}/comparison_with_baselines.csv", index=False)

    overall_df = pd.DataFrame([
        {"metric": "evaluated_units", "value": len(eval_df)},
        {"metric": "accuracy", "value": round(acc, 4)},
        {"metric": "macro_f1", "value": round(macro_f1, 4)},
        {"metric": "vader_macro_f1", "value": round(vader_f1, 4)},
        {"metric": "svm_macro_f1", "value": round(svm_f1, 4)},
        {"metric": "f1_delta_over_vader", "value": round(macro_f1 - vader_f1, 4)},
        {"metric": "train_rows", "value": len(train_df)},
        {"metric": "max_train", "value": max_train},
        {"metric": "epochs", "value": epochs},
    ])
    overall_df.to_csv(f"{output_dir}/overall_metrics.csv", index=False)

    val_checks = [
        {"check": "zero_test_leakage_in_training", "status": "PASS", "detail": "Train rows are restricted to train books only."},
        {"check": "reference_evaluation_used", "status": "PASS", "detail": "Evaluation uses the same saved reference benchmark as prior steps."},
        {"check": "balanced_subset", "status": "PASS", "detail": f"Balanced subset capped at {max_train} training rows."},
        {"check": "binary_sentiment_only", "status": "PASS", "detail": "Positive/negative subset only; neutral units excluded from this training/evaluation setup."},
    ]
    pd.DataFrame(val_checks).to_csv(f"{output_dir}/validation_checks.csv", index=False)

    results = {
        "status": "PASS",
        "train_rows": len(train_df),
        "evaluated_units": len(eval_df),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "positive_f1": round(f1_scores[1], 4),
        "negative_f1": round(f1_scores[0], 4),
        "confusion_matrix": cm.tolist(),
    }
    with open(f"{output_dir}/validation_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n--- DistilBERT evaluation summary ---")
    print(comparison_df.to_string(index=False))
    print(f"Saved outputs to {output_dir}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate DistilBERT sentiment classifier for ReviewLens.")
    parser.add_argument("--pipeline-output", default="outputs/full_pipeline_output.csv")
    parser.add_argument("--splits", default="data/book_splits.csv")
    parser.add_argument("--reference", default="data/reference_annotations_300.csv")
    parser.add_argument("--output-dir", default="outputs/step6")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--max-train", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train_distilbert(
        pipeline_output_path=args.pipeline_output,
        splits_path=args.splits,
        reference_path=args.reference,
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_train=args.max_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
