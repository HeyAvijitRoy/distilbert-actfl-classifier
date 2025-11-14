"""
train.py

Train a DistilBERT classifier on asap_plus_hf.csv (if present) or asap_actfl_labeled.csv.

Saves model to: distilbert-actfl-english/

Usage:
    python train.py
"""

import os
import numpy as np
import pandas as pd
from datasets import Dataset, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score

MODEL_NAME = "distilbert-base-uncased"
OUT_DIR = "distilbert-actfl-english"
CSV_PREFERRED = "asap_plus_hf.csv"
CSV_FALLBACK = "asap_actfl_labeled.csv"


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1}


def load_csv_dataset(csv_path):
    df = pd.read_csv(csv_path)
    # Ensure columns essay and actfl_level exist
    if "essay" not in df.columns or "actfl_level" not in df.columns:
        raise RuntimeError(f"{csv_path} must contain 'essay' and 'actfl_level' columns")
    return Dataset.from_pandas(df.reset_index(drop=True))


def main():
    csv_file = CSV_PREFERRED if os.path.exists(CSV_PREFERRED) else CSV_FALLBACK
    if not os.path.exists(csv_file):
        raise RuntimeError(f"Neither {CSV_PREFERRED} nor {CSV_FALLBACK} found in project root.")
    print(f"Using CSV: {csv_file}")

    ds = load_csv_dataset(csv_file)
    print(f"Loaded dataset: {len(ds)} rows")

    # Label class mapping (must match how labels were created)
    actfl_labels = [
        "Novice Low", "Novice Mid", "Novice High",
        "Intermediate Low", "Intermediate Mid", "Intermediate High",
        "Advanced Low", "Advanced Mid", "Advanced High",
        "Superior"
    ]
    cls = ClassLabel(names=actfl_labels)

    # Cast column and rename for Trainer compatibility
    ds = ds.class_encode_column("actfl_level")  # will create 'actfl_level'->int
    ds = ds.rename_column("actfl_level", "labels")

    # train/validation split
    split = ds.train_test_split(test_size=0.2, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_fn(batch):
        return tokenizer(
            batch["essay"],
            padding="max_length",
            truncation=True,
            max_length=256
        )

    train_ds = train_ds.map(tokenize_fn, batched=True)
    eval_ds = eval_ds.map(tokenize_fn, batched=True)

    cols = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format(type="torch", columns=cols)
    eval_ds.set_format(type="torch", columns=cols)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=cls.num_classes
    )

    args = TrainingArguments(
        output_dir=OUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="logs",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(OUT_DIR)
    print(f"Model saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
