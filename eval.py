"""
eval.py

Load the saved model (distilbert-actfl-english/) and run evaluation on the
validation split derived from asap_plus_hf.csv (or fallback).

Usage:
    python eval.py
"""

import os
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import torch

MODEL_DIR = "distilbert-actfl-english"
CSV_PREFERRED = "asap_plus_hf.csv"
CSV_FALLBACK = "asap_actfl_labeled.csv"

LABELS = [
    "Novice Low", "Novice Mid", "Novice High",
    "Intermediate Low", "Intermediate Mid", "Intermediate High",
    "Advanced Low", "Advanced Mid", "Advanced High",
    "Superior"
]


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"eval_accuracy": acc, "eval_f1_macro": f1}


def main():
    csv_file = CSV_PREFERRED if os.path.exists(CSV_PREFERRED) else CSV_FALLBACK
    if not os.path.exists(csv_file):
        raise RuntimeError("No input CSV found for evaluation.")

    df = pd.read_csv(csv_file)
    ds = Dataset.from_pandas(df.reset_index(drop=True))

    # If the CSV still contains string labels, map them to ints consistent with LABELS
    if "actfl_level" in df.columns and df["actfl_level"].dtype == object:
        label_map = {name: i for i, name in enumerate(LABELS)}
        ds = ds.map(lambda ex: {"labels": label_map.get(ex["actfl_level"], -1)})
    else:
        # assume already encoded, fallback to using 'labels' column
        if "labels" not in ds.column_names:
            raise RuntimeError("CSV must contain 'actfl_level' or 'labels' column")

    # Split for a simple eval: mirror train.py test split
    split = ds.train_test_split(test_size=0.2, seed=42)
    eval_ds = split["test"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    device = 0 if torch.cuda.is_available() else -1

    def tokenize_fn(batch):
        return tokenizer(batch["essay"], truncation=True, padding="max_length", max_length=256)

    eval_ds = eval_ds.map(tokenize_fn, batched=True)
    cols = ["input_ids", "attention_mask", "labels"]
    eval_ds.set_format(type="torch", columns=cols)

    trainer = Trainer(model=model, tokenizer=tokenizer)

    print("▶ Starting evaluation on validation split...")
    res = trainer.evaluate(eval_dataset=eval_ds, metric_key_prefix="eval")

    # Get predictions for confusion matrix
    preds_output = trainer.predict(eval_ds)
    preds = np.argmax(preds_output.predictions, axis=1)
    labels = preds_output.label_ids

    cm = confusion_matrix(labels, preds, labels=list(range(len(LABELS))))
    print("Confusion Matrix:\n", cm)
    print("\n✅ Evaluation Results:", res)


if __name__ == "__main__":
    main()
