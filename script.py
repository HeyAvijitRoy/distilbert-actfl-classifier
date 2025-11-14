"""
script.py

Load the original ASAP files, compute ACTFL-level labels by quantiles,
and save a labeled CSV (asap_actfl_labeled.csv).

Usage:
    python script.py
"""

import os
import pandas as pd

# Choose which training file to read from the asap-aes folder
ASAP_FOLDER = "asap-aes"
TRAIN_TSV = os.path.join(ASAP_FOLDER, "training_set_rel3.tsv")


def load_asap(tsv_path):
    # Try several encodings if necessary
    for enc in ("utf-8", "latin1", "cp1252"):
        try:
            df = pd.read_csv(tsv_path, sep="\t", encoding=enc)
            return df
        except Exception:
            continue
    raise RuntimeError(f"Unable to read {tsv_path} with common encodings")


def map_to_actfl(score_series, labels):
    """
    Convert raw numeric scores into 10 buckets (ACTFL levels)
    using pandas.qcut to get (approx) equal-sized buckets.
    labels: list of 10 label names in the order from lowest -> highest.
    """
    # Ensure there are at least 10 unique values
    unique_vals = score_series.dropna().unique()
    if len(unique_vals) < 10:
        # fallback to pd.cut
        return pd.cut(score_series, bins=10, labels=labels)
    return pd.qcut(score_series, q=10, labels=labels, duplicates="drop")


def main():
    print(f"Reading ASAP training from: {TRAIN_TSV}")
    df = load_asap(TRAIN_TSV)
    print(f"Loaded {len(df)} rows. Columns: {list(df.columns)}")

    # ASAP dataset uses 'domain1_score' (or 'domain2_score' for some sets);
    # Prefer domain1_score if present.
    score_col = None
    for c in ["domain1_score", "domain2_score", "domain_score", "score"]:
        if c in df.columns:
            score_col = c
            break
    if score_col is None:
        raise RuntimeError("No score column found in ASAP training TSV.")

    # Keep only a minimal set of columns
    keep_cols = ["essay_id", "essay_set", "essay", score_col]
    df = df[[c for c in keep_cols if c in df.columns]].rename(columns={score_col: "domain1_score"})

    # ACTFL labels low→high (10 categories)
    actfl_labels = [
        "Novice Low", "Novice Mid", "Novice High",
        "Intermediate Low", "Intermediate Mid", "Intermediate High",
        "Advanced Low", "Advanced Mid", "Advanced High",
        "Superior"
    ]

    # Create actfl_level column using qcut
    df["actfl_level"] = map_to_actfl(df["domain1_score"], actfl_labels)

    # For rows where qcut dropped duplicates (rare), fallback to cut
    if df["actfl_level"].isnull().any():
        df["actfl_level"] = df["actfl_level"].cat.add_categories(actfl_labels).fillna(
            pd.cut(df["domain1_score"], bins=10, labels=actfl_labels)
        )

    out = "asap_actfl_labeled.csv"
    df.to_csv(out, index=False)
    print(f"Saved labeled training data to {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
