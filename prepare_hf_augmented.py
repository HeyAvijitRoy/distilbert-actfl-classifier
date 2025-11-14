# prepare_hf_augmented.py
#
# Augment  ASAP dataset with high-proficiency samples taken from
# the CommonLit "Ease-of-Readability" dataset on Hugging Face Hub.
#
# This script expects:
#  - asap_actfl_labeled.csv to exist (created by script.py)
#  - network access to HF datasets to download CommonLit

import pandas as pd
import numpy as np
from datasets import load_dataset

# This script augments ASAP dataset with high-proficiency samples
# from the CommonLit Ease-of-Readability dataset.

def main():
    # Load existing ASAP labeled data
    df_asap = pd.read_csv("asap_actfl_labeled.csv")
    print(f"ASAP base size: {len(df_asap)}")

    # Load CommonLit Ease-of-Readability train split
    cl_ds = load_dataset("casey-martin/CommonLit-Ease-of-Readability", split="train")
    df_cl = cl_ds.to_pandas()
    print(f"CommonLit raw size: {len(df_cl)}")

    # Select the text and readability metric columns explicitly
    # 'Excerpt' contains the passage text, 'Flesch-Reading-Ease' is a good difficulty proxy
    text_col = "Excerpt"
    score_col = "Flesch-Reading-Ease"
    if text_col not in df_cl.columns or score_col not in df_cl.columns:
        raise KeyError(f"Expected columns '{text_col}' and '{score_col}' not found in CommonLit dataset.")

    df_cl = df_cl[[text_col, score_col]].rename(columns={
        text_col: "essay",
        score_col: "difficulty"
    })

    # Compute top deciles of difficulty (lower scores = more difficult)
    # Flesch score: lower values indicate harder readability
    p10 = df_cl["difficulty"].quantile(0.10)
    p20 = df_cl["difficulty"].quantile(0.20)
    print(f"Difficulty thresholds → 10th pct: {p10:.2f}, 20th pct: {p20:.2f}")

    # Extract Superior & Advanced High subsets
    df_sup = df_cl[df_cl["difficulty"] <= p10].copy()
    df_sup["actfl_level"] = "Superior"

    df_ah = df_cl[(df_cl["difficulty"] > p10) & (df_cl["difficulty"] <= p20)].copy()
    df_ah["actfl_level"] = "Advanced High"

    print(f"Derived Superior samples: {len(df_sup)}")
    print(f"Derived Advanced High samples: {len(df_ah)}")

    # Concatenate and save
    df_aug = pd.concat([df_asap, df_sup, df_ah], ignore_index=True)
    out_csv = "asap_plus_hf.csv"
    df_aug.to_csv(out_csv, index=False)
    print(f"Total augmented size: {len(df_aug)} → saved to {out_csv}")

if __name__ == "__main__":
    main()
