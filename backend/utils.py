import os
import json
import pandas as pd
import re


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def preprocess_df(df):
    # Keep only text, label, optional topic
    cols = [c for c in ['text', 'label', 'topic'] if c in df.columns]
    df = df[cols].dropna(subset=['text', 'label']).copy()

    # Map labels to binary
    df['label'] = df['label'].astype(str).str.upper().map({'REAL': 1, 'FAKE': 0})
    df = df[df['label'].notna()].copy()
    df['label'] = df['label'].astype(int)

    # Clean text
    df['text'] = df['text'].astype(str).str.lower().str.replace(r'[^\w\s]', '', regex=True)
    df = df.reset_index(drop=True)
    return df
