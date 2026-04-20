"""Module 5 sentiment evaluation on MELD (7-class emotion classification).

MELD loads from HuggingFace `declare-lab/MELD` (dev split). We sample N utterances
and compare GPT-4o-mini's emotion label against the gold label.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.llm_client import LLMClient
from src.schemas import Transcript, Utterance
from src.sentiment import EMOTIONS, score_utterances


def _load_meld(n_samples: int, csv_path: Path = Path("data/meld/dev_sent_emo.csv"), seed: int = 42) -> list[dict]:
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    out = []
    for _, row in df.iterrows():
        emo = str(row.get("Emotion", "neutral")).lower()
        if emo not in EMOTIONS:
            continue
        out.append({"speaker": str(row.get("Speaker", "S")), "text": str(row["Utterance"]), "emotion": emo})
        if len(out) >= n_samples:
            break
    return out


def evaluate(n_samples: int, out_path: Path) -> pd.DataFrame:
    samples = _load_meld(n_samples)
    utterances = [
        Utterance(speaker=s["speaker"], start=float(i), end=float(i + 1), text=s["text"], lang="en")
        for i, s in enumerate(samples)
    ]
    transcript = Transcript(
        utterances=utterances,
        audio_path="meld",
        audio_sha256="meld",
        dominant_lang="en",
    )
    client = LLMClient()
    points = score_utterances(transcript, client)

    preds = ["neutral"] * len(samples)
    for p in points:
        if p.utterance_idx < len(preds):
            preds[p.utterance_idx] = p.emotion
    golds = [s["emotion"] for s in samples]

    acc = accuracy_score(golds, preds)
    macro_f1 = f1_score(golds, preds, average="macro", zero_division=0)
    report = classification_report(golds, preds, zero_division=0)
    print(f"Accuracy: {acc:.3f}  Macro-F1: {macro_f1:.3f}")
    print(report)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"text": [s["text"] for s in samples], "gold": golds, "pred": preds})
    df.to_csv(out_path, index=False)
    pd.DataFrame([{"accuracy": acc, "macro_f1": macro_f1, "n": len(samples)}]).to_csv(
        out_path.with_name("sentiment_summary.csv"), index=False
    )
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--out", type=Path, default=Path("eval/results/sentiment_meld.csv"))
    args = p.parse_args()
    evaluate(args.n, args.out)
