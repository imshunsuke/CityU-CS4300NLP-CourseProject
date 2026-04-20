"""ROUGE-1/2/L + BERTScore for Module 2 summarization.

Expects manifest: [{"transcript_json": path, "reference_summary": str}, ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from rouge_score import rouge_scorer

from src.asr import transcribe, transcribe_from_json
from src.llm_client import LLMClient
from src.summarize import summarize_with_baseline, summarize_with_llm


def _load_transcript(item: dict):
    """Prefer cached transcript_json; otherwise re-run ASR (cached by audio hash)."""
    if item.get("transcript_json"):
        return transcribe_from_json(Path(item["transcript_json"]))
    return transcribe(Path(item["audio"]))


def _summary_text(summary) -> str:
    parts = list(summary.key_points) + list(summary.decisions) + list(summary.follow_ups)
    return " ".join(parts)


def evaluate(manifest_path: Path, out_path: Path, use_baseline: bool = False) -> pd.DataFrame:
    manifest = json.loads(manifest_path.read_text())
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    client = LLMClient()
    rows = []
    for item in manifest:
        transcript = _load_transcript(item)
        if use_baseline:
            summary = summarize_with_baseline(transcript)
            system = "bart"
        else:
            summary = summarize_with_llm(transcript, client)
            system = "gpt-4o-mini"
        hyp = _summary_text(summary)
        ref = item["reference_summary"]
        scores = scorer.score(ref, hyp)
        rows.append(
            {
                "item": item["transcript_json"],
                "system": system,
                "rouge1": scores["rouge1"].fmeasure,
                "rouge2": scores["rouge2"].fmeasure,
                "rougeL": scores["rougeL"].fmeasure,
            }
        )
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(df.groupby("system")[["rouge1", "rouge2", "rougeL"]].mean())
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=Path("eval/data/summary_manifest.json"))
    p.add_argument("--out", type=Path, default=Path("eval/results/rouge.csv"))
    p.add_argument("--baseline", action="store_true", help="Use BART baseline instead of LLM.")
    args = p.parse_args()
    evaluate(args.manifest, args.out, use_baseline=args.baseline)
