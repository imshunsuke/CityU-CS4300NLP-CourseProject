"""LLM-as-Judge evaluation layer (Novelty contribution).

For each (transcript, summary, action_items) triple, ask GPT-4o to rate on:
- faithfulness (1-5): does output stay grounded in the transcript?
- coverage (1-5): does it capture the important content?
- specificity (1-5): is it concrete rather than vague?

Output CSV then correlate with ROUGE / F1 scores in the paper.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.asr import transcribe_from_json
from src.llm_client import LLMClient


JUDGE_SYSTEM = (
    "You are an expert evaluator of meeting assistant outputs. "
    "Given a meeting transcript and a system output (summary or action items), rate it on three axes "
    "from 1 (poor) to 5 (excellent): faithfulness, coverage, specificity. "
    "Return strict JSON: {\"faithfulness\":int,\"coverage\":int,\"specificity\":int,\"rationale\":str}."
)


def judge(transcript_text: str, output_text: str, kind: str, client: LLMClient) -> dict:
    user = (
        f"OUTPUT TYPE: {kind}\n\n"
        f"TRANSCRIPT:\n{transcript_text}\n\n"
        f"SYSTEM OUTPUT:\n{output_text}\n\n"
        "Evaluate now."
    )
    return client.chat_json(
        [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        model=client.judge_model,
        temperature=0.0,
    )


def evaluate(manifest_path: Path, out_path: Path) -> pd.DataFrame:
    """Manifest: [{"transcript_json": path, "output_text": str, "kind": "summary"|"actions", "system": str}, ...]"""
    manifest = json.loads(manifest_path.read_text())
    client = LLMClient()
    rows = []
    for item in manifest:
        transcript = transcribe_from_json(Path(item["transcript_json"]))
        scores = judge(
            transcript.as_plain_text(with_speaker=True)[:8000],
            item["output_text"],
            item["kind"],
            client,
        )
        rows.append(
            {
                "item": item["transcript_json"],
                "kind": item["kind"],
                "system": item.get("system", "unknown"),
                "faithfulness": scores.get("faithfulness"),
                "coverage": scores.get("coverage"),
                "specificity": scores.get("specificity"),
                "rationale": scores.get("rationale", ""),
            }
        )
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(df.groupby(["kind", "system"])[["faithfulness", "coverage", "specificity"]].mean())
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=Path("eval/data/judge_manifest.json"))
    p.add_argument("--out", type=Path, default=Path("eval/results/llm_judge.csv"))
    args = p.parse_args()
    evaluate(args.manifest, args.out)
