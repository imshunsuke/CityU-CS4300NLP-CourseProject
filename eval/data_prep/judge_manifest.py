"""Generate a manifest for eval/llm_judge.py.

For each row in ami_summary_manifest.json, produce GPT and BART summaries +
GPT action items, and write a judge_manifest.json with (transcript, output, kind, system).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.actions import extract_action_items
from src.asr import transcribe, transcribe_from_json
from src.llm_client import LLMClient
from src.summarize import summarize_with_baseline, summarize_with_llm


def _load_transcript(item: dict):
    if item.get("transcript_json"):
        return transcribe_from_json(Path(item["transcript_json"]))
    return transcribe(Path(item["audio"]))


def _summary_text(summary) -> str:
    return " ".join(summary.key_points + summary.decisions + summary.follow_ups)


def build(manifest_path: Path, out_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    client = LLMClient()
    rows: list[dict] = []
    for item in manifest:
        transcript = _load_transcript(item)
        tx_path = item.get("transcript_json") or "(via_audio)"

        gpt_summary = summarize_with_llm(transcript, client)
        bart_summary = summarize_with_baseline(transcript)
        action_items = extract_action_items(transcript, client)

        rows.append({
            "meeting": item.get("meeting", item.get("audio", "")),
            "transcript_json": item.get("transcript_json"),
            "audio": item.get("audio"),
            "output_text": _summary_text(gpt_summary),
            "kind": "summary",
            "system": "gpt-4o-mini",
        })
        rows.append({
            "meeting": item.get("meeting", item.get("audio", "")),
            "transcript_json": item.get("transcript_json"),
            "audio": item.get("audio"),
            "output_text": _summary_text(bart_summary),
            "kind": "summary",
            "system": "bart",
        })
        action_text = "\n".join(f"- [{a.priority}] {a.task} (owner: {a.assignee})" for a in action_items.items)
        rows.append({
            "meeting": item.get("meeting", item.get("audio", "")),
            "transcript_json": item.get("transcript_json"),
            "audio": item.get("audio"),
            "output_text": action_text,
            "kind": "actions",
            "system": "gpt-4o-mini",
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"[judge] wrote {len(rows)} rows to {out_path}")


def _main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=Path("eval/data/ami_summary_manifest.json"))
    p.add_argument("--out", type=Path, default=Path("eval/data/judge_manifest.json"))
    args = p.parse_args()
    build(args.manifest, args.out)


if __name__ == "__main__":
    _main()
