"""WER (EN) and CER (ZH) evaluation for Module 1 ASR.

Expects a manifest JSON: [{"audio": path, "reference": str, "lang": "en"|"zh"}, ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jiwer
import pandas as pd

from src.asr import transcribe


def evaluate(manifest_path: Path, out_path: Path) -> pd.DataFrame:
    manifest = json.loads(manifest_path.read_text())
    rows = []
    for item in manifest:
        transcript = transcribe(Path(item["audio"]))
        hyp = transcript.as_plain_text(with_speaker=False)
        ref = item["reference"]
        lang = item.get("lang", "en")
        if lang == "zh":
            metric = jiwer.cer(ref, hyp)
            name = "CER"
        else:
            metric = jiwer.wer(ref, hyp)
            name = "WER"
        rows.append({"audio": item["audio"], "lang": lang, "metric": name, "value": metric})
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(df.groupby(["lang", "metric"])["value"].mean())
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=Path("eval/data/asr_manifest.json"))
    p.add_argument("--out", type=Path, default=Path("eval/results/wer.csv"))
    args = p.parse_args()
    evaluate(args.manifest, args.out)
