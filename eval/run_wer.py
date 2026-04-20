"""WER (EN) and CER (ZH) evaluation for Module 1 ASR.

Expects a manifest JSON: [{"audio": path, "reference": str, "lang": "en"|"zh"}, ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import re

import jiwer
import pandas as pd

from src.asr import transcribe


_EN_TRANSFORM = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)


def _normalize_zh(text: str) -> str:
    """Remove punctuation + whitespace for CER on Chinese."""
    return re.sub(r"[，。、；：？！,.;:?!\"'\s]+", "", text)


def evaluate(manifest_path: Path, out_path: Path) -> pd.DataFrame:
    manifest = json.loads(manifest_path.read_text())
    rows = []
    for item in manifest:
        transcript = transcribe(Path(item["audio"]))
        hyp = transcript.as_plain_text(with_speaker=False)
        ref = item["reference"]
        lang = item.get("lang", "en")
        if lang == "zh":
            hyp_n = _normalize_zh(hyp)
            ref_n = _normalize_zh(ref)
            metric = jiwer.cer(ref_n, hyp_n)
            name = "CER"
        else:
            metric = jiwer.wer(
                ref, hyp, reference_transform=_EN_TRANSFORM, hypothesis_transform=_EN_TRANSFORM
            )
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
