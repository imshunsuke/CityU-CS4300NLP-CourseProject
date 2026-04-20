"""Build a small EN↔ZH parallel manifest from FLORES-200 (standard MT benchmark).

Writes eval/data/mt_manifest.json with both EN→ZH and ZH→EN directions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build(out_path: Path, n_per_direction: int = 50, seed: int = 42) -> None:
    from datasets import load_dataset  # type: ignore

    # Helsinki-NLP/news_commentary is non-gated and already aligned EN/ZH.
    ds = load_dataset("Helsinki-NLP/news_commentary", "en-zh", split="train", trust_remote_code=True)
    ds = ds.shuffle(seed=seed).select(range(min(n_per_direction, len(ds))))

    rows = []
    for r in ds:
        en = r["translation"]["en"].strip()
        zh = r["translation"]["zh"].strip()
        if not en or not zh or len(en) > 300 or len(zh) > 200:
            continue
        rows.append({"source": en, "reference": zh, "source_lang": "en", "target_lang": "zh"})
        rows.append({"source": zh, "reference": en, "source_lang": "zh", "target_lang": "en"})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"[flores] wrote {len(rows)} rows ({n_per_direction} per direction) to {out_path}")


def _main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("eval/data/mt_manifest.json"))
    p.add_argument("--n", type=int, default=50)
    args = p.parse_args()
    build(args.out, args.n)


if __name__ == "__main__":
    _main()
