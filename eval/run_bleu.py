"""sacreBLEU evaluation for Module 3 translation.

Expects manifest: [{"source": str, "reference": str, "source_lang": "en"|"zh", "target_lang": "en"|"zh"}, ...]

Scores each translation direction separately with the direction-appropriate
sacreBLEU tokenizer (13a for English targets, zh for Chinese targets). BLEU
is not linearly aggregable across directions, so we do *not* emit a combined
corpus-BLEU row. Per-row hypotheses are written out for auditing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import sacrebleu

from src.llm_client import LLMClient
from src.schemas import Utterance
from src.translate import translate_utterance_llm


# sacreBLEU tokenizer per target language.
TOKENIZER = {"en": "13a", "zh": "zh"}


def _llm_translate(source: str, src_lang, tgt_lang, client: LLMClient) -> str:
    u = Utterance(speaker="S", start=0.0, end=0.0, text=source, lang=src_lang)
    return translate_utterance_llm(u, [], tgt_lang, client)


def evaluate(manifest_path: Path, out_path: Path, system: str = "llm") -> pd.DataFrame:
    manifest = json.loads(manifest_path.read_text())
    client = LLMClient()

    nllb_tok = None
    nllb_model = None
    nllb_device = "cpu"
    if system == "nllb":
        import torch  # type: ignore
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore
        nllb_tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        nllb_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
        if torch.backends.mps.is_available():
            nllb_device = "mps"
            nllb_model = nllb_model.to(nllb_device)
        print(f"[nllb] device: {nllb_device}")

    NLLB_CODES = {"en": "eng_Latn", "zh": "zho_Hans"}

    rows: list[dict] = []
    for item in manifest:
        src = item["source"]
        ref = item["reference"]
        if system == "llm":
            hyp = _llm_translate(src, item["source_lang"], item["target_lang"], client)
        else:
            nllb_tok.src_lang = NLLB_CODES[item["source_lang"]]
            inputs = nllb_tok(src, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(nllb_device) for k, v in inputs.items()}
            out = nllb_model.generate(
                **inputs,
                forced_bos_token_id=nllb_tok.convert_tokens_to_ids(NLLB_CODES[item["target_lang"]]),
                max_length=512,
            )
            hyp = nllb_tok.batch_decode(out, skip_special_tokens=True)[0]
        rows.append(
            {
                "source_lang": item["source_lang"],
                "target_lang": item["target_lang"],
                "direction": f'{item["source_lang"]}2{item["target_lang"]}',
                "source": src,
                "reference": ref,
                "hyp": hyp,
            }
        )

    per_row = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    per_row_path = out_path.with_suffix(".per_row.csv")
    per_row.to_csv(per_row_path, index=False)

    summary: list[dict] = []
    for direction, sub in per_row.groupby("direction"):
        tgt = sub["target_lang"].iloc[0]
        score = sacrebleu.corpus_bleu(
            sub["hyp"].tolist(), [sub["reference"].tolist()], tokenize=TOKENIZER[tgt]
        )
        summary.append(
            {
                "system": system,
                "direction": direction,
                "tokenizer": TOKENIZER[tgt],
                "bleu": score.score,
                "n": len(sub),
            }
        )

    pd.DataFrame(summary).to_csv(out_path, index=False)
    for r in summary:
        print(f"{r['system']} {r['direction']:<10} ({r['tokenizer']}): BLEU {r['bleu']:.2f} (n={r['n']})")
    return per_row


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=Path("eval/data/mt_manifest.json"))
    p.add_argument("--out", type=Path, default=Path("eval/results/bleu.csv"))
    p.add_argument("--system", default="llm", choices=["llm", "nllb"])
    args = p.parse_args()
    evaluate(args.manifest, args.out, system=args.system)
