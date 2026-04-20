"""sacreBLEU evaluation for Module 3 translation.

Expects manifest: [{"source": str, "reference": str, "source_lang": "en"|"zh", "target_lang": "en"|"zh"}, ...]
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


def _llm_translate(source: str, src_lang, tgt_lang, client: LLMClient) -> str:
    u = Utterance(speaker="S", start=0.0, end=0.0, text=source, lang=src_lang)
    return translate_utterance_llm(u, [], tgt_lang, client)


def evaluate(manifest_path: Path, out_path: Path, system: str = "llm") -> pd.DataFrame:
    manifest = json.loads(manifest_path.read_text())
    client = LLMClient()

    nllb_tok = None
    nllb_model = None
    if system == "nllb":
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore
        nllb_tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        nllb_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

    NLLB_CODES = {"en": "eng_Latn", "zh": "zho_Hans"}
    hyps: list[str] = []
    refs: list[str] = []
    for item in manifest:
        src = item["source"]
        ref = item["reference"]
        if system == "llm":
            hyp = _llm_translate(src, item["source_lang"], item["target_lang"], client)
        else:
            nllb_tok.src_lang = NLLB_CODES[item["source_lang"]]
            inputs = nllb_tok(src, return_tensors="pt", truncation=True, max_length=512)
            out = nllb_model.generate(
                **inputs,
                forced_bos_token_id=nllb_tok.convert_tokens_to_ids(NLLB_CODES[item["target_lang"]]),
                max_length=512,
            )
            hyp = nllb_tok.batch_decode(out, skip_special_tokens=True)[0]
        hyps.append(hyp)
        refs.append(ref)

    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"system": system, "bleu": bleu.score, "n": len(hyps)}]).to_csv(out_path, index=False)
    print(f"{system} BLEU: {bleu.score:.2f}")
    return pd.DataFrame({"source": [m["source"] for m in manifest], "hyp": hyps, "ref": refs})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=Path("eval/data/mt_manifest.json"))
    p.add_argument("--out", type=Path, default=Path("eval/results/bleu.csv"))
    p.add_argument("--system", default="llm", choices=["llm", "nllb"])
    args = p.parse_args()
    evaluate(args.manifest, args.out, system=args.system)
