"""Action-item extraction evaluation: task-level P/R/F1 + assignee accuracy.

Expects manifest: [{"transcript_json": path, "gold_items":[{task, assignee}, ...]}, ...]
Matching strategy: greedy on best cosine similarity of task strings (text-embedding-3-small).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.actions import extract_action_items
from src.asr import transcribe, transcribe_from_json
from src.llm_client import LLMClient


def _load_transcript(item: dict):
    if item.get("transcript_json"):
        return transcribe_from_json(Path(item["transcript_json"]))
    return transcribe(Path(item["audio"]))


_EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_st_model = None


def _embed_local(texts: list[str]) -> np.ndarray:
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _st_model = SentenceTransformer(_EMBED_MODEL_NAME)
    return _st_model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)


def _match_gold_to_pred(gold: list[dict], pred: list[dict], client: LLMClient, threshold: float = 0.55):
    if not gold or not pred:
        return [], []
    gold_texts = [g["task"] for g in gold]
    pred_texts = [p["task"] for p in pred]
    g_emb = _embed_local(gold_texts)
    p_emb = _embed_local(pred_texts)
    sim = g_emb @ p_emb.T

    matched_g, matched_p = [], []
    used_p: set[int] = set()
    pairs = [(i, j, sim[i, j]) for i in range(len(gold)) for j in range(len(pred))]
    pairs.sort(key=lambda x: -x[2])
    for i, j, s in pairs:
        if s < threshold:
            break
        if i in matched_g or j in used_p:
            continue
        matched_g.append(i)
        matched_p.append(j)
        used_p.add(j)
    return matched_g, matched_p


def evaluate(manifest_path: Path, out_path: Path) -> pd.DataFrame:
    manifest = json.loads(manifest_path.read_text())
    client = LLMClient()
    rows = []
    tp = fp = fn = assign_correct = 0
    for item in manifest:
        transcript = _load_transcript(item)
        pred_list = extract_action_items(transcript, client)
        pred = [p.model_dump() for p in pred_list.items]
        gold = item["gold_items"]
        mg, mp = _match_gold_to_pred(gold, pred, client)
        tp_i = len(mg)
        fp_i = len(pred) - tp_i
        fn_i = len(gold) - tp_i
        ac_i = sum(1 for gi, pi in zip(mg, mp) if gold[gi].get("assignee") == pred[pi].get("assignee"))
        tp += tp_i
        fp += fp_i
        fn += fn_i
        assign_correct += ac_i
        rows.append(
            {
                "item": item["transcript_json"],
                "gold": len(gold),
                "pred": len(pred),
                "matched": tp_i,
                "assignee_correct": ac_i,
            }
        )
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    assignee_acc = assign_correct / tp if tp else 0.0

    summary = pd.DataFrame([{"precision": precision, "recall": recall, "f1": f1, "assignee_acc": assignee_acc}])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    pd.DataFrame(rows).to_csv(out_path.with_suffix(".per_item.csv"), index=False)
    print(summary)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=Path("eval/data/actions_manifest.json"))
    p.add_argument("--out", type=Path, default=Path("eval/results/actions_f1.csv"))
    args = p.parse_args()
    evaluate(args.manifest, args.out)
