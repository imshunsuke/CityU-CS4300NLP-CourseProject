"""Generate result figures for the report from eval/results CSVs.

Outputs (all PDF):
- fig_rouge.pdf    : ROUGE-1/L bars for GPT vs BART at 2-min clip vs full 30-min
- fig_bleu.pdf     : BLEU bar for GPT vs NLLB
- fig_judge.pdf    : LLM-as-Judge Likert scores on faithfulness/coverage/specificity
- fig_sentiment.pdf: sentiment timeline from eval/results/pipeline_run.json
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = Path(__file__).parent
RESULTS = ROOT / "eval" / "results"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

GPT_COLOR = "#1f77b4"
BASELINE_COLOR = "#ff7f0e"


def fig_rouge() -> None:
    gpt_2min = {"rouge1": 0.320, "rougeL": 0.194}
    bart_2min = {"rouge1": 0.254, "rougeL": 0.185}
    gpt_full = {"rouge1": 0.393, "rougeL": 0.180}
    bart_full = {"rouge1": 0.163, "rougeL": 0.116}

    labels = ["2-min clip", "Full 30-min meeting"]
    x = np.arange(len(labels))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)

    ax1.bar(x - width/2, [gpt_2min["rouge1"], gpt_full["rouge1"]], width,
            label="GPT-4o-mini", color=GPT_COLOR)
    ax1.bar(x + width/2, [bart_2min["rouge1"], bart_full["rouge1"]], width,
            label="BART", color=BASELINE_COLOR)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Score")
    ax1.set_title("ROUGE-1")
    ax1.legend(frameon=False, loc="upper right")
    ax1.set_ylim(0, 0.45)
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x - width/2, [gpt_2min["rougeL"], gpt_full["rougeL"]], width,
            label="GPT-4o-mini", color=GPT_COLOR)
    ax2.bar(x + width/2, [bart_2min["rougeL"], bart_full["rougeL"]], width,
            label="BART", color=BASELINE_COLOR)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_title("ROUGE-L")
    ax2.set_ylim(0, 0.45)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = FIG_DIR / "fig_rouge.pdf"
    plt.savefig(out)
    plt.close()
    print(f"wrote {out}")


def fig_bleu() -> None:
    systems = ["GPT-4o-mini\n(context-aware)", "NLLB-200\n(distilled 600M)"]
    scores = [22.18, 16.29]
    colors = [GPT_COLOR, BASELINE_COLOR]

    fig, ax = plt.subplots(figsize=(4.5, 2.6))
    bars = ax.bar(systems, scores, color=colors, width=0.5)
    ax.set_ylabel("sacreBLEU")
    ax.set_title("Translation quality on 60 news-commentary EN–ZH pairs")
    ax.set_ylim(0, 28)
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, scores):
        ax.text(b.get_x() + b.get_width()/2, v + 0.4, f"{v:.2f}",
                ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    out = FIG_DIR / "fig_bleu.pdf"
    plt.savefig(out)
    plt.close()
    print(f"wrote {out}")


def fig_judge() -> None:
    axes_labels = ["Faithfulness", "Coverage", "Specificity"]
    gpt_scores = [2.0, 3.0, 3.0]
    bart_scores = [3.0, 2.0, 2.0]

    x = np.arange(len(axes_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(5, 2.8))
    bars1 = ax.bar(x - width/2, gpt_scores, width, label="GPT-4o-mini", color=GPT_COLOR)
    bars2 = ax.bar(x + width/2, bart_scores, width, label="BART", color=BASELINE_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(axes_labels)
    ax.set_ylabel("GPT-4o judge Likert (1--5)")
    ax.set_title("LLM-as-Judge: summarization trade-off")
    ax.set_ylim(0, 5)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    for bars in (bars1, bars2):
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.1,
                    f"{b.get_height():.0f}", ha="center", fontsize=9)
    plt.tight_layout()
    out = FIG_DIR / "fig_judge.pdf"
    plt.savefig(out)
    plt.close()
    print(f"wrote {out}")


def fig_sentiment() -> None:
    run_path = RESULTS / "pipeline_run.json"
    if not run_path.exists():
        print(f"SKIP fig_sentiment: {run_path} not found")
        return
    data = json.loads(run_path.read_text())
    points = data["sentiment"]["points"]
    if not points:
        print("SKIP fig_sentiment: no points")
        return

    by_speaker: dict[str, list] = {}
    for p in points:
        by_speaker.setdefault(p["speaker"], []).append(p)

    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, (spk, pts) in enumerate(sorted(by_speaker.items())):
        xs = [p["t"] for p in pts]
        ys = [p["valence"] for p in pts]
        sizes = [20 + 80 * p["arousal"] for p in pts]
        color = palette[i % len(palette)]
        ax.plot(xs, ys, marker="o", markersize=0, color=color, alpha=0.5, linewidth=1)
        ax.scatter(xs, ys, s=sizes, color=color, alpha=0.7, label=spk, edgecolors="white")

    tops = data["sentiment"].get("top_moments") or []
    for idx in tops:
        p = points[idx]
        ax.scatter(p["t"], p["valence"], marker="*", s=200, color="red",
                   zorder=5, edgecolors="black", linewidth=0.5)
        ax.annotate(p["emotion"], (p["t"], p["valence"]),
                    xytext=(4, 6), textcoords="offset points", fontsize=8)

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Meeting time (s)")
    ax.set_ylabel("Valence")
    ax.set_title("Sentiment timeline (AMI ES2004a, 2-min clip; size $\\propto$ arousal)")
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="lower right", fontsize=8, ncol=4)
    plt.tight_layout()
    out = FIG_DIR / "fig_sentiment.pdf"
    plt.savefig(out)
    plt.close()
    print(f"wrote {out}")


def main() -> None:
    fig_rouge()
    fig_bleu()
    fig_judge()
    fig_sentiment()


if __name__ == "__main__":
    main()
