"""Generate result figures for the report from eval/results CSVs.

Outputs (all PDF):
- fig_rouge.pdf    : ROUGE-1/L bars for GPT vs BART at 2-min clip vs full 30-min,
                     with value labels on bars.
- fig_bleu.pdf     : Per-direction sacreBLEU (en->zh, zh->en) for GPT vs NLLB,
                     loaded from eval/results/bleu_{gpt,nllb}.csv.
- fig_judge.pdf    : LLM-as-Judge pilot Likert scores.
- fig_sentiment.pdf: Sentiment timeline from eval/results/pipeline_run.json,
                     with legend outside the axes and staggered top-moment labels.
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


# ---------------------------------------------------------------------------
# fig_rouge
# ---------------------------------------------------------------------------
def _read_rouge_csvs() -> dict[str, dict[str, dict[str, float]]]:
    """Return {regime: {system: {rouge1, rougeL}}}.

    Falls back to the numbers reported in the paper if a CSV is missing, so
    the build never breaks on a clean checkout.
    """
    result: dict[str, dict[str, dict[str, float]]] = {
        "2min": {
            "gpt":  {"rouge1": 0.320, "rougeL": 0.194},
            "bart": {"rouge1": 0.254, "rougeL": 0.185},
        },
        "full": {
            "gpt":  {"rouge1": 0.393, "rougeL": 0.180},
            "bart": {"rouge1": 0.163, "rougeL": 0.116},
        },
    }
    pairs = [
        ("2min", "gpt",  RESULTS / "rouge_ami_gpt.csv"),
        ("2min", "bart", RESULTS / "rouge_ami_bart.csv"),
        ("full", "gpt",  RESULTS / "rouge_ami_gpt_full.csv"),
        ("full", "bart", RESULTS / "rouge_ami_bart_full.csv"),
    ]
    for regime, sys, path in pairs:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if {"rouge1", "rougeL"}.issubset(df.columns):
            result[regime][sys] = {
                "rouge1": float(df["rouge1"].iloc[0]),
                "rougeL": float(df["rougeL"].iloc[0]),
            }
    return result


def fig_rouge() -> None:
    r = _read_rouge_csvs()
    labels = ["2-min clip", "Full 30-min meeting"]
    x = np.arange(len(labels))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)

    def _bars(ax, metric, title):
        gpt_vals = [r["2min"]["gpt"][metric], r["full"]["gpt"][metric]]
        bart_vals = [r["2min"]["bart"][metric], r["full"]["bart"][metric]]
        b1 = ax.bar(x - width/2, gpt_vals, width, label="GPT-4o-mini", color=GPT_COLOR)
        b2 = ax.bar(x + width/2, bart_vals, width, label="BART-large-CNN", color=BASELINE_COLOR)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.set_ylim(0, 0.48)
        ax.grid(axis="y", alpha=0.3)
        for bars in (b1, b2):
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.008,
                        f"{h:.3f}", ha="center", fontsize=8)
        return b1, b2

    _bars(ax1, "rouge1", "ROUGE-1")
    _bars(ax2, "rougeL", "ROUGE-L")
    ax1.set_ylabel("Score")
    ax1.legend(frameon=False, loc="upper left", fontsize=9)
    plt.tight_layout()
    out = FIG_DIR / "fig_rouge.pdf"
    plt.savefig(out)
    plt.close()
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# fig_bleu
# ---------------------------------------------------------------------------
def _read_bleu() -> dict[str, dict[str, float]]:
    """{system: {direction: bleu}} from eval/results/bleu_{gpt,nllb}.csv."""
    def _load(path: Path) -> dict[str, float]:
        df = pd.read_csv(path)
        return {str(row["direction"]): float(row["bleu"]) for _, row in df.iterrows()}
    try:
        return {
            "gpt":  _load(RESULTS / "bleu_gpt.csv"),
            "nllb": _load(RESULTS / "bleu_nllb.csv"),
        }
    except FileNotFoundError as e:
        raise SystemExit(f"BLEU CSV missing: {e}")


def fig_bleu() -> None:
    bleu = _read_bleu()
    directions = ["en2zh", "zh2en"]
    dir_labels = [r"EN$\to$ZH" + "\n(zh tokenizer)", r"ZH$\to$EN" + "\n(13a tokenizer)"]

    gpt_vals = [bleu["gpt"][d] for d in directions]
    nllb_vals = [bleu["nllb"][d] for d in directions]

    x = np.arange(len(directions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    b1 = ax.bar(x - width/2, gpt_vals, width, color=GPT_COLOR, label="GPT-4o-mini")
    b2 = ax.bar(x + width/2, nllb_vals, width, color=BASELINE_COLOR, label="NLLB-200 (distilled 600M)")
    ax.set_xticks(x)
    ax.set_xticklabels(dir_labels)
    ax.set_ylabel("sacreBLEU")
    ax.set_title("Per-direction translation quality (n=13 per direction)")
    top = max(gpt_vals + nllb_vals) + 6
    ax.set_ylim(0, top)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                    f"{h:.2f}", ha="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    out = FIG_DIR / "fig_bleu.pdf"
    plt.savefig(out)
    plt.close()
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# fig_judge
# ---------------------------------------------------------------------------
def fig_judge() -> None:
    # LLM-as-Judge pilot (n=1 per system, summary pair).
    axes_labels = ["Faithfulness", "Coverage", "Specificity"]
    gpt_scores = [2.0, 3.0, 3.0]
    bart_scores = [3.0, 2.0, 2.0]

    csv_path = RESULTS / "llm_judge.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            needed = {"kind", "system", "faithfulness", "coverage", "specificity"}
            if needed.issubset(df.columns):
                summary_rows = df[df["kind"] == "summary"]

                def _row(sys_exact: str) -> list[float] | None:
                    s = summary_rows[summary_rows["system"] == sys_exact]
                    if s.empty:
                        return None
                    return [float(s["faithfulness"].iloc[0]),
                            float(s["coverage"].iloc[0]),
                            float(s["specificity"].iloc[0])]

                g = _row("gpt-4o-mini")
                b = _row("bart")
                if g and b:
                    gpt_scores, bart_scores = g, b
        except Exception:
            pass

    x = np.arange(len(axes_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    bars1 = ax.bar(x - width/2, gpt_scores, width, label="GPT-4o-mini", color=GPT_COLOR)
    bars2 = ax.bar(x + width/2, bart_scores, width, label="BART", color=BASELINE_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(axes_labels)
    ax.set_ylabel("GPT-4o judge Likert (1--5)")
    ax.set_title("LLM-as-Judge pilot (n=1 per system, summary pair)")
    ax.set_ylim(0, 5)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    for bars in (bars1, bars2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f"{bar.get_height():.0f}", ha="center", fontsize=9)
    plt.tight_layout()
    out = FIG_DIR / "fig_judge.pdf"
    plt.savefig(out)
    plt.close()
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# fig_sentiment
# ---------------------------------------------------------------------------
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

    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, (spk, pts) in enumerate(sorted(by_speaker.items())):
        xs = [p["t"] for p in pts]
        ys = [p["valence"] for p in pts]
        sizes = [20 + 80 * p["arousal"] for p in pts]
        color = palette[i % len(palette)]
        ax.plot(xs, ys, color=color, alpha=0.5, linewidth=1)
        ax.scatter(xs, ys, s=sizes, color=color, alpha=0.7, label=spk,
                   edgecolors="white", linewidth=0.5)

    # Top moments with staggered annotations to avoid overlap
    tops = data["sentiment"].get("top_moments") or []
    stagger_offsets = [(6, 10), (6, -14), (10, -4)]
    for k, idx in enumerate(tops):
        p = points[idx]
        ax.scatter(p["t"], p["valence"], marker="*", s=180, color="red",
                   zorder=5, edgecolors="black", linewidth=0.5)
        dx, dy = stagger_offsets[k % len(stagger_offsets)]
        ax.annotate(p["emotion"], (p["t"], p["valence"]),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=8, zorder=6)

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Meeting time (s)")
    ax.set_ylabel("Valence")
    ax.set_title(r"Sentiment timeline (AMI ES2004a, 2-min clip; marker size $\propto$ arousal)")
    ax.set_ylim(-1.2, 1.2)
    ax.grid(True, alpha=0.3)

    # Legend OUTSIDE the plot so speaker labels never collide with points.
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=min(5, len(by_speaker)),
        fontsize=8,
    )
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
