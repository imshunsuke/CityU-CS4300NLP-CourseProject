from __future__ import annotations

import math
from collections import Counter
from typing import Any

from .llm_client import LLMClient
from .schemas import Emotion, SentimentPoint, SentimentReport, Transcript


EMOTIONS: list[Emotion] = ["neutral", "joy", "anger", "sadness", "surprise", "fear", "disgust"]


SYSTEM_PROMPT = (
    "You are an affective computing analyst. For each meeting utterance you receive, estimate:\n"
    "- valence in [-1, 1] (negative to positive)\n"
    "- arousal in [0, 1] (calm to intense)\n"
    "- one emotion label from: neutral, joy, anger, sadness, surprise, fear, disgust.\n"
    "Return a JSON object with a 'points' array of length equal to the input, preserving order. "
    "Each element: {i, valence, arousal, emotion}."
)


def _batch(items: list, n: int):
    for i in range(0, len(items), n):
        yield i, items[i : i + n]


def score_utterances(transcript: Transcript, client: LLMClient, *, batch_size: int = 20) -> list[SentimentPoint]:
    points: list[SentimentPoint] = []
    for offset, batch in _batch(transcript.utterances, batch_size):
        lines = [
            {"i": offset + k, "speaker": u.speaker, "text": u.text}
            for k, u in enumerate(batch)
        ]
        user = (
            f"Emotions allowed: {EMOTIONS}.\n"
            f"UTTERANCES (JSON):\n{lines}\n\n"
            'Return {"points":[{"i":...,"valence":...,"arousal":...,"emotion":"..."}, ...]}'
        )
        data: dict[str, Any] = client.chat_json(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
        )
        for entry in data.get("points", []):
            idx = int(entry["i"])
            if idx >= len(transcript.utterances):
                continue
            u = transcript.utterances[idx]
            emo = entry.get("emotion", "neutral")
            if emo not in EMOTIONS:
                emo = "neutral"
            points.append(
                SentimentPoint(
                    utterance_idx=idx,
                    speaker=u.speaker,
                    t=u.start,
                    valence=max(-1.0, min(1.0, float(entry.get("valence", 0.0)))),
                    arousal=max(0.0, min(1.0, float(entry.get("arousal", 0.0)))),
                    emotion=emo,
                )
            )
    points.sort(key=lambda p: p.utterance_idx)
    return points


def compute_engagement(transcript: Transcript) -> tuple[dict[str, float], float]:
    speaking_time: dict[str, float] = {}
    for u in transcript.utterances:
        speaking_time[u.speaker] = speaking_time.get(u.speaker, 0.0) + u.duration
    total = sum(speaking_time.values()) or 1.0
    entropy = 0.0
    for t in speaking_time.values():
        p = t / total
        if p > 0:
            entropy -= p * math.log(p)
    return speaking_time, entropy


def analyze_sentiment(
    transcript: Transcript,
    client: LLMClient,
    *,
    top_k_moments: int = 3,
) -> SentimentReport:
    points = score_utterances(transcript, client)
    if not points:
        return SentimentReport()
    valence_mean = sum(p.valence for p in points) / len(points)
    arousal_mean = sum(p.arousal for p in points) / len(points)
    ranked = sorted(
        range(len(points)),
        key=lambda i: abs(points[i].valence) + points[i].arousal,
        reverse=True,
    )
    top = ranked[:top_k_moments]
    speaking_time, entropy = compute_engagement(transcript)
    return SentimentReport(
        points=points,
        overall_valence=valence_mean,
        overall_arousal=arousal_mean,
        top_moments=top,
        speaking_time=speaking_time,
        engagement_entropy=entropy,
    )


def dominant_emotion(report: SentimentReport) -> Emotion:
    if not report.points:
        return "neutral"
    counts = Counter(p.emotion for p in report.points)
    return counts.most_common(1)[0][0]


def make_timeline_figure(report: SentimentReport):
    import plotly.graph_objects as go

    if not report.points:
        return go.Figure()

    fig = go.Figure()
    by_speaker: dict[str, list[SentimentPoint]] = {}
    for p in report.points:
        by_speaker.setdefault(p.speaker, []).append(p)
    for spk, pts in by_speaker.items():
        fig.add_trace(
            go.Scatter(
                x=[p.t for p in pts],
                y=[p.valence for p in pts],
                mode="lines+markers",
                name=f"{spk} valence",
                marker=dict(size=[5 + 10 * p.arousal for p in pts]),
            )
        )
    moments = [report.points[i] for i in report.top_moments]
    if moments:
        fig.add_trace(
            go.Scatter(
                x=[m.t for m in moments],
                y=[m.valence for m in moments],
                mode="markers+text",
                name="Top moments",
                marker=dict(size=16, symbol="star", color="red"),
                text=[m.emotion for m in moments],
                textposition="top center",
            )
        )
    fig.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Valence",
        yaxis=dict(range=[-1.1, 1.1]),
        title="Meeting sentiment timeline",
        height=420,
    )
    return fig
