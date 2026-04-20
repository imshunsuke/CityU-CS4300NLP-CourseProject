from __future__ import annotations

from typing import Literal

from .llm_client import LLMClient
from .schemas import Summary, Transcript


SYSTEM_PROMPT = (
    "You are a meeting note-taker. Given a meeting transcript, produce a JSON object with three arrays: "
    "key_points (main topics discussed), decisions (concrete choices made), and follow_ups (action items, "
    "open questions, or next steps). Each array entry should be one concise sentence. "
    "Write in the same language as the transcript's dominant language. Return ONLY valid JSON."
)


def summarize_with_llm(
    transcript: Transcript,
    client: LLMClient,
    *,
    max_items: int = 5,
) -> Summary:
    text = transcript.as_plain_text(with_speaker=True)
    lang_hint = "Chinese" if transcript.dominant_lang == "zh" else "English"
    user = (
        f"Language: {lang_hint}. Cap each array at {max_items} items.\n\n"
        f"TRANSCRIPT:\n{text}\n\n"
        'Return JSON schema: {"key_points":[], "decisions":[], "follow_ups":[]}'
    )
    data = client.chat_json(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return Summary(
        key_points=list(data.get("key_points", []))[:max_items],
        decisions=list(data.get("decisions", []))[:max_items],
        follow_ups=list(data.get("follow_ups", []))[:max_items],
    )


def summarize_with_baseline(
    transcript: Transcript,
    backend: Literal["bart", "mt5"] = "bart",
    *,
    max_length: int = 180,
    min_length: int = 40,
) -> Summary:
    from transformers import pipeline  # type: ignore

    model_name = (
        "facebook/bart-large-cnn" if backend == "bart" else "csebuetnlp/mT5_multilingual_XLSum"
    )
    summarizer = pipeline("summarization", model=model_name)
    text = transcript.as_plain_text(with_speaker=False)
    text = text[:4000]
    out = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    summary_text = out[0]["summary_text"] if out else ""
    sentences = [s.strip() for s in summary_text.replace("。", ".").split(".") if s.strip()]
    return Summary(key_points=sentences, decisions=[], follow_ups=[])


def summarize(transcript: Transcript, client: LLMClient, **kwargs) -> Summary:
    return summarize_with_llm(transcript, client, **kwargs)
