from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from .actions import extract_action_items
from .asr import transcribe, transcribe_from_json
from .llm_client import LLMClient
from .qa import MeetingQA
from .schemas import Language, MeetingAnalysis, Transcript
from .sentiment import analyze_sentiment
from .summarize import summarize
from .translate import translate_transcript


def run(
    audio_path: Optional[Path] = None,
    transcript_json: Optional[Path] = None,
    target_lang: Language = "zh",
    client: Optional[LLMClient] = None,
) -> MeetingAnalysis:
    if client is None:
        client = LLMClient()

    if transcript_json is not None:
        transcript: Transcript = transcribe_from_json(Path(transcript_json))
    elif audio_path is not None:
        transcript = transcribe(Path(audio_path))
    else:
        raise ValueError("Provide audio_path or transcript_json.")

    summary = summarize(transcript, client)

    translation = None
    if target_lang != transcript.dominant_lang:
        translation = translate_transcript(transcript, target_lang, client)

    actions = extract_action_items(transcript, client)
    sentiment = analyze_sentiment(transcript, client)

    return MeetingAnalysis(
        transcript=transcript,
        summary=summary,
        translation=translation,
        action_items=actions,
        sentiment=sentiment,
    )


def build_qa(transcript: Transcript, client: Optional[LLMClient] = None) -> MeetingQA:
    return MeetingQA(transcript, client or LLMClient())


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run the Smart Meeting Assistant pipeline.")
    parser.add_argument("--audio", type=Path, help="Path to meeting audio file.")
    parser.add_argument("--transcript", type=Path, help="Use an existing transcript JSON instead of ASR.")
    parser.add_argument("--target-lang", default="zh", choices=["en", "zh"])
    parser.add_argument("--out", type=Path, default=Path("eval/results/pipeline_run.json"))
    args = parser.parse_args()

    result = run(
        audio_path=args.audio,
        transcript_json=args.transcript,
        target_lang=args.target_lang,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(result.model_dump_json(indent=2))
    print(f"Saved analysis to {args.out}")
    print(f"- {len(result.transcript.utterances)} utterances, speakers: {result.transcript.speakers}")
    print(f"- summary: {len(result.summary.key_points)} key points")
    print(f"- action items: {len(result.action_items.items)}")
    print(f"- sentiment points: {len(result.sentiment.points)}")


if __name__ == "__main__":
    _main()
