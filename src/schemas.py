from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


Language = Literal["en", "zh", "auto"]
Emotion = Literal["neutral", "joy", "anger", "sadness", "surprise", "fear", "disgust"]
Priority = Literal["H", "M", "L"]


class Utterance(BaseModel):
    speaker: str
    start: float
    end: float
    text: str
    lang: Language = "auto"

    @property
    def duration(self) -> float:
        return self.end - self.start


class Transcript(BaseModel):
    utterances: list[Utterance]
    audio_path: str
    audio_sha256: str
    dominant_lang: Language = "auto"

    @property
    def speakers(self) -> list[str]:
        seen: list[str] = []
        for u in self.utterances:
            if u.speaker not in seen:
                seen.append(u.speaker)
        return seen

    def as_plain_text(self, with_speaker: bool = True) -> str:
        lines = []
        for u in self.utterances:
            prefix = f"[{u.speaker}] " if with_speaker else ""
            lines.append(f"{prefix}{u.text}")
        return "\n".join(lines)


class Summary(BaseModel):
    key_points: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    follow_ups: list[str] = Field(default_factory=list)


class TranslationPair(BaseModel):
    source: str
    source_lang: Language
    target: str
    target_lang: Language


class TranslatedTranscript(BaseModel):
    pairs: list[TranslationPair]
    target_lang: Language


class ActionItem(BaseModel):
    task: str
    assignee: str
    due: Optional[str] = None
    priority: Priority = "M"
    evidence_span: str


class ActionItemList(BaseModel):
    items: list[ActionItem] = Field(default_factory=list)


class SentimentPoint(BaseModel):
    utterance_idx: int
    speaker: str
    t: float
    valence: float = Field(ge=-1.0, le=1.0)
    arousal: float = Field(ge=0.0, le=1.0)
    emotion: Emotion


class SentimentReport(BaseModel):
    points: list[SentimentPoint] = Field(default_factory=list)
    overall_valence: float = 0.0
    overall_arousal: float = 0.0
    top_moments: list[int] = Field(
        default_factory=list,
        description="Indices into points for top emotionally salient moments.",
    )
    speaking_time: dict[str, float] = Field(default_factory=dict)
    engagement_entropy: float = 0.0


class QAResult(BaseModel):
    question: str
    answer: str
    cited_utterance_indices: list[int] = Field(default_factory=list)


class MeetingAnalysis(BaseModel):
    transcript: Transcript
    summary: Summary
    translation: Optional[TranslatedTranscript] = None
    action_items: ActionItemList
    sentiment: SentimentReport
