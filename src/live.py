"""Live microphone streaming: chunked Whisper transcription + per-utterance translation.

Gradio emits a (sample_rate, np.int16) tuple on every micro-chunk from the browser.
We buffer audio until we have at least `CHUNK_SECONDS` of new material, run a
compact Whisper model on that chunk, append the resulting utterances to the
running transcript, and translate each new utterance via OpenAI.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from functools import lru_cache
from typing import Optional

import numpy as np

from .llm_client import LLMClient
from .schemas import Language, Utterance
from .translate import translate_utterance_llm


CHUNK_SECONDS = 1.0        # minimum audio (s) per Whisper call
TARGET_SR = 16000          # Whisper's expected sample rate
WHISPER_MODEL = "base"     # Live uses a smaller, faster model (145 MB, ~2x faster than small)
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

_TRANSLATE_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="mic-translate")


@lru_cache(maxsize=1)
def _get_whisper():
    from faster_whisper import WhisperModel  # type: ignore

    return WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)


def _resample_to_16k(audio: np.ndarray, src_sr: int) -> np.ndarray:
    if src_sr == TARGET_SR:
        return audio
    n_out = int(round(len(audio) * TARGET_SR / src_sr))
    idx = np.linspace(0, len(audio) - 1, n_out).astype(np.int64)
    return audio[idx]


def _to_float32_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if audio.size and np.abs(audio).max() > 1.5:
        # int16 → float in [-1, 1]
        audio = audio / 32768.0
    return audio


class MicStream:
    """Stateful buffer for Gradio streaming audio callbacks."""

    def __init__(self, target_lang: Language = "zh", client: Optional[LLMClient] = None):
        self.target_lang: Language = target_lang
        self.client = client or LLMClient()
        self.buffer: np.ndarray = np.zeros(0, dtype=np.float32)
        self.buffer_sr: int = TARGET_SR
        self.elapsed_s: float = 0.0
        self.transcript: list[dict] = []
        self.translation: list[dict] = []
        self._counter = 0
        self._pending: list[tuple[int, Future]] = []  # (idx_into_translation, future)

    def reset(self):
        self.buffer = np.zeros(0, dtype=np.float32)
        self.elapsed_s = 0.0
        self.transcript = []
        self.translation = []
        self._counter = 0
        self._pending = []

    def _drain_pending(self) -> None:
        """Fill in translations that finished since last snapshot."""
        still_pending: list[tuple[int, Future]] = []
        for idx, fut in self._pending:
            if fut.done():
                try:
                    self.translation[idx]["target"] = fut.result()
                except Exception as e:
                    self.translation[idx]["target"] = f"[translate error: {type(e).__name__}]"
            else:
                still_pending.append((idx, fut))
        self._pending = still_pending

    def force_flush(self) -> dict:
        """Flush whatever is buffered (called on pause button)."""
        if len(self.buffer) / TARGET_SR < 0.3:
            return self.snapshot("buffer too short to flush")
        self._flush_chunk()
        return self.snapshot(f"forced flush @ {self.elapsed_s:.1f}s")

    def _flush_chunk(self) -> None:
        """Run Whisper on the current buffer, update transcript immediately, dispatch
        translation to a background thread pool. Translation results get merged in
        on subsequent snapshots via _drain_pending().
        """
        model = _get_whisper()
        segments, info = model.transcribe(
            self.buffer,
            language=None,
            vad_filter=True,
            beam_size=1,
            condition_on_previous_text=False,
        )
        lang: Language = info.language if info.language in ("en", "zh") else "auto"
        t0 = self.elapsed_s
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            u = Utterance(
                speaker=f"SPK_{self._counter % 2}",
                start=t0 + seg.start,
                end=t0 + seg.end,
                text=text,
                lang=lang,
            )
            self.transcript.append(
                {
                    "speaker": u.speaker,
                    "start": round(u.start, 2),
                    "end": round(u.end, 2),
                    "text": u.text,
                    "lang": lang,
                }
            )
            idx = len(self.translation)
            self.translation.append(
                {
                    "source": u.text,
                    "target": "…translating…",
                    "source_lang": lang,
                    "target_lang": self.target_lang,
                }
            )
            if lang == self.target_lang:
                self.translation[idx]["target"] = u.text
            else:
                fut = _TRANSLATE_POOL.submit(
                    translate_utterance_llm, u, [], self.target_lang, self.client
                )
                self._pending.append((idx, fut))
            self._counter += 1

        self.elapsed_s += len(self.buffer) / self.buffer_sr
        self.buffer = np.zeros(0, dtype=np.float32)

    def push(self, sample_rate: int, audio: np.ndarray) -> dict:
        """Append a new mic chunk. Returns a snapshot dict (may be unchanged)."""
        # First, drain any translations that finished since last tick.
        self._drain_pending()

        if audio is None or audio.size == 0:
            return self.snapshot("idle")

        audio_f32 = _to_float32_mono(audio)
        audio_16k = _resample_to_16k(audio_f32, sample_rate)
        self.buffer = np.concatenate([self.buffer, audio_16k])
        self.buffer_sr = TARGET_SR

        if len(self.buffer) / TARGET_SR >= CHUNK_SECONDS:
            self._flush_chunk()
            return self.snapshot(f"processed chunk @ {self.elapsed_s:.1f}s")

        pending = len(self.buffer) / TARGET_SR
        return self.snapshot(f"buffering… {pending:.1f}s / {CHUNK_SECONDS:.0f}s")

    def snapshot(self, status: str) -> dict:
        return {
            "elapsed_audio_s": self.elapsed_s,
            "transcript": list(self.transcript),
            "translation": list(self.translation),
            "status": status,
        }
