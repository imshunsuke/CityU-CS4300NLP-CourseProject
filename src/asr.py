from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from .schemas import Transcript, Utterance

load_dotenv()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_cached_transcript(audio_path: Path, cache_dir: Path = Path(".cache/asr")) -> Optional[Transcript]:
    digest = _sha256_file(audio_path)
    cache_file = cache_dir / f"{digest}.json"
    if cache_file.exists():
        return Transcript.model_validate_json(cache_file.read_text())
    return None


def save_transcript(t: Transcript, cache_dir: Path = Path(".cache/asr")) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{t.audio_sha256}.json").write_text(t.model_dump_json(indent=2))


def transcribe_with_whisperx(
    audio_path: Path,
    *,
    model_name: str = "large-v3",
    device: str = "cpu",
    compute_type: str = "int8",
    batch_size: int = 16,
    hf_token: Optional[str] = None,
) -> Transcript:
    import whisperx  # type: ignore

    hf_token = hf_token or os.getenv("HF_TOKEN")

    audio = whisperx.load_audio(str(audio_path))
    asr_model = whisperx.load_model(model_name, device=device, compute_type=compute_type)
    result = asr_model.transcribe(audio, batch_size=batch_size)
    lang = result.get("language", "auto")

    align_model, metadata = whisperx.load_align_model(language_code=lang, device=device)
    result = whisperx.align(result["segments"], align_model, metadata, audio, device)

    diar_pipeline = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
    diar = diar_pipeline(str(audio_path))
    result = whisperx.assign_word_speakers(diar, result)

    utterances: list[Utterance] = []
    for seg in result["segments"]:
        utterances.append(
            Utterance(
                speaker=seg.get("speaker") or "SPK_UNK",
                start=float(seg["start"]),
                end=float(seg["end"]),
                text=seg["text"].strip(),
                lang=lang,
            )
        )

    return Transcript(
        utterances=utterances,
        audio_path=str(audio_path),
        audio_sha256=_sha256_file(audio_path),
        dominant_lang=lang,
    )


def transcribe_from_json(json_path: Path, audio_path: Optional[Path] = None) -> Transcript:
    data = json.loads(Path(json_path).read_text())
    if audio_path is not None:
        data.setdefault("audio_path", str(audio_path))
        data.setdefault("audio_sha256", _sha256_file(audio_path))
    return Transcript.model_validate(data)


def transcribe(audio_path: Path, **kwargs) -> Transcript:
    audio_path = Path(audio_path)
    cached = load_cached_transcript(audio_path)
    if cached is not None:
        return cached
    t = transcribe_with_whisperx(audio_path, **kwargs)
    save_transcript(t)
    return t
