"""Parse AliMeeting Praat TextGrid annotations into eval manifests.

Produces:
- eval/data/ali_asr_manifest.json (for CER: audio + reference transcript)
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TGInterval:
    xmin: float
    xmax: float
    text: str
    speaker: str


INTERVAL_RE = re.compile(
    r"intervals\s*\[\s*\d+\s*\]:\s*"
    r"xmin\s*=\s*([-\d.]+)\s*"
    r"xmax\s*=\s*([-\d.]+)\s*"
    r'text\s*=\s*"((?:[^"\\]|\\.)*)"',
    re.DOTALL,
)

TIER_NAME_RE = re.compile(r'name\s*=\s*"([^"]+)"')


def parse_textgrid(path: Path) -> list[TGInterval]:
    """Parse a Praat TextGrid file; returns all non-empty intervals in time order."""
    content = path.read_text(encoding="utf-8", errors="replace")
    # Split into tiers by finding "item [n]:" boundaries
    tier_blocks = re.split(r"\n\s*item\s*\[\s*\d+\s*\]\s*:", content)
    out: list[TGInterval] = []
    for block in tier_blocks[1:]:
        name_match = TIER_NAME_RE.search(block)
        speaker = name_match.group(1) if name_match else "UNK"
        for m in INTERVAL_RE.finditer(block):
            xmin = float(m.group(1))
            xmax = float(m.group(2))
            text = m.group(3).strip().replace('""', '"')
            if text:
                out.append(TGInterval(xmin=xmin, xmax=xmax, text=text, speaker=speaker))
    out.sort(key=lambda i: i.xmin)
    return out


def reference_transcript(
    intervals: list[TGInterval],
    max_time: float | None = None,
    strip_punctuation: bool = True,
) -> str:
    """Concatenate intervals into a single reference string (for CER)."""
    if max_time is not None:
        intervals = [i for i in intervals if i.xmin < max_time]
    text = "".join(i.text for i in intervals)
    if strip_punctuation:
        # Remove common Chinese + English punctuation for CER.
        text = re.sub(r"[，。、；：？！,.;:?!\s]+", "", text)
    return text


def build_manifest(
    textgrid_path: Path,
    audio_path: Path,
    meeting_id: str,
    out_path: Path,
    clip_duration: float | None = None,
    append: bool = False,
) -> None:
    intervals = parse_textgrid(textgrid_path)
    ref = reference_transcript(intervals, max_time=clip_duration)
    row = {
        "audio": str(audio_path),
        "reference": ref,
        "lang": "zh",
        "meeting": meeting_id,
    }
    existing: list[dict] = []
    if append and out_path.exists():
        existing = json.loads(out_path.read_text())
    # de-dup by audio path
    existing = [r for r in existing if r.get("audio") != row["audio"]]
    existing.append(row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
    print(f"[alimeeting] {meeting_id}: {len(intervals)} intervals, {len(ref)} chars → {out_path}")


def _main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--textgrid", type=Path, required=True)
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--meeting", type=str, required=True)
    p.add_argument("--out", type=Path, default=Path("eval/data/ami_asr_manifest.json"))
    p.add_argument("--clip-duration", type=float, default=None)
    p.add_argument("--append", action="store_true", help="Append to existing manifest.")
    args = p.parse_args()
    build_manifest(args.textgrid, args.audio, args.meeting, args.out, args.clip_duration, args.append)


if __name__ == "__main__":
    _main()
