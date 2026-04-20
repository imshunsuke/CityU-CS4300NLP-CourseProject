"""Parse AMI NXT XML annotations into evaluation-ready manifests.

Produces:
- eval/data/ami_asr_manifest.json  (for WER: audio + reference transcript)
- eval/data/ami_summary_manifest.json  (for ROUGE: transcript_json + reference summary)
- eval/data/ami_actions_manifest.json  (for action-item F1: transcript + gold actions)
"""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


NITE_NS = "{http://nite.sourceforge.net/}"


@dataclass
class AMIWord:
    text: str
    start: float
    end: float
    speaker: str


def parse_words_xml(xml_path: Path, speaker_label: str) -> list[AMIWord]:
    """Extract word-level tokens from one speaker's NXT .words.xml file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    words: list[AMIWord] = []
    for elem in root:
        tag = elem.tag
        if tag.endswith("}w") or tag == "w":
            if elem.attrib.get("punc") == "true":
                continue
            start = elem.attrib.get("starttime")
            end = elem.attrib.get("endtime")
            text = (elem.text or "").strip()
            if text and start is not None and end is not None:
                words.append(AMIWord(text, float(start), float(end), speaker_label))
    return words


def collect_meeting_words(annotations_root: Path, meeting: str) -> list[AMIWord]:
    all_words: list[AMIWord] = []
    words_dir = annotations_root / "words"
    for speaker in ["A", "B", "C", "D", "E"]:
        p = words_dir / f"{meeting}.{speaker}.words.xml"
        if p.exists():
            all_words.extend(parse_words_xml(p, f"SPEAKER_{speaker}"))
    all_words.sort(key=lambda w: w.start)
    return all_words


def reference_transcript(words: list[AMIWord], max_time: Optional[float] = None) -> str:
    if max_time is not None:
        words = [w for w in words if w.start < max_time]
    return " ".join(w.text for w in words)


def parse_abstract_summary(annotations_root: Path, meeting: str) -> dict[str, list[str]]:
    """Return {'abstract': [...], 'actions': [...], 'decisions': [...], 'problems': [...]}.

    AMI has multiple annotators per meeting; we merge all of them and dedupe.
    """
    abs_dir = annotations_root / "abstractive"
    out = {"abstract": [], "actions": [], "decisions": [], "problems": []}
    for f in sorted(abs_dir.glob(f"{meeting}.*.abssumm.xml")) + [abs_dir / f"{meeting}.abssumm.xml"]:
        if not f.exists():
            continue
        try:
            tree = ET.parse(f)
        except ET.ParseError:
            continue
        root = tree.getroot()
        for section, key in [("abstract", "abstract"), ("actions", "actions"), ("decisions", "decisions"), ("problems", "problems")]:
            for node in root.iter(section):
                for sent in node.iter("sentence"):
                    t = (sent.text or "").strip()
                    if t and t != "NA." and t not in out[key]:
                        out[key].append(t)
    return out


def build_manifests(
    annotations_root: Path,
    audio_map: dict[str, Path],
    out_dir: Path,
    transcripts_dir: Optional[Path] = None,
    clip_duration: Optional[float] = None,
) -> None:
    """Create the three manifest JSON files.

    audio_map maps meeting id (e.g. "ES2004a") -> audio file path.
    transcripts_dir optionally supplies precomputed transcript JSON paths for
    summary/actions manifests (so run_rouge / run_actions can skip ASR).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    asr_rows, summary_rows, actions_rows = [], [], []

    for meeting, audio_path in audio_map.items():
        words = collect_meeting_words(annotations_root, meeting)
        if not words:
            print(f"[ami] no words for {meeting}; skipping")
            continue
        ref = reference_transcript(words, max_time=clip_duration)
        summaries = parse_abstract_summary(annotations_root, meeting)

        asr_rows.append({"audio": str(audio_path), "reference": ref, "lang": "en", "meeting": meeting})

        transcript_json = None
        if transcripts_dir is not None:
            candidate = transcripts_dir / f"{meeting}.transcript.json"
            if candidate.exists():
                transcript_json = str(candidate)

        if summaries["abstract"]:
            summary_rows.append(
                {
                    "meeting": meeting,
                    "transcript_json": transcript_json,
                    "reference_summary": " ".join(summaries["abstract"]),
                }
            )

        if summaries["actions"]:
            actions_rows.append(
                {
                    "meeting": meeting,
                    "transcript_json": transcript_json,
                    "gold_items": [{"task": a, "assignee": ""} for a in summaries["actions"]],
                }
            )

    (out_dir / "ami_asr_manifest.json").write_text(json.dumps(asr_rows, indent=2, ensure_ascii=False))
    (out_dir / "ami_summary_manifest.json").write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False))
    (out_dir / "ami_actions_manifest.json").write_text(json.dumps(actions_rows, indent=2, ensure_ascii=False))
    print(f"[ami] wrote {len(asr_rows)} asr, {len(summary_rows)} summary, {len(actions_rows)} actions rows")


def _main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--annotations-root", type=Path, default=Path("data/ami-annotations/ami_manual"))
    p.add_argument("--out-dir", type=Path, default=Path("eval/data"))
    p.add_argument("--transcripts-dir", type=Path, default=None)
    p.add_argument(
        "--audio-map",
        type=str,
        default="ES2004a:data/ami/ES2004a.Mix-Headset.wav",
        help="Comma-separated meeting:audio pairs.",
    )
    p.add_argument("--clip-duration", type=float, default=None, help="Truncate reference to first N seconds.")
    args = p.parse_args()

    audio_map: dict[str, Path] = {}
    for pair in args.audio_map.split(","):
        meeting, path = pair.split(":", 1)
        audio_map[meeting.strip()] = Path(path.strip())

    build_manifests(args.annotations_root, audio_map, args.out_dir, args.transcripts_dir, args.clip_duration)


if __name__ == "__main__":
    _main()
