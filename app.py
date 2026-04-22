from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import gradio as gr
import pandas as pd

from src.live import MicStream
from src.llm_client import LLMClient
from src.pipeline import build_qa, run
from src.qa import MeetingQA
from src.schemas import Language, MeetingAnalysis
from src.sentiment import make_timeline_figure


SPEAKER_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


def _color_for(speaker: str, mapping: dict[str, str]) -> str:
    if speaker not in mapping:
        mapping[speaker] = SPEAKER_COLORS[len(mapping) % len(SPEAKER_COLORS)]
    return mapping[speaker]


def _format_transcript_html(analysis: MeetingAnalysis) -> str:
    colors: dict[str, str] = {}
    rows = []
    for u in analysis.transcript.utterances:
        c = _color_for(u.speaker, colors)
        ts = f"{u.start:6.1f}s"
        rows.append(
            f'<div style="margin:4px 0"><span style="color:{c};font-weight:bold">'
            f"[{u.speaker}]</span> <span style='color:#888'>{ts}</span> {u.text}</div>"
        )
    return "<div style='max-height:500px;overflow-y:auto;font-family:monospace'>" + "".join(rows) + "</div>"


def _format_summary_md(analysis: MeetingAnalysis) -> str:
    s = analysis.summary

    def _bullets(title: str, items: list[str]) -> str:
        if not items:
            return f"### {title}\n_(none)_\n"
        return f"### {title}\n" + "\n".join(f"- {x}" for x in items) + "\n"

    return (
        _bullets("Key points", s.key_points)
        + _bullets("Decisions", s.decisions)
        + _bullets("Follow-ups", s.follow_ups)
    )


def _format_translation_df(analysis: MeetingAnalysis) -> pd.DataFrame:
    if analysis.translation is None:
        return pd.DataFrame(columns=["Speaker", "Source", "Target"])
    rows = []
    for u, pair in zip(analysis.transcript.utterances, analysis.translation.pairs):
        rows.append({"Speaker": u.speaker, "Source": pair.source, "Target": pair.target})
    return pd.DataFrame(rows)


def _format_actions_df(analysis: MeetingAnalysis) -> pd.DataFrame:
    rows = []
    for a in analysis.action_items.items:
        rows.append(
            {
                "Task": a.task,
                "Assignee": a.assignee,
                "Due": a.due or "",
                "Priority": a.priority,
                "Evidence": a.evidence_span,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["Task", "Assignee", "Due", "Priority", "Evidence"])
    return pd.DataFrame(rows)


def _format_sentiment_summary(analysis: MeetingAnalysis) -> str:
    r = analysis.sentiment
    if not r.points:
        return "_No sentiment data._"
    speaking = "\n".join(f"- {k}: {v:.1f}s" for k, v in r.speaking_time.items())
    top = "\n".join(
        f"- t={r.points[i].t:.1f}s [{r.points[i].speaker}] "
        f"{r.points[i].emotion} (val={r.points[i].valence:+.2f}, aro={r.points[i].arousal:.2f})"
        for i in r.top_moments
    )
    return (
        f"**Overall valence:** {r.overall_valence:+.2f}  \n"
        f"**Overall arousal:** {r.overall_arousal:.2f}  \n"
        f"**Engagement entropy:** {r.engagement_entropy:.3f} (higher = more balanced)  \n\n"
        f"### Speaking time\n{speaking}\n\n### Top moments\n{top}"
    )


def process_meeting(audio_file, target_lang: Language, state):
    if audio_file is None:
        gr.Warning("Please upload an audio file first.")
        return state, "", "", pd.DataFrame(), pd.DataFrame(), None, ""

    client = LLMClient()
    analysis = run(audio_path=Path(audio_file), target_lang=target_lang, client=client)
    qa = build_qa(analysis.transcript, client)
    new_state: dict[str, Any] = {"analysis": analysis, "qa": qa}

    return (
        new_state,
        _format_transcript_html(analysis),
        _format_summary_md(analysis),
        _format_translation_df(analysis),
        _format_actions_df(analysis),
        make_timeline_figure(analysis.sentiment),
        _format_sentiment_summary(analysis),
    )


def answer_question(question: str, state):
    if not state or "qa" not in state:
        return "_Please process a meeting first._"
    if not question.strip():
        return ""
    qa: MeetingQA = state["qa"]
    result = qa.ask(question)
    cites = ", ".join(f"[{i}]" for i in result.cited_utterance_indices)
    return f"{result.answer}\n\n**Cited utterances:** {cites}"


LIVE_COLORS = SPEAKER_COLORS


def _render_live_transcript(items: list[dict]) -> str:
    colors: dict[str, str] = {}
    rows = []
    for it in items:
        c = _color_for(it["speaker"], colors)
        rows.append(
            f'<div style="margin:4px 0"><span style="color:{c};font-weight:bold">'
            f"[{it['speaker']}]</span> <span style='color:#888'>{it['start']:6.1f}s</span> {it['text']}</div>"
        )
    return (
        "<div style='max-height:500px;overflow-y:auto;font-family:monospace;"
        "padding:8px;border:1px solid #eee;border-radius:6px'>"
        + "".join(rows) + "</div>"
    )


def _render_live_translation(items: list[dict]) -> str:
    rows = []
    for it in items:
        rows.append(
            f"<div style='margin:6px 0;padding:6px;background:#f6f8fa;border-radius:4px'>"
            f"<div style='color:#888;font-size:11px'>{it['source_lang']} → {it['target_lang']}</div>"
            f"<div style='font-family:monospace;font-size:12px;color:#444'>{it['source']}</div>"
            f"<div style='margin-top:4px'>{it['target']}</div></div>"
        )
    return (
        "<div style='max-height:500px;overflow-y:auto;padding:8px;"
        "border:1px solid #eee;border-radius:6px'>"
        + "".join(rows) + "</div>"
    )


def live_init(target_lang: Language):
    return MicStream(target_lang=target_lang)


def live_reset(stream: MicStream, target_lang: Language):
    if stream is None:
        stream = MicStream(target_lang=target_lang)
    else:
        stream.target_lang = target_lang
        stream.reset()
    return stream, f"t=0.0s  |  Cleared.", "", ""


def live_on_stream(new_chunk, stream: MicStream, target_lang: Language):
    if new_chunk is None:
        return stream, f"Waiting for microphone…", "", ""
    if stream is None:
        stream = MicStream(target_lang=target_lang)
    else:
        stream.target_lang = target_lang
    sr, audio = new_chunk
    snap = stream.push(sr, audio)
    return (
        stream,
        f"t={snap['elapsed_audio_s']:.1f}s  |  {snap['status']}",
        _render_live_transcript(snap["transcript"]),
        _render_live_translation(snap["translation"]),
    )


def live_force_flush(stream: MicStream, target_lang: Language):
    if stream is None:
        stream = MicStream(target_lang=target_lang)
    snap = stream.force_flush()
    return (
        stream,
        f"t={snap['elapsed_audio_s']:.1f}s  |  {snap['status']}",
        _render_live_transcript(snap["transcript"]),
        _render_live_translation(snap["translation"]),
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Smart Meeting Assistant") as demo:
        gr.Markdown("# Smart Meeting Assistant")

        with gr.Tabs():
            # ===== Batch Analysis Tab =====
            with gr.Tab("Batch analysis"):
                gr.Markdown("Upload a meeting audio file, then explore the five analysis modules.")
                state = gr.State({})

                with gr.Row():
                    audio = gr.Audio(type="filepath", label="Meeting audio (.wav/.mp3)")
                    target = gr.Dropdown(choices=["zh", "en"], value="zh", label="Translation target language")
                    run_btn = gr.Button("Process meeting", variant="primary")

                with gr.Tabs():
                    with gr.Tab("Transcript"):
                        transcript_html = gr.HTML()
                    with gr.Tab("Summary"):
                        summary_md = gr.Markdown()
                    with gr.Tab("Translation"):
                        translation_df = gr.Dataframe(wrap=True)
                    with gr.Tab("Action items"):
                        actions_df = gr.Dataframe(wrap=True)
                    with gr.Tab("Sentiment"):
                        sentiment_plot = gr.Plot()
                        sentiment_md = gr.Markdown()
                    with gr.Tab("Q&A"):
                        question = gr.Textbox(
                            label="Ask about this meeting",
                            placeholder="e.g., What did Alice say about the budget?",
                        )
                        ask_btn = gr.Button("Ask")
                        answer = gr.Markdown()

                run_btn.click(
                    process_meeting,
                    inputs=[audio, target, state],
                    outputs=[state, transcript_html, summary_md, translation_df, actions_df, sentiment_plot, sentiment_md],
                )
                ask_btn.click(answer_question, inputs=[question, state], outputs=[answer])

            # ===== Live Microphone Tab =====
            with gr.Tab("Live microphone (S2T + Translation)"):
                gr.Markdown(
                    "Record from your microphone. Whisper runs on every "
                    f"~5-second buffer; each utterance is translated live.\n\n"
                    "**Tip:** Click the mic button to start; speak a full sentence, pause, repeat."
                )

                live_state = gr.State(None)

                with gr.Row():
                    live_target = gr.Dropdown(choices=["zh", "en"], value="zh", label="Translation target language")
                    live_flush_btn = gr.Button("Process now", variant="primary")
                    live_clear_btn = gr.Button("Clear transcript")

                live_mic = gr.Audio(
                    sources=["microphone"],
                    streaming=True,
                    type="numpy",
                    label="Microphone (click to record)",
                )

                live_status = gr.Textbox(label="Status", interactive=False, value="Ready. Click mic to record.")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Transcript (source)")
                        live_transcript = gr.HTML()
                    with gr.Column():
                        gr.Markdown("### Translation")
                        live_translation = gr.HTML()

                live_mic.stream(
                    live_on_stream,
                    inputs=[live_mic, live_state, live_target],
                    outputs=[live_state, live_status, live_transcript, live_translation],
                    stream_every=0.5,
                )
                live_flush_btn.click(
                    live_force_flush,
                    inputs=[live_state, live_target],
                    outputs=[live_state, live_status, live_transcript, live_translation],
                )
                live_clear_btn.click(
                    live_reset,
                    inputs=[live_state, live_target],
                    outputs=[live_state, live_status, live_transcript, live_translation],
                )

    return demo


if __name__ == "__main__":
    build_ui().launch(theme=gr.themes.Soft())
