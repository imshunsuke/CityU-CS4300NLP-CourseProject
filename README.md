# Smart Meeting Assistant

CS4300 NLP group project · Topic 5.
A bilingual (Chinese–English) meeting assistant that turns a raw meeting audio into five structured views: speaker-labeled transcript, summary, translation, action items, and sentiment/engagement timeline — with a bonus retrieval-augmented Q&A over the meeting.

## Features
1. **ASR + diarization** — WhisperX (Whisper large-v3 + pyannote 3.1).
2. **Summarization** — GPT-4o-mini structured (key_points / decisions / follow_ups) vs BART / mT5 baseline.
3. **Translation** — GPT-4o-mini context-aware vs NLLB-200 baseline (ZH ↔ EN).
4. **Action items** — GPT-4o-mini function calling with speaker-constrained assignee enum.
5. **Sentiment & engagement** — per-utterance valence/arousal/emotion + speaking-time entropy + top emotional moments timeline.
6. **Bonus · Q&A** — embedding retrieval over utterances → grounded GPT-4o-mini answer.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # then fill OPENAI_API_KEY and HF_TOKEN

# Launch the Gradio UI
python app.py

# Or run the CLI pipeline
python -m src.pipeline --audio data/samples/demo.wav --target-lang zh
```

## Datasets
- **AMI Meeting Corpus** (EN, multi-speaker scenario meetings)
- **AliMeeting / M2MeT** (ZH, ICASSP 2022 challenge)
- **MeetingBank** (EN, long-form summaries)
- **MELD** (sentiment calibration)

See the plan / report for access URLs and preprocessing scripts.

## Repository layout

```
src/          Core modules (schemas, llm_client, asr, summarize, translate, actions, sentiment, qa, pipeline)
eval/         Metric scripts (WER, ROUGE, BLEU, action-F1, MELD sentiment, LLM-as-judge)
app.py        Gradio entrypoint
config.yaml   Model names and hyperparameters
data/samples/ Committed demo audio (small)
report/       LaTeX report source
slides/       Presentation deck
```

## Team

CS4300 2025-2026 Q2, CityU HK. Solo developer (see plan for 2-person division of labor, adapted).
