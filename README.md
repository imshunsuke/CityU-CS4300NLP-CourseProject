# Smart Meeting Assistant

CS4300 NLP group project · Topic 5.
A bilingual (Chinese–English) meeting assistant that turns a meeting audio into six structured views: speaker-labeled transcript, summary, translation, action items, sentiment/engagement timeline, and retrieval-augmented Q&A — plus a **live microphone mode** for real-time transcription + translation.

## Features

| Module | Main system | Baseline | Metric(s) |
|---|---|---|---|
| **M1** ASR + diarization | WhisperX (Whisper large-v3 + pyannote 3.1) | — | WER / CER, DER |
| **M2** Summarization | GPT-4o-mini (structured JSON: key_points / decisions / follow_ups) | BART-large-cnn | ROUGE-1/2/L |
| **M3** Translation | GPT-4o-mini (context-aware) | NLLB-200-600M | sacreBLEU |
| **M4** Action items | GPT-4o-mini function calling (speaker-constrained assignee enum) | — | P / R / F1 |
| **M5** Sentiment + engagement | GPT-4o-mini per-utterance valence/arousal/emotion | — | MELD 7-way accuracy |
| **Q&A** (bonus) | Local sentence-transformers retrieval → GPT-4o-mini | — | qualitative |
| **Live mode** | faster-whisper `base` + async GPT-4o-mini translation | — | — |
| **LLM-as-Judge** | GPT-4o rating Likert 1–5 faithful/coverage/specificity | — | (novelty) |

---

## 1. Prerequisites

- **macOS** or Linux (tested on macOS Apple Silicon)
- **miniforge3 / miniconda / anaconda** (we use conda because ffmpeg is a system dep)
- **Git**
- **OpenAI API key** (any OpenAI-compatible endpoint; we use a proxy — see `.env`)
- **HuggingFace account + token** (for pyannote diarization, gated models — one-time manual accept needed)

## 2. One-time setup

```bash
# a. Clone
git clone git@github.com:imshunsuke/CityU-CS4300NLP-CourseProject.git smart-meeting-assistant
cd smart-meeting-assistant

# b. Create conda env (python 3.10 + ffmpeg via conda-forge)
conda create -n sma python=3.10 ffmpeg -c conda-forge -y
conda activate sma

# c. Install Python deps (first install is ~10 min: pulls torch + whisperx + pyannote)
pip install -r requirements.txt

# d. Set up secrets
cp .env.example .env     # then edit .env and fill in:
#   OPENAI_API_KEY=sk-...
#   OPENAI_BASE_URL=https://api.openai.com/v1   (or a proxy URL)
#   HF_TOKEN=hf_...
```

### 2.1. Accept HuggingFace gated models (critical!)

pyannote speaker-diarization models are **gated**. Log in to HuggingFace (the account whose token you put in `.env`) and click "Agree and access repository" on each page:

1. https://hf.co/pyannote/speaker-diarization-3.1
2. https://hf.co/pyannote/segmentation-3.0
3. https://hf.co/pyannote/speaker-diarization-community-1

One-time operation — WhisperX will silently fail without it.

## 3. Running the app

### 3.1. Gradio UI (main interface)

```bash
conda activate sma
python app.py
# Opens http://localhost:7860
```

The UI has two top-level tabs:

#### (a) **Batch analysis** tab — upload an audio file, get all 6 views

1. Drag an audio file into the upload widget
2. Pick translation target (`zh` / `en`)
3. Click **Process meeting**
4. Explore sub-tabs: Transcript · Summary · Translation · Action items · Sentiment · Q&A

First-run on a new audio: ~90 seconds for a 2-min clip (WhisperX + all LLM calls). Results are cached on disk (sha256 of audio), so re-runs of the same file are instant.

#### (b) **Live microphone** tab — real-time S2T + translation

1. Pick translation target
2. Click the microphone button (browser will ask for permission)
3. Speak a sentence, pause
4. (Optional) click **Process now** to force-flush the buffer
5. Transcript appears on the left after ~1-2s; translation fills in ~1-2s later (async)
6. Click **Clear transcript** to reset

Uses `faster-whisper base` (145 MB, auto-downloaded on first use). Buffer threshold is 1 s.

### 3.2. CLI pipeline (scripted, non-UI)

```bash
# Process an audio file and save analysis JSON
python -m src.pipeline --audio data/ami/ES2004a_short.wav --target-lang zh
# Writes eval/results/pipeline_run.json
```

## 4. Data

All datasets are public and free. Place under `data/` (git-ignored).

### 4.1. AMI Meeting Corpus (English)

```bash
mkdir -p data/ami data/ami-annotations
# Audio (one meeting, ~32 MB)
curl -L -o data/ami/ES2004a.Mix-Headset.wav \
  https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2004a/audio/ES2004a.Mix-Headset.wav

# Manual annotations (transcripts + abstract summaries + actions, 22 MB)
cd data/ami-annotations
curl -L -o ami_public_manual_1.6.2.zip \
  https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip
unzip -q ami_public_manual_1.6.2.zip -d ami_manual
cd ../..

# Make a short 2-min clip for quick testing
ffmpeg -y -i data/ami/ES2004a.Mix-Headset.wav -t 120 -c copy data/ami/ES2004a_short.wav
```

### 4.2. AliMeeting / M2MeT (Mandarin)

```bash
mkdir -p data/alimeeting && cd data/alimeeting

# Eval partition (~3.5 GB). Aliyun CDN is fast in Asia.
curl -L -O https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Eval_Ali.tar.gz

# Extract just one meeting (saves disk)
tar -xzf Eval_Ali.tar.gz \
  "Eval_Ali/Eval_Ali_far/audio_dir/R8001_M8004_MS801.wav" \
  "Eval_Ali/Eval_Ali_far/textgrid_dir/R8001_M8004.TextGrid"

# Downmix 8-channel array to mono 16 kHz + make a 2-min clip
ffmpeg -y -i Eval_Ali/Eval_Ali_far/audio_dir/R8001_M8004_MS801.wav -ac 1 -ar 16000 R8001_M8004_mono.wav
ffmpeg -y -i R8001_M8004_mono.wav -t 120 -c copy R8001_M8004_short.wav

# (Optional) free up disk after extraction
rm Eval_Ali.tar.gz
cd ../..
```

### 4.3. MELD (sentiment ground truth)

```bash
mkdir -p data/meld && curl -L -o data/meld/dev_sent_emo.csv \
  https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/dev_sent_emo.csv
```

## 5. Running evaluations

```bash
# Build manifest files for AMI / AliMeeting
python -m eval.data_prep.ami \
  --audio-map "ES2004a:data/ami/ES2004a.Mix-Headset.wav"
python -m eval.data_prep.alimeeting \
  --textgrid data/alimeeting/Eval_Ali/Eval_Ali_far/textgrid_dir/R8001_M8004.TextGrid \
  --audio data/alimeeting/R8001_M8004_mono.wav \
  --meeting R8001_M8004 --out eval/data/ali_asr_manifest.json
python -m eval.data_prep.flores --n 30        # BLEU manifest (news_commentary EN-ZH)
python -m eval.data_prep.judge_manifest       # LLM-as-Judge manifest

# Run each evaluator (all results go to eval/results/*.csv)
python -m eval.run_wer         --manifest eval/data/ami_asr_manifest.json --out eval/results/wer_ami.csv
python -m eval.run_wer         --manifest eval/data/ali_asr_manifest.json --out eval/results/cer_ali.csv
python -m eval.run_rouge       --out eval/results/rouge_gpt.csv            # GPT-4o-mini
python -m eval.run_rouge       --out eval/results/rouge_bart.csv --baseline  # BART
python -m eval.run_bleu        --out eval/results/bleu_gpt.csv  --system llm
python -m eval.run_bleu        --out eval/results/bleu_nllb.csv --system nllb
python -m eval.run_actions_f1  --out eval/results/actions_f1.csv
python -m eval.run_sentiment   --n 150 --out eval/results/sentiment_meld.csv
python -m eval.llm_judge       --out eval/results/llm_judge.csv
```

### Our first numbers (for reference)

| Metric | Our (GPT-4o-mini) | Baseline | Δ |
|---|---|---|---|
| AMI 30-min WER | 0.407 | — | — |
| AliMeeting CER (2-min clip) | 0.325 | — | — |
| ROUGE-1 | **0.393** | BART 0.163 | +0.230 |
| ROUGE-L | **0.180** | BART 0.116 | +0.064 |
| sacreBLEU (60 EN↔ZH) | **22.18** | NLLB 16.29 | +5.89 |
| Actions F1 | 0.40 (P=0.25 R=1.0) | — | — |
| MELD 7-way accuracy | 0.58 / macro-F1 0.52 | — | — |
| LLM-judge summary (f/c/s) | 2 / **3** / **3** | BART **3** / 2 / 2 | — |

## 6. Repository layout

```
src/                 Core modules
  schemas.py         pydantic contract shared by everything
  llm_client.py      OpenAI wrapper + diskcache + retry
  asr.py             WhisperX ASR + pyannote diarization
  summarize.py       M2 (GPT + BART/mT5)
  translate.py       M3 (GPT + NLLB)
  actions.py         M4 (function-calling schema)
  sentiment.py       M5 (per-utterance scoring + plotly timeline)
  qa.py              Bonus RAG (local sentence-transformers)
  live.py            Live microphone streaming (faster-whisper base + async translate)
  pipeline.py        CLI orchestrator
app.py               Gradio UI (Batch + Live tabs)
config.yaml          Model names + hyperparameters
eval/
  data_prep/         Manifest builders (AMI, AliMeeting, FLORES/news_commentary, judge)
  run_*.py           One script per metric
  llm_judge.py       LLM-as-Judge evaluator
  results/           Generated CSVs
data/                (git-ignored) audio + downloaded corpora
report/              LaTeX report source (main.tex + refs.bib)
slides/              Presentation deck
```

## 7. Known gotchas (we hit all of these — save yourself some time)

- **conda run vs conda activate**: if you call `python` directly (not via `conda activate`), `ffmpeg` won't be on PATH and WhisperX fails. Fix: either `conda activate sma` first, or wrap with `conda run -n sma python ...`.
- **wav2vec2 alignment download truncates** (`PytorchStreamReader failed reading zip archive`). torch.hub's downloader is flaky. Fix: manually pull with curl:
  ```bash
  curl -L --retry 5 -C - -o ~/.cache/torch/hub/checkpoints/wav2vec2_fairseq_base_ls960_asr_ls960.pth \
    https://download.pytorch.org/torchaudio/models/wav2vec2_fairseq_base_ls960_asr_ls960.pth
  ```
- **pyannote "Cannot access gated repo"**: you forgot step 2.1. Accept the three models on HF.
- **`sentence-transformers` 5.x crash** (`Could not load libtorchcodec`): pin to `<4` in requirements.txt (5.x hard-imports torchcodec, which needs ffmpeg 4–7, but conda-forge ships 8 now).
- **OpenAI proxy does not support embeddings**: the zhizengzeng proxy returns 500 on `/v1/embeddings`. `src/qa.py` and `eval/run_actions_f1.py` therefore use local `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` instead of OpenAI embeddings — no config change needed.
- **Whisper ASR stays on CPU, everything else uses MPS**: `ctranslate2` (faster-whisper backend) does not support Apple Silicon GPU. We run Whisper on CPU, alignment + diarization on MPS. Don't try `device="mps"` on `whisperx.load_model()` — it will error.
- **Live mode "buffering… 4.5s / 5s" stuck**: the mic stops sending chunks when you pause. Either lower `CHUNK_SECONDS` in `src/live.py` (already at 1.0) or click **Process now** to flush.

## 8. Environment details

- Python 3.10
- PyTorch 2.8 (CPU for ctranslate2, MPS for everything else on Apple Silicon)
- WhisperX 3.8.5 (requires the `.diarize.DiarizationPipeline` submodule + `token=...` kwarg since 3.8)
- pyannote.audio 4.0.x
- Gradio 6.x
- `sentence-transformers` pinned to `>=3,<4`
- OpenAI SDK 2.x

## 9. Team

CS4300 2025-2026 Q2, Department of Computer Science, City University of Hong Kong.
