#!/usr/bin/env bash
# Smart Meeting Assistant — one-click classroom demo launcher.
#
# Double-click this file in Finder. macOS opens it in Terminal.app, the
# script activates the `sma` conda env, starts the Gradio app, and opens
# http://localhost:7860 in your default browser as soon as the server is up.
#
# To stop: hit Ctrl+C in the terminal window.

set -uo pipefail

# ---------------------------------------------------------------------------
# 1. cd into the script's own directory (works from any double-click location)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "📂 Project: $SCRIPT_DIR"
echo

# ---------------------------------------------------------------------------
# 2. Locate and source conda (try common install paths, then `conda info`)
# ---------------------------------------------------------------------------
CONDA_SH=""
for candidate in \
  "$HOME/miniconda3/etc/profile.d/conda.sh" \
  "$HOME/anaconda3/etc/profile.d/conda.sh" \
  "$HOME/miniforge3/etc/profile.d/conda.sh" \
  "/opt/miniconda3/etc/profile.d/conda.sh" \
  "/opt/anaconda3/etc/profile.d/conda.sh" \
  "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh" \
  "/usr/local/Caskroom/miniconda/base/etc/profile.d/conda.sh"; do
  if [ -f "$candidate" ]; then
    CONDA_SH="$candidate"
    break
  fi
done

if [ -z "$CONDA_SH" ] && command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ] \
    && CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"
fi

if [ -z "$CONDA_SH" ]; then
  echo "❌ Couldn't find conda."
  echo "   Install miniconda first, or edit CONDA_SH near the top of this script."
  echo
  read -r -p "Press Enter to close this window..."
  exit 1
fi

# shellcheck source=/dev/null
source "$CONDA_SH"

# ---------------------------------------------------------------------------
# 3. Activate the `sma` env
# ---------------------------------------------------------------------------
if ! conda activate sma; then
  echo
  echo "❌ Conda env 'sma' not found. Create it with:"
  echo "     conda create -n sma python=3.10 ffmpeg -c conda-forge -y"
  echo "     conda activate sma && pip install -r requirements.txt"
  echo
  read -r -p "Press Enter to close this window..."
  exit 1
fi

echo "✅ env: sma  ($(python --version 2>&1))"

# ---------------------------------------------------------------------------
# 4. Sanity checks (warn-only, never fatal — let the demo try to run)
# ---------------------------------------------------------------------------
if [ ! -f .env ]; then
  echo "⚠️  No .env file found."
  echo "   Translation / summary / sentiment / Q&A need OPENAI_API_KEY."
  echo "   Fix: cp .env.example .env   # then edit with your key"
  echo
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "⚠️  ffmpeg not on PATH. WhisperX will fail to load audio."
  echo "   Fix: conda install -n sma -c conda-forge ffmpeg"
  echo
fi

# ---------------------------------------------------------------------------
# 5. Open Finder at the demo_files/ folder so you can drag clips into Gradio
# ---------------------------------------------------------------------------
[ -d demo_files ] && open demo_files

# ---------------------------------------------------------------------------
# 6. Auto-open the browser as soon as Gradio is up (background task)
# ---------------------------------------------------------------------------
(
  for _ in $(seq 1 60); do
    if curl -sf http://localhost:7860 >/dev/null 2>&1; then
      open "http://localhost:7860"
      exit 0
    fi
    sleep 1
  done
) &
OPEN_PID=$!

# ---------------------------------------------------------------------------
# 7. Launch the Gradio app
# ---------------------------------------------------------------------------
echo
echo "🚀 Starting Smart Meeting Assistant…"
echo "   URL: http://localhost:7860  (will auto-open when ready, ~10–30 s)"
echo "   Stop: Ctrl+C in this window"
echo

python app.py
EXIT_CODE=$?

# Clean up the background browser-opener if Gradio died before it fired
kill "$OPEN_PID" 2>/dev/null || true

echo
if [ $EXIT_CODE -eq 0 ] || [ $EXIT_CODE -eq 130 ]; then
  echo "👋 App stopped cleanly."
else
  echo "❌ App exited with code $EXIT_CODE"
  echo "   Scroll up to see the error."
fi
echo
read -r -p "Press Enter to close this window..."
