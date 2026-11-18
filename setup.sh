#!/usr/bin/env bash
# bash setup.sh            -> .venv with faster-whisper (cutting)
# bash setup.sh --denoise  -> also .venv-denoise with DeepFilterNet 3 (needs Python 3.10 or 3.11)
set -euo pipefail
cd "$(dirname "$0")"
command -v ffmpeg >/dev/null || { echo "ffmpeg is required (brew install ffmpeg / apt install ffmpeg)"; exit 1; }
unset PYTHONPATH                      # a global PYTHONPATH can leak foreign packages into the venv
PY=${PYTHON:-python3}
"$PY" -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
echo "ok: .venv/bin/python clean_takes.py RAW.mp4 OUT.mp4"
if [ "${1:-}" = "--denoise" ]; then
  PYD=""
  for c in python3.11 python3.10; do command -v $c >/dev/null && { PYD=$c; break; }; done
  if command -v uv >/dev/null; then
    uv venv -q -p "${PYD:-3.11}" .venv-denoise
    PIP="uv pip install -q --python .venv-denoise/bin/python"
  else
    [ -n "$PYD" ] || { echo "DeepFilterNet needs Python 3.10/3.11 (or install uv)"; exit 1; }
    "$PYD" -m venv .venv-denoise; .venv-denoise/bin/pip install -q --upgrade pip
    PIP=".venv-denoise/bin/pip install -q"
  fi
  $PIP "numpy<2" torch==2.2.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
  $PIP -r requirements-denoise.txt
  echo "ok: .venv-denoise/bin/python denoise.py RAW.mp4 clean.wav"
fi
