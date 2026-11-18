#!/usr/bin/env bash
# Edge cases of the whole pipeline (needs ffmpeg + the venv with faster-whisper):  bash tests/test_edge.sh
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
pass=0; fail=0
check(){ if eval "$2"; then echo "ok   $1"; pass=$((pass+1)); else echo "FAIL $1"; fail=$((fail+1)); fi; }
ffmpeg -v error -f lavfi -i color=c=gray:s=320x568:d=3 -c:v libx264 "$T/noaudio.mp4"
ffmpeg -v error -f lavfi -i color=c=gray:s=320x568:d=5 -f lavfi -i anullsrc=r=48000:cl=stereo -t 5 -c:v libx264 -c:a aac "$T/silence.mp4"
$PY clean_takes.py "$T/nope.mp4" "$T/o1.mp4" >/dev/null 2>&1;                check "missing file -> error"   "[ \$? -ne 0 ]"
$PY clean_takes.py "$T/noaudio.mp4" "$T/o2.mp4" 2>&1 | grep -q "no audio";     check "no audio -> clear error" "[ \$? -eq 0 ]"
$PY clean_takes.py "$T/silence.mp4" "$T/o3.mp4" --plan-only >/dev/null 2>&1;  check "silence -> no crash, clear message" "[ \$? -ne 0 ] && [ -f \"$T/o3_report.md\" ]"
echo "passed $pass, failed $fail"; [ $fail -eq 0 ]
