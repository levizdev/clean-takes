---
name: clean-takes
description: Cut a raw talking-head recording (one long take with retakes, stumbles, "ммм" and pauses) down to only the cleanly spoken phrases; optional AI noise removal. Use when the user filmed themselves and asks to cut out the bad takes, mistakes, pauses or noise.
---

# clean-takes

Raw video in, clean cut out: for every phrase the speaker repeated, the **last** full attempt is kept;
false starts, restarts inside a phrase, lone «а/э/мм», and pauses longer than 0.3 s are removed.
Code: https://github.com/levizdev/clean-takes

## Setup (once per machine/sandbox)
```bash
git clone https://github.com/levizdev/clean-takes ~/clean-takes && cd ~/clean-takes && bash setup.sh            # cutting
bash setup.sh --denoise     # optional: AI noise removal (needs Python 3.10/3.11 or uv)
```
- Needs ffmpeg and access to huggingface.co once (Whisper "small", ~500 MB, cached in `$HF_HOME`). Then `HF_HUB_OFFLINE=1` works.
- If pip fails with `Permission denied`, a global PYTHONPATH leaks in: `unset PYTHONPATH` (setup.sh does it).

## Run
1. **Plan first** (fast to review, nothing rendered):
   `.venv/bin/python clean_takes.py RAW.mp4 out/cut.mp4 --plan-only [--lang ru] [--prompt "имена, бренды"]`
   Read `out/cut_report.md`: every phrase with ✅ kept / ❌ dropped + reason. Show the user the dropped ones if anything looks doubtful.
2. **Fix decisions** if needed: edit `"keep"` in `out/cut_plan.json`, then render with `--plan out/cut_plan.json`.
3. **Render**: same command without `--plan-only` (or with `--plan …`). Output: `out/cut.mp4` (1080 on the short side, 30 fps, −14 LUFS), `_report.md` with a re-transcription of the result, `_edl.json`.
4. **Noise** (optional, before step 3): `.venv-denoise/bin/python denoise.py RAW.mp4 out/clean.wav` then add `--audio out/clean.wav` to the render.

## Speed / heavy files
- Whisper small on CPU: ~1 min per minute of audio. 4K HEVC from phones/DJI decodes slowly: ~4 s of CPU per 1 s of output at 1080p. Use `--jobs` (parallel pieces) and run in the background if the shell has a time limit.
- For a fast look, render a low-res proxy first: `--size 540x960 --crf 28`.

## Read the result honestly
- Detection works on Whisper words: a stumble Whisper "smooths over" (it writes the intended word) can survive. Ask the user to watch the cut and name seconds — then drop that phrase in the plan or re-cut.
- `⚠️ неразборчиво` in the report = average word confidence < 0.35: check those phrases by ear.
- Short one-word phrases followed by a sentence that starts with the same word are treated as false starts.
- Denoise: `--atten 32` is the default. Unlimited removal (`--atten 100`) eats words when the mic is far (camera mic at 1–2 m by a pool/street). With a lavalier it is safe to go higher.
