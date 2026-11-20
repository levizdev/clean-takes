#!/usr/bin/env python3
"""denoise.py: AI noise removal for a voice track (DeepFilterNet 3) + gentle "podcast" polish.

    python denoise.py INPUT(.mp4/.wav/...) OUT.wav [--atten 32] [--no-polish]

Then cut with the clean audio:  python clean_takes.py RAW.mp4 OUT.mp4 --audio OUT.wav

- Runs on CPU, ~5 s per minute of audio after the model is downloaded (from GitHub, ~10 MB).
- The file is processed in 8 s chunks, each with 2 s of warm-up before and 1 s after that are
  discarded: on long files DeepFilterNet sometimes collapses into silence after ~25 s, and right
  after a (re)start it needs ~1 s to settle - without the warm-up it ate quiet words at chunk joints.
- --atten limits how much noise is removed, in dB. Unlimited removal also eats words when the
  microphone is far (camera mic at 1-2 m, pool/street/wind): 32 dB was the sweet spot on our
  test (noise 19 dB under the voice, words intact). With a lavalier you can go higher (--atten 100).
- Polish (default on): high-pass 90 Hz, low-pass 11 kHz, de-esser, soft compressor, +2.5 dB
  presence at 3.2 kHz, gentle gate in pauses, loudness -16 LUFS.
Needs the extra environment from `setup.sh --denoise` (Python 3.10/3.11, torch 2.2).
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

POLISH = ('highpass=f=90,lowpass=f=11000,deesser=i=0.4,'
          'acompressor=threshold=-20dB:ratio=2:attack=5:release=90:makeup=1.5,'
          'equalizer=f=3200:t=q:w=1.2:g=2.5,'
          'agate=threshold=0.03:ratio=3:range=0.06:attack=6:release=200:knee=3,'
          'loudnorm=I=-16:TP=-1.5')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input'); ap.add_argument('output')
    ap.add_argument('--atten', type=float, default=32.0, help='max noise reduction, dB (default 32)')
    ap.add_argument('--chunk', type=float, default=8.0)
    ap.add_argument('--no-polish', action='store_true')
    a = ap.parse_args()
    try:
        import torch
        from df.enhance import enhance, init_df
    except ImportError:
        sys.exit('DeepFilterNet is not installed: run  bash setup.sh --denoise  and use .venv-denoise/bin/python')
    sr = 48000
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', a.input, '-map', '0:a:0', '-ac', '1', '-ar', str(sr), '-f', 's16le', '-'],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    rms = np.sqrt(np.mean(x[np.abs(x) > 0.003] ** 2) + 1e-12)
    g = min(10 ** (-20 / 20) / rms, 30.0)          # quiet camera audio -> about -20 dBFS before the model
    x = np.clip(x * g, -1, 1)
    # chunks [s, e) with a 2 s warm-up before and 1 s after that are thrown away: the model needs
    # ~1 s to settle and goes soft at the very end of its buffer (that ate words on chunk joints)
    C, PRE, POST, XF, T = int(a.chunk * sr), int(2.0 * sr), int(1.0 * sr), int(0.25 * sr), len(x)
    y, w = np.zeros(T, np.float32), np.zeros(T, np.float32)
    s = 0
    while s < T:
        e = min(T, s + C)
        i0, i1 = max(0, s - PRE), min(T, e + XF + POST)
        model, st, _ = init_df(log_level='ERROR', post_filter=True)
        z = enhance(model, st, torch.from_numpy(x[i0:i1].copy())[None], atten_lim_db=a.atten).numpy()[0]
        ce = min(T, e + XF)
        z = z[s - i0:ce - i0]
        win = np.ones(len(z), np.float32)
        if s > 0:
            win[:XF] = np.linspace(0, 1, XF)
        if ce < T:
            win[-XF:] = np.linspace(1, 0, XF)
        y[s:ce] += z * win; w[s:ce] += win
        print(f'\r  {e / T * 100:5.1f}%', end='', flush=True)
        s = e
    print()
    y = y / np.maximum(w, 1e-6)
    with tempfile.TemporaryDirectory() as tmp:
        t = os.path.join(tmp, 'dn.raw')
        (np.clip(y, -1, 1) * 32767).astype(np.int16).tofile(t)
        af = [] if a.no_polish else ['-af', POLISH]
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 's16le', '-ar', str(sr), '-ac', '1', '-i', t, *af, '-ar', str(sr), a.output], check=True)
    print('done →', a.output)


if __name__ == '__main__':
    main()
