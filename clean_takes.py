#!/usr/bin/env python3
"""clean-takes: a raw talking-head recording with retakes -> a cut that keeps only the clean takes.

    python clean_takes.py RAW.mp4 OUT.mp4 [--lang ru] [--plan-only] [--plan plan.json] [--audio clean.wav]

How it works
 1. Whisper (faster-whisper) transcribes the audio with a timestamp for every word.
 2. Words are grouped into phrases (split on sentence punctuation and on pauses).
 3. A phrase is dropped as a failed take when one of the next phrases (within --window seconds)
    starts the same way or repeats most of it -- the LAST attempt wins. Phrases made only of
    fillers ("э", "мм", a lone "а"...) are dropped too.
 4. The kept phrases are cut on the audio envelope (not on Whisper's rough word times), pauses
    longer than --pause inside a phrase are removed, every piece is snapped to the frame grid.
 5. Video pieces are encoded frame-exact, audio is cut sample-exact with 10 ms fades, then
    everything is joined and loudness-normalised. The result is transcribed again as a check.

Outputs next to OUT: OUT_report.md (every phrase: kept / dropped + why), OUT_plan.json
(edit "keep" there and re-render with --plan), OUT_edl.json (final pieces in source seconds).
"""
import argparse
import concurrent.futures as cf
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import wave

import numpy as np

FILLERS = {'э', 'ээ', 'эээ', 'эм', 'эмм', 'мм', 'ммм', 'м', 'а', 'аа', 'ааа', 'ну', 'хм', 'гм',
           'uh', 'um', 'umm', 'eh', 'er', 'erm', 'hmm', 'mm', 'ah'}
SR_A = 16000      # analysis rate
HOP = 0.01        # 10 ms envelope


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def ffprobe(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-print_format', 'json', '-show_streams', '-show_format', path],
                         capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    v = next((s for s in j['streams'] if s['codec_type'] == 'video'), None)
    a = next((s for s in j['streams'] if s['codec_type'] == 'audio'), None)
    return j, v, a


def load_pcm(path, sr, ch=1):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-map', '0:a:0', '-ac', str(ch), '-ar', str(sr),
                          '-f', 's16le', '-'], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32).reshape(-1, ch) / 32768


def norm_tokens(text):
    t = text.lower().replace('ё', 'е')
    t = re.sub(r'[^\w\s-]', ' ', t)
    return [w for w in re.split(r'[\s-]+', t) if w]


# ----------------------------------------------------------------------------- ASR
def transcribe(a16, lang, model_name, prompt=None, threads=0):
    from faster_whisper import WhisperModel
    m = WhisperModel(model_name, device='cpu', compute_type='int8', cpu_threads=threads)
    segs, _ = m.transcribe(a16, language=None if lang == 'auto' else lang, word_timestamps=True, beam_size=5,
                           condition_on_previous_text=False, vad_filter=False, initial_prompt=prompt)
    words = []
    for s in segs:
        for w in (s.words or []):
            if w.word.strip():
                words.append({'s': float(w.start), 'e': float(w.end), 'w': w.word.strip(), 'p': round(float(w.probability), 3)})
    return words


# ----------------------------------------------------------------------------- phrases & retakes
def to_phrases(words, gap):
    ph, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        end_sent = bool(re.search(r'[.?!…]["»]?$', w['w']))
        if nxt is None or end_sent or nxt['s'] - w['e'] > gap:
            ph.append({'s': cur[0]['s'], 'e': cur[-1]['e'], 'words': cur, 'text': ' '.join(x['w'] for x in cur)})
            cur = []
    for k, p in enumerate(ph):
        p['id'] = k
        p['tok'] = norm_tokens(p['text'])
        p['p'] = round(float(np.mean([x['p'] for x in p['words']])), 3)
    return ph


def _mk(words, why=''):
    return {'s': words[0]['s'], 'e': words[-1]['e'], 'words': words, 'text': ' '.join(x['w'] for x in words), 'restart': why}


def split_restarts(ph):
    """inside one phrase: 'и вот обещанная мерзость и вот обещанная мразь …' -> drop the first attempt.
    A restart = the same 2+ words appear again within 12 words (3+ if the gap is longer than 4 words)."""
    out = []
    for p in ph:
        ws = p['words']
        wn = [''.join(norm_tokens(w['w'])) for w in ws]
        cut = None
        for i in range(len(ws)):
            if not wn[i]:
                continue
            best = None
            for j in range(i + 1, min(len(ws), i + 13)):
                k = 0
                while j + k < len(ws) and i + k < j and wn[i + k] == wn[j + k] and wn[i + k]:
                    k += 1
                if k >= 2 and (k >= 3 or j - i <= 4):
                    best = j
            if best is not None:
                cut = (i, best)
                break
        if cut is None:
            out.append(p)
            continue
        i, j = cut
        if i > 0:
            out.append(_mk(ws[:i]))
        out.append(_mk(ws[i:j], 'restart'))
        out.extend(split_restarts([_mk(ws[j:])]))
    for k, p in enumerate(out):
        p['id'] = k
        p['tok'] = norm_tokens(p['text'])
        p['p'] = round(float(np.mean([x['p'] for x in p['words']])), 3)
    return out


def is_filler(word):
    t = norm_tokens(word)
    return bool(t) and all(x in FILLERS for x in t)


def contained(a, b):
    """share of tokens of a that appear in b in the same order"""
    if not a:
        return 0.0
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return sum(m.size for m in sm.get_matching_blocks()) / len(a)


def prefix_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def mark_retakes(ph, window):
    for p in ph:
        p['keep'], p['why'] = True, ''
        if p.get('restart'):
            p['keep'], p['why'] = False, f'restart:{p["id"] + 1}'
        elif p['tok'] and all(t in FILLERS for t in p['tok']):
            p['keep'], p['why'] = False, 'filler'
    for i, p in enumerate(ph):
        if not p['keep'] or not p['tok']:
            continue
        for q in ph[i + 1:]:
            if q['s'] - p['e'] > window:
                break
            if not q['tok'] or q['why'] == 'filler':
                continue
            a, b = p['tok'], q['tok']
            pre, con = prefix_len(a, b), contained(a, b)
            ratio = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
            short_false_start = len(a) <= 3 and pre >= 1 and len(b) > len(a)
            if (pre >= 2 and con >= 0.5) or ratio >= 0.6 or con >= 0.8 or short_false_start:
                p['keep'], p['why'] = False, f'retake:{q["id"]}'
                break
    return ph


# ----------------------------------------------------------------------------- envelope
def envelope(a16):
    h = int(SR_A * HOP)
    n = len(a16) // h
    r = np.sqrt(np.mean(a16[:n * h].reshape(n, h) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(r)
    return np.convolve(db, np.ones(3) / 3, 'same')


def speech_threshold(db):
    floor, loud = np.percentile(db, 10), np.percentile(db, 90)
    return float(floor + 0.35 * (loud - floor))


def refine(db, th, s0, e0):
    """move rough boundaries to where the voice really starts / ends (bounded moves)"""
    i, j = int(s0 / HOP), int(e0 / HOP)
    i, j = max(0, min(i, len(db) - 1)), max(1, min(j, len(db)))
    a = i
    while a < j and db[a] < th:
        a += 1
    k = 0
    while a > 0 and db[a - 1] >= th and k < 25:
        a -= 1; k += 1
    b = j
    while b > a and db[b - 1] < th:
        b -= 1
    k = 0
    while b < len(db) and db[b] >= th and k < 35:
        b += 1; k += 1
    s, e = a * HOP - 0.06, b * HOP + 0.10
    s = min(s, s0 + 0.02)
    e = max(min(e, e0 + 0.15), e0)
    return max(0.0, s), e


def voiced(db, th, s, e):
    return float(np.sum(db[int(s / HOP):int(e / HOP)] >= th) * HOP)


def split_pauses(db, th, s, e, pause, keep_before, keep_after, min_piece, blips, words=()):
    i0, i1 = int(s / HOP), int(e / HOP)
    q = db[i0:i1] < th
    pieces, cur, run_ = [], s, 0
    for n, quiet in enumerate(q):
        if quiet:
            run_ += 1
            continue
        if run_ * HOP >= pause:
            ps, pe = (i0 + n - run_) * HOP, (i0 + n) * HOP
            pieces.append([cur, ps + keep_after])
            cur = pe - keep_before
        run_ = 0
    pieces.append([cur, e])
    # a lone "а"/"мм"/breath before or after the phrase, cut off by a pause: < 0.25 s of voice -> drop
    def has_word(p):
        for w in words:
            if norm_tokens(w['w']) and all(t in FILLERS for t in norm_tokens(w['w'])):
                continue
            mid = (w['s'] + w['e']) / 2
            if p[0] <= mid <= p[1] or (p[0] <= w['s'] <= p[1] and w['e'] - w['s'] < 0.8):
                return True
        return False
    while len(pieces) > 1 and voiced(db, th, *pieces[0]) < 0.45 and not has_word(pieces[0]):
        blips.append(pieces.pop(0))
    while len(pieces) > 1 and voiced(db, th, *pieces[-1]) < 0.45 and not has_word(pieces[-1]):
        blips.append(pieces.pop())
    out = []
    for p in pieces:               # do not cut a pause that would leave a sliver in the middle
        if out and out[-1][1] - out[-1][0] < min_piece:
            out[-1][1] = p[1]
        else:
            out.append(p)
    if len(out) > 1 and out[-1][1] - out[-1][0] < min_piece:
        out[-2][1] = out[-1][1]; out.pop()
    return out


def build_edl(ph, db, th, a, fps, dur):
    kept = [p for p in ph if p['keep']]
    ranges = []
    for p in kept:
        ws = list(p['words'])
        # a drawn-out «а…» / «э…» / «ну…» at the edge of a phrase is a hesitation, not a word
        while len(ws) > 1 and is_filler(ws[0]['w']) and ws[0]['e'] - ws[0]['s'] >= 0.45:
            ws.pop(0)
        while len(ws) > 1 and is_filler(ws[-1]['w']) and ws[-1]['e'] - ws[-1]['s'] >= 0.45:
            ws.pop()
        p['words_cut'] = ws
        s, e = refine(db, th, ws[0]['s'] - 0.08, ws[-1]['e'] + 0.06)
        ranges.append([s, min(e, dur), p['id']])
    # a dropped phrase must never leak into a neighbour's padding
    drops = [(p['words'][0]['s'], p['words'][-1]['e']) for p in ph if not p['keep']]
    for r in ranges:
        for ds, de in drops:
            if r[0] < de <= r[1] and ds < r[0] + 0.25:
                r[0] = de + 0.02
            if r[0] <= ds < r[1] and de > r[1] - 0.25:
                r[1] = ds - 0.02
    ranges.sort()
    merged = []
    for r in ranges:
        if merged and r[0] <= merged[-1][1] + 0.02:
            merged[-1][1] = max(merged[-1][1], r[1]); merged[-1][2].append(r[2])
        else:
            merged.append([r[0], r[1], [r[2]]])
    pieces, blips = [], []
    byid = {p['id']: p for p in ph}
    for s, e, ids in merged:
        ws = [w for i in ids for w in byid[i].get('words_cut', byid[i]['words'])]
        for ps, pe in split_pauses(db, th, s, e, a.pause, a.keep_before, a.keep_after, 0.45, blips, ws):
            pieces.append([ps, pe, ids])
    build_edl.blips = blips
    edl, prev_end = [], 0.0
    for s, e, ids in pieces:
        s = max(s, prev_end)
        n = int(round((e - s) * fps))
        if n < 2:
            continue
        e = s + n / fps
        edl.append({'s': round(s, 4), 'e': round(e, 4), 'frames': n, 'phrases': ids})
        prev_end = e
    return edl


# ----------------------------------------------------------------------------- render
def scale_filter(v, size):
    if size != 'keep':
        w, h = map(int, size.lower().split('x'))
        return f'scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,crop={w}:{h}'
    return "scale='if(gt(ih,iw),1080,-2)':'if(gt(ih,iw),-2,1080)':flags=lanczos"


def render_piece(src, s, n, fps, vf, crf, preset, outp):
    run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-ss', f'{s:.4f}', '-i', src, '-map', '0:v:0', '-an',
         '-vf', f'{vf},fps={fps},format=yuv420p', '-frames:v', str(n), '-c:v', 'libx264', '-preset', preset,
         '-crf', str(crf), '-video_track_timescale', '15360', outp])
    return outp


def render(a, src, edl, tmp, audio_src):
    fps = a.fps
    vf = scale_filter(None, a.size)
    files = [os.path.join(tmp, f'v_{i:03d}.mp4') for i in range(len(edl))]
    t0 = time.time()
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        futs = [ex.submit(render_piece, src, p['s'], p['frames'], fps, vf, a.crf, a.preset, f) for p, f in zip(edl, files)]
        for k, fu in enumerate(cf.as_completed(futs)):
            fu.result()
            print(f'\r  video pieces {k + 1}/{len(files)}  ({time.time() - t0:.0f}s)', end='', flush=True)
    print()
    # audio: sample exact, 10 ms fades at every joint
    sr = 48000
    au = load_pcm(audio_src, sr, 2)
    F = int(0.010 * sr)
    ramp = np.linspace(0, 1, F, dtype=np.float32)[:, None]
    parts = []
    for p in edl:
        i, L = int(round(p['s'] * sr)), int(round(p['frames'] * sr / fps))
        x = au[i:i + L].copy()
        if len(x) < L:
            x = np.vstack([x, np.zeros((L - len(x), 2), np.float32)])
        x[:F] *= ramp; x[-F:] *= ramp[::-1]
        parts.append(x)
    y = np.vstack(parts)
    wav = os.path.join(tmp, 'audio.wav')
    with wave.open(wav, 'wb') as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())
    lst = os.path.join(tmp, 'list.txt')
    with open(lst, 'w') as f:
        f.writelines(f"file '{os.path.abspath(x)}'\n" for x in files)
    vcat = os.path.join(tmp, 'video.mp4')
    run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', lst, '-c', 'copy', vcat])
    af = 'highpass=f=80'
    if a.loudnorm is not None:
        af += f',loudnorm=I={a.loudnorm}:TP=-1.5:LRA=11'
    run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', vcat, '-i', wav, '-map', '0:v', '-map', '1:a', '-c:v', 'copy',
         '-af', af, '-ar', '48000', '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', a.output])
    return len(y) / sr


# ----------------------------------------------------------------------------- report
def fmt(t):
    return f'{int(t // 60)}:{t % 60:05.2f}'


def reason(why):
    k, _, n = why.partition(':')
    return {'retake': f'неудачный дубль, оставлен #{n}', 'restart': f'оговорка, перезапуск в #{n}',
            'filler': 'мычание / слово-паразит', 'manual': 'убрана вручную'}.get(k, why)


def write_report(path, a, ph, edl, th, dur, out_dur, check_text):
    L = [f'# clean-takes: {os.path.basename(a.input)}', '',
         f'- Исходник: {fmt(dur)} · результат: {fmt(out_dur) if out_dur else "— (только план)"} · кусков: {len(edl)}',
         f'- Порог речи по громкости: {th:.1f} дБ · паузы длиннее {a.pause} с вырезаны', '',
         '| # | Время в исходнике | Фраза | Решение |', '|---|---|---|---|']
    for p in ph:
        flag = ' ⚠️ неразборчиво' if p['p'] < 0.35 else ''
        st = '✅ оставлена' if p['keep'] else '❌ ' + reason(p['why'])
        L.append(f'| {p["id"]} | {fmt(p["s"])}–{fmt(p["e"])} | {p["text"]}{flag} | {st} |')
    bl = getattr(build_edl, 'blips', [])
    if bl:
        L += ['', 'Вырезаны короткие звуки на краях фраз (мычание, «а», вдох): ' + ', '.join(f'{fmt(b[0])}–{fmt(b[1])}' for b in bl)]
    if check_text is not None:
        L += ['', '## Проверка: что слышно в готовом ролике', '', check_text]
    L += ['', 'Поменять решение: открой *_plan.json, поставь "keep": true/false и запусти снова с `--plan`.']
    open(path, 'w').write('\n'.join(L) + '\n')


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input'); ap.add_argument('output')
    ap.add_argument('--lang', default='ru', help='language code or auto (default ru)')
    ap.add_argument('--model', default='small', help='faster-whisper model: small (default), medium, large-v3')
    ap.add_argument('--prompt', default=None, help='hint words for Whisper: names, brands, slang')
    ap.add_argument('--gap', type=float, default=0.45, help='pause that ends a phrase, s')
    ap.add_argument('--window', type=float, default=20.0, help='how far ahead a retake is searched, s')
    ap.add_argument('--pause', type=float, default=0.30, help='cut pauses longer than this inside phrases, s')
    ap.add_argument('--keep-before', type=float, default=0.08)
    ap.add_argument('--keep-after', type=float, default=0.10)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--size', default='keep', help='keep (1080 on the short side) or WxH, e.g. 1080x1920')
    ap.add_argument('--crf', type=int, default=16)
    ap.add_argument('--preset', default='veryfast')
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument('--loudnorm', type=float, default=-14.0, help='target LUFS; pass nan to skip')
    ap.add_argument('--audio', default=None, help='use this full-length audio (e.g. denoised) instead of the camera track')
    ap.add_argument('--plan-only', action='store_true', help='stop after the plan + report (no rendering)')
    ap.add_argument('--plan', default=None, help='render from an edited *_plan.json')
    ap.add_argument('--no-check', action='store_true', help='skip the final re-transcription')
    a = ap.parse_args()
    if a.loudnorm is not None and np.isnan(a.loudnorm):
        a.loudnorm = None
    for tool in ('ffmpeg', 'ffprobe'):
        if not shutil.which(tool):
            sys.exit(f'{tool} is required')
    if not os.path.exists(a.input):
        sys.exit(f'no such file: {a.input}')
    j, v, au = ffprobe(a.input)
    if au is None:
        sys.exit('the input has no audio track')
    dur = float(j['format']['duration'])
    base = os.path.splitext(a.output)[0]
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    print(f'[1/5] audio · {fmt(dur)}', flush=True)
    a16 = load_pcm(a.input, SR_A)[:, 0]
    # cut points are found on the cleanest audio we have: after denoise the pauses are real silence
    db = envelope(load_pcm(a.audio, SR_A)[:, 0] if a.audio else a16)
    th = speech_threshold(db)
    if a.plan:
        plan = json.load(open(a.plan))
        ph = plan['phrases']
        for p in ph:
            p.setdefault('why', '' if p['keep'] else 'manual')
    else:
        print(f'[2/5] whisper {a.model} ({a.lang}) …', flush=True)
        t0 = time.time()
        words = transcribe(a16, a.lang, a.model, a.prompt)
        print(f'      {len(words)} words in {time.time() - t0:.0f}s', flush=True)
        ph = mark_retakes(split_restarts(to_phrases(words, a.gap)), a.window)
        json.dump({'input': os.path.abspath(a.input), 'phrases': [{k: p[k] for k in ('id', 's', 'e', 'text', 'p', 'keep', 'why', 'words')} for p in ph]},
                  open(base + '_plan.json', 'w'), ensure_ascii=False, indent=1)
    for p in ph:
        p.setdefault('tok', norm_tokens(p['text']))
    print(f'[3/5] phrases: {len(ph)} · kept {sum(p["keep"] for p in ph)} · dropped {sum(not p["keep"] for p in ph)}', flush=True)
    edl = build_edl(ph, db, th, a, a.fps, dur)
    json.dump(edl, open(base + '_edl.json', 'w'), indent=1)
    if a.plan_only:
        write_report(base + '_report.md', a, ph, edl, th, dur, None, None)
        print(f'plan only → {base}_report.md  (edit {base}_plan.json, then rerun with --plan)')
        return
    print(f'[4/5] render {len(edl)} pieces · {sum(p["frames"] for p in edl) / a.fps:.1f}s', flush=True)
    with tempfile.TemporaryDirectory(prefix='clean_takes_') as tmp:
        out_dur = render(a, a.input, edl, tmp, a.audio or a.input)
    check = None
    if not a.no_check:
        print('[5/5] check: transcribing the result …', flush=True)
        words = transcribe(load_pcm(a.output, SR_A)[:, 0], a.lang, a.model, a.prompt)
        check = ' '.join(w['w'] for w in words)
    write_report(base + '_report.md', a, ph, edl, th, dur, out_dur, check)
    print(f'done → {a.output}  ({fmt(out_dur)})\nreport → {base}_report.md')


if __name__ == '__main__':
    main()
