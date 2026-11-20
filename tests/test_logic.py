"""Retake / restart / filler detection without any model:  python tests/test_logic.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import clean_takes as c  # noqa: E402


def W(text, t0=0.0, step=0.3):
    return [{'s': t0 + i * step, 'e': t0 + i * step + 0.25, 'w': w, 'p': 0.9} for i, w in enumerate(text.split())]


def plan(words):
    return [(p['text'], p['keep']) for p in c.mark_retakes(c.split_restarts(c.to_phrases(words, 0.45)), 20)]


CASES = [
    # whole phrase said twice -> the second one stays
    ('retake', W('У вас тоже бесит эта сова?') + W('У вас тоже бесит эта сова в самый неподходящий момент?', 5),
     [False, True]),
    # same start, different ending (the speaker changed the joke) -> last attempt stays
    ('changed ending', W('Короче, это не приложение для изучения языков, это зель.') + W('Короче, это не приложение для языков, это абьюзер.', 6),
     [False, True]),
    # restart inside one phrase
    ('restart', W('и вот обещанная мерзость и вот обещанная мразь если ты выключил уведомления.'), [False, True]),
    # false start: one word, then the sentence
    ('false start', W('Короче,') + W('Короче, внутри есть папка churn.', 2), [False, True]),
    # filler phrase
    ('filler', W('Ээ.') + W('Дальше Wi-Fi.', 2), [False, True]),
    # normal speech with repeated words must survive
    ('no false alarm', W('Сколько уроков прошёл, сколько раз накосячил, вот вот сгорит серия.') + W('Она давит ровно на то, что тебя цепляет.', 5),
     [True, True]),
    # list of questions starting with the same word must survive
    ('enumeration', W('Сколько потратил?') + W('Сколько часов назад смотрел рекламу?', 1.5), [True, True]),
    ('same first word, different sentences', W('Она запоминает твои сети.') + W('Она давит ровно туда, где больно.', 3), [True, True]),
]

bad = 0
for name, words, want in CASES:
    got = [k for _, k in plan(words)]
    ok = got == want
    bad += not ok
    print(('ok  ' if ok else 'FAIL'), name, got if not ok else '')
sys.exit(1 if bad else 0)
