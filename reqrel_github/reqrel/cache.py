"""Disk cache of NLI scores so that experiments can be re-run without recomputation."""
from __future__ import annotations
import hashlib, json, os
from . import nli

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "cache")


def cached_scores(pairs, model_name, batch_size=32):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, model_name.replace("/", "__") + ".json")
    cache = json.load(open(path)) if os.path.exists(path) else {}
    key = lambda a, b: hashlib.sha1((a + "␞" + b).encode()).hexdigest()
    todo = [(a, b) for a, b in dict.fromkeys(pairs) if key(a, b) not in cache]
    if todo:
        for (a, b), s in zip(todo, nli.score_pairs(todo, model_name, batch_size)):
            cache[key(a, b)] = [s.e_ab, s.e_ba, s.c_ab, s.c_ba]
        json.dump(cache, open(path, "w"))
    return [nli.PairScores(*cache[key(a, b)]) for a, b in pairs]
