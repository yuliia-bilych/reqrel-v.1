"""End-to-end method: Stage 1 (hybrid top-k) -> Stage 2 (bidirectional NLI) -> Stage 3 (constraints)."""
from __future__ import annotations
from dataclasses import dataclass
from . import retrieval, nli, constraints


@dataclass
class Relation:
    i: int
    j: int
    label: str
    s_dup: float
    s_con: float
    e_ab: float
    e_ba: float


def detect(reqs: list[str], k: int = 3, retrieval_method: str = "hybrid_cc",
           nli_model: str = "cross-encoder/nli-deberta-v3-base", tau_e: float = 0.5, tau_c: float = 0.5,
           use_constraints: bool = True, rules: tuple = ("numeric",)) -> list[Relation]:
    """Recommended configuration: hybrid convex retrieval, k = 3, DeBERTaV3-base, numeric rule of Stage 3."""
    S = retrieval.similarity(reqs, retrieval_method)
    cand = sorted(retrieval.topk_pairs(S, k))
    scores = nli.score_pairs([(reqs[i], reqs[j]) for i, j in cand], nli_model)
    out = []
    for (i, j), s in zip(cand, scores):
        lab = nli.decide(s, tau_e, tau_c)
        if use_constraints:
            lab = constraints.fuse(lab, constraints.check(reqs[i], reqs[j]), rules)
        out.append(Relation(i, j, lab, s.s_dup, s.s_con, s.e_ab, s.e_ba))
    return out
