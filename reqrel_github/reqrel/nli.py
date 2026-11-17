"""Stage 2: bidirectional NLI scoring and relation decision.

For a pair (a, b), P(a, b) means premise = a, hypothesis = b.
  s_dup(a,b) = min(P_E(a,b), P_E(b,a))         (1)
  s_con(a,b) = max(P_C(a,b), P_C(b,a))         (2)
Decision: contradiction if s_con >= tau_c and s_con dominates entailment;
          duplicate if s_dup >= tau_e (mutual entailment);
          refinement if exactly one direction entails (max >= tau_e);
          neutral otherwise.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

LABELS = ("duplicate", "refinement", "contradiction", "neutral")
_MODELS = {}


@dataclass
class PairScores:
    e_ab: float
    e_ba: float
    c_ab: float
    c_ba: float

    @property
    def s_dup(self) -> float:
        return min(self.e_ab, self.e_ba)

    @property
    def s_con(self) -> float:
        return max(self.c_ab, self.c_ba)

    @property
    def s_ent_max(self) -> float:
        return max(self.e_ab, self.e_ba)


def _model(name: str):
    from sentence_transformers import CrossEncoder
    if name not in _MODELS:
        _MODELS[name] = CrossEncoder(name, device="cpu")
    return _MODELS[name]


def score_pairs(pairs: list[tuple[str, str]], model_name: str = "cross-encoder/nli-deberta-v3-base",
                batch_size: int = 32) -> list[PairScores]:
    """Runs the NLI cross-encoder in both directions for every pair."""
    if not pairs:
        return []
    m = _model(model_name)
    id2label = {int(k): v.lower() for k, v in m.model.config.id2label.items()}
    ci = [k for k, v in id2label.items() if v.startswith("contra")][0]
    ei = [k for k, v in id2label.items() if v.startswith("entail")][0]
    fwd = m.predict(pairs, batch_size=batch_size, apply_softmax=True, show_progress_bar=False)
    bwd = m.predict([(b, a) for a, b in pairs], batch_size=batch_size, apply_softmax=True, show_progress_bar=False)
    fwd, bwd = np.asarray(fwd), np.asarray(bwd)
    return [PairScores(float(f[ei]), float(b[ei]), float(f[ci]), float(b[ci])) for f, b in zip(fwd, bwd)]


def decide(s: PairScores, tau_e: float = 0.5, tau_c: float = 0.5) -> str:
    if s.s_con >= tau_c and s.s_con >= s.s_ent_max:
        return "contradiction"
    if s.s_dup >= tau_e:
        return "duplicate"
    if s.s_ent_max >= tau_e:
        return "refinement"
    return "neutral"


def decide_one_direction(s: PairScores, tau_e: float = 0.5, tau_c: float = 0.5) -> str:
    """Ablation: classic one-directional NLI (premise a, hypothesis b only)."""
    if s.c_ab >= tau_c and s.c_ab >= s.e_ab:
        return "contradiction"
    if s.e_ab >= tau_e:
        return "duplicate"
    return "neutral"
