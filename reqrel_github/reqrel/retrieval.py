"""Stage 1: candidate retrieval. For each requirement keep its top-k neighbours by
lexical (TF-IDF / BM25), dense (bi-encoder) or hybrid similarity; the candidate set is the
union of (i, j) neighbour pairs, so |C| <= k*n instead of n(n-1)/2."""
from __future__ import annotations
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def tfidf_sim(reqs: list[str]) -> np.ndarray:
    X = TfidfVectorizer().fit_transform(reqs)
    S = (X @ X.T).toarray()
    np.fill_diagonal(S, -np.inf)
    return S


def bm25_sim(reqs: list[str], k1: float = 1.2, b: float = 0.75) -> np.ndarray:
    docs = [tokenize(r) for r in reqs]
    bm = BM25Okapi(docs, k1=k1, b=b)
    S = np.vstack([bm.get_scores(q) for q in docs])
    S = (S + S.T) / 2.0                      # symmetrise: BM25 is asymmetric
    np.fill_diagonal(S, -np.inf)
    return S


_ENCODERS = {}


def dense_sim(reqs: list[str], model_name: str = "BAAI/bge-small-en-v1.5") -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    if model_name not in _ENCODERS:
        _ENCODERS[model_name] = SentenceTransformer(model_name, device="cpu")
    E = _ENCODERS[model_name].encode(reqs, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    S = E @ E.T
    np.fill_diagonal(S, -np.inf)
    return S


def _ranks(S: np.ndarray) -> np.ndarray:
    """rank[i, j] = 1-based position of j in the neighbour list of i."""
    order = np.argsort(-S, axis=1, kind="stable")
    R = np.empty_like(order)
    rows = np.arange(S.shape[0])[:, None]
    R[rows, order] = np.arange(1, S.shape[1] + 1)[None, :]
    return R


def rrf_sim(*mats: np.ndarray, k0: int = 60) -> np.ndarray:
    """Reciprocal Rank Fusion of several similarity matrices."""
    F = sum(1.0 / (k0 + _ranks(S)) for S in mats)
    F = (F + F.T) / 2.0
    np.fill_diagonal(F, -np.inf)
    return F


def convex_sim(S_lex: np.ndarray, S_dense: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Convex combination of min-max normalised scores (alpha = weight of dense)."""
    def norm(S):
        v = S[np.isfinite(S)]
        return (S - v.min()) / (v.max() - v.min() + 1e-12)
    F = alpha * norm(S_dense) + (1 - alpha) * norm(S_lex)
    np.fill_diagonal(F, -np.inf)
    return F


def topk_pairs(S: np.ndarray, k: int) -> set[tuple[int, int]]:
    order = np.argsort(-S, axis=1, kind="stable")[:, :k]
    return {(min(i, j), max(i, j)) for i in range(S.shape[0]) for j in order[i]}


def similarity(reqs: list[str], method: str, **kw) -> np.ndarray:
    if method == "tfidf":
        return tfidf_sim(reqs)
    if method == "bm25":
        return bm25_sim(reqs)
    if method == "dense":
        return dense_sim(reqs, kw.get("model_name", "BAAI/bge-small-en-v1.5"))
    if method == "hybrid_rrf":
        return rrf_sim(bm25_sim(reqs), dense_sim(reqs, kw.get("model_name", "BAAI/bge-small-en-v1.5")))
    if method == "hybrid_cc":
        return convex_sim(bm25_sim(reqs), dense_sim(reqs, kw.get("model_name", "BAAI/bge-small-en-v1.5")), kw.get("alpha", 0.5))
    raise ValueError(method)
