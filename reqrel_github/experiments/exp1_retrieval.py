"""E1: Stage 1 candidate retrieval on five corpora: Recall@k, share of kept pairs, AP of similarity ranking."""
import json, os, sys, time, random
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reqrel import data, retrieval
random.seed(42); np.random.seed(42)
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
METHODS = ["tfidf", "bm25", "dense", "hybrid_rrf", "hybrid_cc"]
KS = [1, 2, 3, 5, 10]
rows = []
for name, c in data.load_all().items():
    for m in METHODS:
        t = time.perf_counter(); S = retrieval.similarity(c.reqs, m); dt = time.perf_counter() - t
        iu = np.triu_indices(c.n, 1)
        y = np.array([(i, j) in c.conflicts for i, j in zip(*iu)])
        ap = average_precision_score(y, S[iu])
        for k in KS:
            kept = retrieval.topk_pairs(S, k)
            rows.append(dict(corpus=name, method=m, k=k, n=c.n, conflicts=len(c.conflicts),
                             recall=len(kept & c.conflicts) / len(c.conflicts),
                             kept_pct=100 * len(kept) / c.all_pairs, kept=len(kept), ap=ap, sim_time_s=dt))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "e1_retrieval.csv"), index=False)
print(df[df.k == 3].pivot_table(index="corpus", columns="method", values=["recall", "kept_pct", "ap"]).round(3).to_string())
tot = df.groupby(["method", "k"]).apply(lambda g: pd.Series({
    "recall_micro": (g.recall * g.conflicts).sum() / g.conflicts.sum(), "kept_pct_mean": g.kept_pct.mean()}), include_groups=False)
print(tot.round(3).to_string())
