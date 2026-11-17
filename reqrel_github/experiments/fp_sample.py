"""Sample of pairs flagged by the full method (Stage 1 k=3 + M2) that are labelled Neutral in the corpora, for manual review."""
import os, sys, random
import pandas as pd, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from reqrel import data, retrieval, nli
from reqrel.cache import cached_scores
rows = []
for name, c in data.load_all().items():
    cand = sorted(retrieval.topk_pairs(retrieval.similarity(c.reqs, "hybrid_cc"), 3))
    S = cached_scores([(c.reqs[i], c.reqs[j]) for i, j in cand], "cross-encoder/nli-deberta-v3-base")
    for (i, j), s in zip(cand, S):
        lab = nli.decide(s)
        if lab != "neutral" and (i, j) not in c.conflicts:
            rows.append(dict(corpus=name, pred=lab, a=c.reqs[i], b=c.reqs[j]))
df = pd.DataFrame(rows); print(len(df), df.pred.value_counts().to_dict())
smp = df.groupby("corpus", group_keys=False).apply(lambda g: g.sample(8, random_state=42))
smp.to_csv(os.path.join(ROOT, "results", "fp_sample.csv"), index=False)
pd.set_option("display.max_colwidth", 130); pd.set_option("display.width", 300)
for k, r in smp.reset_index(drop=True).iterrows(): print(k, r.corpus, r.pred, "|", r.a, "||", r.b)
