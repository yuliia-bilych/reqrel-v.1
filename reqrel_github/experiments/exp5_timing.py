"""E5: throughput of NLI models on CPU and end-to-end time of the method vs. exhaustive NLI."""
import os, sys, time
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from reqrel import data, retrieval, nli
OUT = os.path.join(ROOT, "results")
c = data.load_corpus("worldvista")
pairs = [(c.reqs[i], c.reqs[i + 1]) for i in range(64)]
rows = []
for size in ["small", "base", "large"]:
    m = f"cross-encoder/nli-deberta-v3-{size}"
    nli.score_pairs(pairs[:8], m)                      # warm-up / load
    t = time.perf_counter(); nli.score_pairs(pairs, m); dt = time.perf_counter() - t
    rows.append(dict(model=size, pairs=len(pairs), sec_per_pair_both_dirs=dt / len(pairs)))
df = pd.DataFrame(rows)
t = time.perf_counter(); S = retrieval.similarity(c.reqs, "hybrid_cc"); cand = retrieval.topk_pairs(S, 3); t1 = time.perf_counter() - t
spp = df.set_index("model").loc["base", "sec_per_pair_both_dirs"]
summary = dict(corpus="worldvista", n=c.n, all_pairs=c.all_pairs, candidates=len(cand), stage1_sec=t1,
               method_total_sec_est=t1 + len(cand) * spp, exhaustive_sec_est=c.all_pairs * spp)
for n in (1000, 5000):
    summary[f"n{n}_method_pairs"] = 3 * n; summary[f"n{n}_exhaustive_pairs"] = n * (n - 1) // 2
df.to_csv(os.path.join(OUT, "e5_throughput.csv"), index=False); pd.DataFrame([summary]).to_csv(os.path.join(OUT, "e5_pipeline_time.csv"), index=False)
print(df.round(4).to_string()); print(summary)
