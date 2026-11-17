"""E6: additional pair-level benchmark CN (synthetic perturbations of real requirements: numbers,
negation, modality, paraphrases; labels Conflict / Neutral). Zero-shot method vs supervised baselines."""
import os, sys, random
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, f1_score
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from reqrel import data, nli, constraints
from reqrel.cache import cached_scores
from exp2_corpora_ft import finetune_ce
random.seed(42); np.random.seed(42)
OUT = os.path.join(ROOT, "results")
rd = lambda s: pd.read_csv(os.path.join(data.DATA_DIR, s, "cn.csv"))
tr, te = rd("train"), rd("test")
for d in (tr, te):
    d["Text1"] = d.Text1.astype(str).str.strip(); d["Text2"] = d.Text2.astype(str).str.strip()
ytr = (tr.Class == "Conflict").values.astype(int); yte = (te.Class == "Conflict").values.astype(int)
pte = list(zip(te.Text1, te.Text2))
from sentence_transformers import SentenceTransformer
enc = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
E = lambda xs: enc.encode(list(xs), normalize_embeddings=True, batch_size=128, show_progress_bar=False)
A, B, TA, TB = E(te.Text1), E(te.Text2), E(tr.Text1), E(tr.Text2)
prf = lambda y, p: dict(zip("PRF", precision_recall_fscore_support(y, p, average="binary", zero_division=0)[:3]))
rows = []
cos_tr, cos_te = (TA * TB).sum(1), (A * B).sum(1)
th = max(np.unique(np.round(cos_tr, 3)), key=lambda t: f1_score(ytr, cos_tr >= t))
rows.append(dict(method="B1 cosine threshold", **prf(yte, cos_te >= th)))
lr = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42).fit(np.hstack([TA + TB, abs(TA - TB), TA * TB]), ytr)
rows.append(dict(method="B2 bi-encoder + LR (supervised)", **prf(yte, lr.predict(np.hstack([A + B, abs(A - B), A * B])))))
b3 = os.path.join(OUT, "cache", "b3_cn.npy")
if not os.path.exists(b3):
    np.save(b3, finetune_ce(list(zip(tr.Text1, tr.Text2, ytr)), pte))
rows.append(dict(method="B3 fine-tuned NLI cross-encoder (supervised)", **prf(yte, np.load(b3))))
S = cached_scores(pte, "cross-encoder/nli-deberta-v3-base")
lab = {"M1 NLI one-direction": [nli.decide_one_direction(s) for s in S], "M2 NLI bidirectional": [nli.decide(s) for s in S]}
lab["M3n M2 + numeric rule"] = [constraints.fuse(l, constraints.check(a, b), ("numeric",)) for l, (a, b) in zip(lab["M2 NLI bidirectional"], pte)]
lab["M3 M2 + all Stage-3 rules"] = [constraints.fuse(l, constraints.check(a, b)) for l, (a, b) in zip(lab["M2 NLI bidirectional"], pte)]
for k, v in lab.items():
    rows.append(dict(method=k + " [any relation]", **prf(yte, np.array([x != "neutral" for x in v]))))
    rows.append(dict(method=k + " [contradiction only]", **prf(yte, np.array([x == "contradiction" for x in v]))))
df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "e6_cn.csv"), index=False)
dist = pd.crosstab(te.Class, pd.Series(lab["M2 NLI bidirectional"], name="pred")); dist.to_csv(os.path.join(OUT, "e6_cn_dist.csv"))
print(len(tr), len(te), yte.mean().round(3)); print(df.round(3).to_string()); print(dist)
