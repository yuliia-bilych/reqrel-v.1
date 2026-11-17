"""E4: do supervised baselines trained on the five corpora transfer to the paraphrase test set (PTS)?
B2 and B3 are trained on the pooled training data of all corpora (neutral train pairs + augmented conflicts)."""
import os, sys, random
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from reqrel import data
random.seed(42); np.random.seed(42)
OUT = os.path.join(ROOT, "results")
tr = []
for name, c in data.load_all().items():
    aug = pd.read_csv(os.path.join(data.DATA_DIR, "train", f"{name}_augment.csv"))
    tr += [(c.reqs[i], c.reqs[j], y) for (i, j), y in c.train_pairs.items()]
    tr += [(str(a).strip(), str(b).strip(), 1) for a, b in zip(aug.Text1, aug.Text2)]
pts = pd.read_csv(os.path.join(ROOT, "data", "pts.csv")); test = list(zip(pts.a, pts.b))
from sentence_transformers import SentenceTransformer
enc = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
rng = np.random.default_rng(42)
pos = [t for t in tr if t[2] == 1]; neg = [t for t in tr if t[2] == 0]
sub = [pos[i] for i in rng.permutation(len(pos))[:5000]] + [neg[i] for i in rng.permutation(len(neg))[:15000]]
TA = enc.encode([a for a, _, _ in sub], normalize_embeddings=True, batch_size=128); TB = enc.encode([b for _, b, _ in sub], normalize_embeddings=True, batch_size=128)
y = np.array([t[2] for t in sub])
lr = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42).fit(np.hstack([TA + TB, np.abs(TA - TB), TA * TB]), y)
PA = enc.encode(pts.a.tolist(), normalize_embeddings=True); PB = enc.encode(pts.b.tolist(), normalize_embeddings=True)
p2 = lr.predict(np.hstack([PA + PB, np.abs(PA - PB), PA * PB]))
from exp2_corpora_ft import finetune_ce
p3 = finetune_ce(tr, test)
rows = []
for nm, p in (("B2 bi-encoder + LR", p2), ("B3 fine-tuned NLI cross-encoder", p3)):
    for task, yt in (("contradiction vs rest", (pts.label == "contradiction").values), ("related vs neutral", (pts.label != "neutral").values)):
        P, R, F, _ = precision_recall_fscore_support(yt, p.astype(bool), average="binary", zero_division=0)
        rows.append(dict(method=nm, task=task, P=P, R=R, F=F))
    for lab in ["duplicate", "refinement", "contradiction", "neutral"]:
        rows.append(dict(method=nm, task=f"predicted conflict rate on {lab}", P=np.nan, R=p[(pts.label == lab).values].mean(), F=np.nan))
df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "e4_transfer.csv"), index=False); print(df.round(3).to_string())
