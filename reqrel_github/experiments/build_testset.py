"""Builds data/pts.csv – the 4-class paraphrase test set (PTS) from seeds + hard neutral negatives."""
import os, sys, random, re
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "data"))
from reqrel import data
from paraphrase_testset_seeds import SEEDS, NEUTRAL_PICK
random.seed(42)
norm = lambda t: re.sub(r"[_\"]", "", t).strip()
corp = data.load_all()
rows = []
for sid, (cname, seed, dup, ref, con, ctype) in enumerate(SEEDS):
    c = corp[cname]
    creqs = [norm(r) for r in c.reqs]
    in_corpus = seed in creqs
    for b, lab, sub in ((dup, "duplicate", ""), (ref, "refinement", ""), (con, "contradiction", ctype)):
        a_, b_ = (seed, b) if random.random() < 0.5 else (b, seed)
        rows.append(dict(seed_id=sid, corpus=cname, a=a_, b=b_, label=lab, con_type=sub, seed_in_corpus=in_corpus))
    # hard negative: manually verified neutral partner among the most similar corpus requirements
    rows.append(dict(seed_id=sid, corpus=cname, a=seed, b=creqs[NEUTRAL_PICK[sid]], label="neutral", con_type="", seed_in_corpus=in_corpus))
df = pd.DataFrame(rows)
# calibration/test split by seed (stratified by contradiction subtype), used only for threshold calibration
import numpy as np
rng = np.random.default_rng(42); split = {}
for ct in sorted(set(s[5] for s in SEEDS)):
    ids = [i for i, s in enumerate(SEEDS) if s[5] == ct]; rng.shuffle(ids)
    for k, i in enumerate(ids):
        split[i] = "dev" if k % 2 == 0 else "test"
df["split"] = df.seed_id.map(split)
tok = lambda t: set(re.findall(r"\w+", t.lower()))
df["jaccard"] = [len(tok(a) & tok(b)) / len(tok(a) | tok(b)) for a, b in zip(df.a, df.b)]
df.to_csv(os.path.join(ROOT, "data", "pts.csv"), index=False)
print(len(SEEDS), "seeds;", df.label.value_counts().to_dict(), "; seeds in corpus:", df.drop_duplicates("seed_id").seed_in_corpus.sum())
print(df.groupby("label").jaccard.mean().round(3).to_dict())
print(df[df.label == "contradiction"].con_type.value_counts().to_dict()); print(df.split.value_counts().to_dict())
