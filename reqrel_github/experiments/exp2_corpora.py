"""E2: Stage 2/3 on the five public corpora (binary labels Conflict / Neutral).
(a) test-split comparison of classifiers; (b) end-to-end pipeline on all pairs: what relation types
the method assigns to the labelled conflicts; (c) timing."""
import os, sys, time, json, random
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, f1_score
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from reqrel import data, retrieval, nli, constraints
from reqrel.cache import cached_scores
random.seed(42); np.random.seed(42)
OUT = os.path.join(ROOT, "results")
MODEL = os.environ.get("NLI_MODEL", "cross-encoder/nli-deberta-v3-base")
ENC = "BAAI/bge-small-en-v1.5"
from sentence_transformers import SentenceTransformer
enc = SentenceTransformer(ENC, device="cpu")

def feats(E, pairs):
    u = E[[i for i, _ in pairs]]; v = E[[j for _, j in pairs]]
    return np.hstack([u + v, np.abs(u - v), u * v])          # symmetric SBERT-style pair features

def prf(y, p):
    P, R, F, _ = precision_recall_fscore_support(y, p, average="binary", zero_division=0)
    return P, R, F

def finetune_ce(tr_txt, test_pairs, base="cross-encoder/nli-deberta-v3-xsmall", n_pos=1000, n_neg=2000, epochs=1):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    torch.manual_seed(42); rng = np.random.default_rng(42)
    pos = [t for t in tr_txt if t[2] == 1]; neg = [t for t in tr_txt if t[2] == 0]
    pos = [pos[i] for i in rng.permutation(len(pos))[:n_pos]]; neg = [neg[i] for i in rng.permutation(len(neg))[:n_neg]]
    data_ = pos + neg; order = rng.permutation(len(data_))
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=2, ignore_mismatched_sizes=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    model.train()
    for _ in range(epochs):
        for s0 in range(0, len(order), 16):
            b = [data_[k] for k in order[s0:s0 + 16]]
            enc_ = tok([x[0] for x in b], [x[1] for x in b], truncation=True, max_length=128, padding=True, return_tensors="pt")
            out = model(**enc_, labels=torch.tensor([x[2] for x in b]))
            out.loss.backward(); opt.step(); opt.zero_grad()
    model.eval(); preds = []
    with torch.no_grad():
        for s0 in range(0, len(test_pairs), 64):
            b = test_pairs[s0:s0 + 64]
            enc_ = tok([x[0] for x in b], [x[1] for x in b], truncation=True, max_length=128, padding=True, return_tensors="pt")
            preds += model(**enc_).logits.argmax(-1).tolist()
    return np.array(preds)

rows, pipe_rows, label_dist = [], [], []
for name, c in data.load_all().items():
    E = enc.encode(c.reqs, normalize_embeddings=True, show_progress_bar=False)
    te = list(c.test_pairs.items()); yte = np.array([y for _, y in te]); pte = [p for p, _ in te]
    # supervised training data: official train split (neutral pairs) + augmented synthetic conflicts of the repository
    aug = pd.read_csv(os.path.join(data.DATA_DIR, "train", f"{name}_augment.csv"))
    tr_txt = [(c.reqs[i], c.reqs[j], y) for (i, j), y in c.train_pairs.items()] + \
             [(str(a).strip(), str(b).strip(), 1) for a, b in zip(aug.Text1, aug.Text2)]
    ytr = np.array([y for _, _, y in tr_txt])
    TA = enc.encode([a for a, _, _ in tr_txt], normalize_embeddings=True, batch_size=128, show_progress_bar=False)
    TB = enc.encode([b for _, b, _ in tr_txt], normalize_embeddings=True, batch_size=128, show_progress_bar=False)
    cos_tr = (TA * TB).sum(1); cos_te = np.array([E[i] @ E[j] for i, j in pte])
    # B1: cosine threshold chosen on the training data (max F1) – S3CDA-style data-driven threshold
    grid = np.unique(np.round(cos_tr, 3))
    th = max(grid, key=lambda t: f1_score(ytr, cos_tr >= t, zero_division=0))
    rows.append(dict(corpus=name, method="B1 cosine threshold", **dict(zip("PRF", prf(yte, cos_te >= th))), thr=th))
    # B2: supervised bi-encoder pair classifier (simplified SR-BERT analogue: frozen encoder + logistic regression)
    Xtr = np.hstack([TA + TB, np.abs(TA - TB), TA * TB])
    lr = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0, random_state=42).fit(Xtr, ytr)
    rows.append(dict(corpus=name, method="B2 bi-encoder + LR (supervised)", **dict(zip("PRF", prf(yte, lr.predict(feats(E, pte)))))))
    # B3: fine-tuned cross-encoder initialised from an MNLI checkpoint (sequential transfer, as in SR-BERT study)
    b3_path = os.path.join(OUT, "cache", f"b3_{name}.npy")
    if os.path.exists(b3_path):
        pb3 = np.load(b3_path)
    else:
        pb3 = finetune_ce(tr_txt, [(c.reqs[i], c.reqs[j]) for i, j in pte]); np.save(b3_path, pb3)
    rows.append(dict(corpus=name, method="B3 fine-tuned NLI cross-encoder (supervised)", **dict(zip("PRF", prf(yte, pb3)))))
    # Proposed: zero-shot bidirectional NLI on every test pair (no Stage 1)
    S = cached_scores([(c.reqs[i], c.reqs[j]) for i, j in pte], MODEL)
    lab2 = [nli.decide(s) for s in S]
    lab3 = [constraints.fuse(l, constraints.check(c.reqs[i], c.reqs[j])) for l, (i, j) in zip(lab2, pte)]
    lab1 = [nli.decide_one_direction(s) for s in S]
    lab3n = [constraints.fuse(l, constraints.check(c.reqs[i], c.reqs[j]), ("numeric",)) for l, (i, j) in zip(lab2, pte)]
    for meth, labs in (("M1 NLI one-direction", lab1), ("M2 NLI bidirectional", lab2), ("M3 NLI bidirectional + Stage 3", lab3)):
        rows.append(dict(corpus=name, method=meth + " [any relation]", **dict(zip("PRF", prf(yte, np.array([l != "neutral" for l in labs]))))))
        rows.append(dict(corpus=name, method=meth + " [contradiction only]", **dict(zip("PRF", prf(yte, np.array([l == "contradiction" for l in labs]))))))
    # Full method = Stage 1 (hybrid_cc, k=3) + M3: pairs not retrieved are predicted neutral
    Sim = retrieval.similarity(c.reqs, "hybrid_cc"); cand = retrieval.topk_pairs(Sim, 3)
    lab_full = [l if p in cand else "neutral" for l, p in zip(lab3, pte)]
    rows.append(dict(corpus=name, method="Full method (Stage 1 k=3 + M3) [any relation]", **dict(zip("PRF", prf(yte, np.array([l != "neutral" for l in lab_full]))))))
    rows.append(dict(corpus=name, method="Full method (Stage 1 k=3 + M3) [contradiction only]", **dict(zip("PRF", prf(yte, np.array([l == "contradiction" for l in lab_full]))))))
    lab_full_n = [l if p in cand else "neutral" for l, p in zip(lab3n, pte)]
    rows.append(dict(corpus=name, method="Full method (Stage 1 k=3 + M3n) [any relation]", **dict(zip("PRF", prf(yte, np.array([l != "neutral" for l in lab_full_n]))))))
    lab_full_2 = [l if p in cand else "neutral" for l, p in zip(lab2, pte)]
    rows.append(dict(corpus=name, method="Stage 1 k=3 + M2 [any relation]", **dict(zip("PRF", prf(yte, np.array([l != "neutral" for l in lab_full_2]))))))
    # (b) end-to-end on all pairs of the corpus
    t0 = time.perf_counter()
    Sim = retrieval.similarity(c.reqs, "hybrid_cc"); cand = sorted(retrieval.topk_pairs(Sim, 3)); t1 = time.perf_counter()
    SC = cached_scores([(c.reqs[i], c.reqs[j]) for i, j in cand], MODEL); t2 = time.perf_counter()
    labs = {p: constraints.fuse(nli.decide(s), constraints.check(c.reqs[p[0]], c.reqs[p[1]])) for p, s in zip(cand, SC)}
    flagged = {p for p, l in labs.items() if l != "neutral"}
    for p in c.conflicts:
        label_dist.append(dict(corpus=name, label=labs.get(p, "not retrieved"), a=c.reqs[p[0]], b=c.reqs[p[1]]))
    pipe_rows.append(dict(corpus=name, n=c.n, all_pairs=c.all_pairs, candidates=len(cand), flagged=len(flagged),
                          flagged_dup=sum(l == "duplicate" for l in labs.values()), flagged_ref=sum(l == "refinement" for l in labs.values()),
                          flagged_con=sum(l == "contradiction" for l in labs.values()),
                          conflicts=len(c.conflicts), conflicts_flagged=len(flagged & c.conflicts),
                          conflicts_as_contradiction=sum(labs.get(p) == "contradiction" for p in c.conflicts)))
    print(name, "done", flush=True)

df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "e2_testsplit.csv"), index=False)
pd.DataFrame(pipe_rows).to_csv(os.path.join(OUT, "e2_pipeline.csv"), index=False)
pd.DataFrame(label_dist).to_csv(os.path.join(OUT, "e2_conflict_labels.csv"), index=False)
print(df.pivot_table(index="method", columns="corpus", values="F").round(3).to_string())
print(df.groupby("method")[["P", "R", "F"]].mean().round(3).to_string())
print(pd.DataFrame(pipe_rows).to_string())
print(pd.DataFrame(label_dist).label.value_counts().to_string())
