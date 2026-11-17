"""E3: 4-class evaluation on the paraphrase test set (PTS): ablation of directions, Stage 3,
NLI model size, threshold sensitivity, bootstrap confidence intervals, timing."""
import os, sys, time, random, json
import numpy as np, pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, f1_score
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from reqrel import nli, constraints, data
from reqrel.cache import cached_scores
random.seed(42); np.random.seed(42)
OUT = os.path.join(ROOT, "results")
L = list(nli.LABELS)
pts = pd.read_csv(os.path.join(ROOT, "data", "pts.csv"))
pairs = list(zip(pts.a, pts.b)); y = pts.label.tolist()
MODELS = {"small": "cross-encoder/nli-deberta-v3-small", "base": "cross-encoder/nli-deberta-v3-base",
          "large": "cross-encoder/nli-deberta-v3-large", "roberta": "cross-encoder/nli-roberta-base",
          "minilm": "cross-encoder/nli-MiniLM2-L6-H768"}

def boot_ci(y, p, n=1000):
    y, p = np.array(y), np.array(p); rng = np.random.default_rng(42); v = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        v.append(f1_score(y[idx], p[idx], labels=L, average="macro", zero_division=0))
    return np.percentile(v, [2.5, 97.5])

rows, preds, timing = [], {}, []
for size, mname in MODELS.items():
    t = time.perf_counter(); S = cached_scores(pairs, mname); dt = time.perf_counter() - t
    timing.append(dict(model=size, pairs=len(pairs), seconds=dt))
    variants = {
        "M1 one-direction": [nli.decide_one_direction(s) for s in S],
        "M2 bidirectional": [nli.decide(s) for s in S],
        "M3 bidirectional + Stage 3": [constraints.fuse(nli.decide(s), constraints.check(a, b)) for s, (a, b) in zip(S, pairs)],
        "M3n bidirectional + numeric rule": [constraints.fuse(nli.decide(s), constraints.check(a, b), ("numeric",)) for s, (a, b) in zip(S, pairs)],
    }
    if size == "base":
        abl = []
        for rules in [(), ("numeric",), ("added",), ("negation",), ("modality",), constraints.ALL_RULES]:
            p_ = [constraints.fuse(nli.decide(s), constraints.check(a, b), rules) for s, (a, b) in zip(S, pairs)]
            abl.append(dict(rules="+".join(rules) or "none", macro_f1=f1_score(y, p_, labels=L, average="macro"),
                            **{f"F1_{l}": v for l, v in zip(L, f1_score(y, p_, labels=L, average=None))}))
        pd.DataFrame(abl).to_csv(os.path.join(OUT, "e3_stage3_ablation.csv"), index=False)
    for v, p in variants.items():
        P, R, F, _ = precision_recall_fscore_support(y, p, labels=L, zero_division=0)
        lo, hi = boot_ci(y, p)
        row = dict(model=size, variant=v, macro_f1=np.mean(F), ci_lo=lo, ci_hi=hi)
        for lab, pp, rr, ff in zip(L, P, R, F):
            row.update({f"P_{lab}": pp, f"R_{lab}": rr, f"F1_{lab}": ff})
        rows.append(row); preds[f"{size}|{v}"] = p
    # threshold sensitivity for M3
    if size == "base":
        sens = []
        for te in np.arange(0.3, 0.91, 0.1):
            for tc in np.arange(0.3, 0.91, 0.1):
                p = [constraints.fuse(nli.decide(s, te, tc), constraints.check(a, b)) for s, (a, b) in zip(S, pairs)]
                sens.append(dict(tau_e=round(te, 1), tau_c=round(tc, 1), macro_f1=f1_score(y, p, labels=L, average="macro")))
        pd.DataFrame(sens).to_csv(os.path.join(OUT, "e3_threshold_sensitivity.csv"), index=False)
        pts_scores = pts.copy()
        pts_scores[["e_ab", "e_ba", "c_ab", "c_ba"]] = [[s.e_ab, s.e_ba, s.c_ab, s.c_ba] for s in S]
        pts_scores["pred_M2"] = variants["M2 bidirectional"]; pts_scores["pred_M3"] = variants["M3 bidirectional + Stage 3"]
        pts_scores.to_csv(os.path.join(OUT, "e3_pts_predictions_base.csv"), index=False)

# Similarity baseline on PTS: cosine threshold learned on the corpora (B1) -> can only say related/unrelated
from sentence_transformers import SentenceTransformer
enc = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
A = enc.encode(pts.a.tolist(), normalize_embeddings=True); B = enc.encode(pts.b.tolist(), normalize_embeddings=True)
cos = (A * B).sum(1)
pts_cos = pts.assign(cos=cos)
thr = float(pd.read_csv(os.path.join(OUT, "e2_testsplit.csv")).query("method=='B1 cosine threshold'").thr.median())
yb = (pts.label != "neutral").values
binrows = []
for nm, pb in [("B1 cosine threshold (corpora)", cos >= thr)] + \
              [(f"{k}", np.array([l != "neutral" for l in preds[k]])) for k in ["base|M2 bidirectional", "base|M3 bidirectional + Stage 3", "large|M3 bidirectional + Stage 3"]]:
    P, R, F, _ = precision_recall_fscore_support(yb, pb, average="binary", zero_division=0)
    binrows.append(dict(method=nm, task="related vs neutral", P=P, R=R, F=F))
yc = (pts.label == "contradiction").values
for nm, pc in [("B1 cosine threshold (corpora)", cos >= thr)] + \
              [(f"{k}", np.array([l == "contradiction" for l in preds[k]])) for k in ["base|M1 one-direction", "base|M2 bidirectional", "base|M3 bidirectional + Stage 3", "large|M3 bidirectional + Stage 3"]]:
    P, R, F, _ = precision_recall_fscore_support(yc, pc, average="binary", zero_division=0)
    binrows.append(dict(method=nm, task="contradiction vs rest", P=P, R=R, F=F))
pd.DataFrame(binrows).to_csv(os.path.join(OUT, "e3_binary.csv"), index=False)
pts_cos.groupby("label").cos.describe().to_csv(os.path.join(OUT, "e3_cosine_by_label.csv"))

res = pd.DataFrame(rows); res.to_csv(os.path.join(OUT, "e3_pts.csv"), index=False)
pd.DataFrame(timing).to_csv(os.path.join(OUT, "e3_timing.csv"), index=False)
for k in ["base|M2 bidirectional", "base|M3 bidirectional + Stage 3", "large|M3 bidirectional + Stage 3"]:
    cm = confusion_matrix(y, preds[k], labels=L)
    pd.DataFrame(cm, index=L, columns=L).to_csv(os.path.join(OUT, f"e3_cm_{k.split('|')[0]}_{k.split('|')[1].split()[0]}.csv"))
# contradiction recall by subtype (base M2 vs M3)
sub = pts[pts.label == "contradiction"].copy()
for k in ["base|M2 bidirectional", "base|M3 bidirectional + Stage 3", "large|M3 bidirectional + Stage 3"]:
    sub[k] = [preds[k][i] == "contradiction" for i in sub.index]
sub.groupby("con_type")[[c for c in sub.columns if "|" in c]].mean().to_csv(os.path.join(OUT, "e3_con_subtype.csv"))
print(res[["model", "variant", "macro_f1", "ci_lo", "ci_hi", "F1_duplicate", "F1_refinement", "F1_contradiction", "F1_neutral"]].round(3).to_string())
print(pd.DataFrame(binrows).round(3).to_string()); print("thr", thr)
print(pts_cos.groupby("label").cos.mean().round(3).to_dict())
print(sub.groupby("con_type")[[c for c in sub.columns if "|" in c]].mean().round(2).to_string())
print(pd.DataFrame(timing).to_string())


# ---------- calibration on the dev half, evaluation on the test half (base model) ----------
from scipy.stats import binomtest
S = cached_scores(pairs, MODELS["base"])
dev = (pts.split == "dev").values; tst = ~dev
yarr = np.array(y)
best = max(((te, tc) for te in np.arange(0.3, 0.91, 0.1) for tc in np.arange(0.3, 0.91, 0.1)),
           key=lambda t: f1_score(yarr[dev], [nli.decide(S[i], *t) for i in np.where(dev)[0]], labels=L, average="macro"))
cal = []
for nm, (te, tc) in (("default 0.5/0.5", (0.5, 0.5)), (f"calibrated {best[0]:.1f}/{best[1]:.1f}", best)):
    p_ = [nli.decide(S[i], te, tc) for i in np.where(tst)[0]]
    lo, hi = boot_ci(yarr[tst], p_)
    cal.append(dict(setting=nm, macro_f1_test=f1_score(yarr[tst], p_, labels=L, average="macro"), ci_lo=lo, ci_hi=hi,
                    **{f"F1_{l}": v for l, v in zip(L, f1_score(yarr[tst], p_, labels=L, average=None))}))
pd.DataFrame(cal).to_csv(os.path.join(OUT, "e3_calibration.csv"), index=False)
print(pd.DataFrame(cal).round(3).to_string())

# ---------- McNemar (exact) tests on per-pair correctness ----------
def mcnemar(a_ok, b_ok):
    b = int(np.sum(a_ok & ~b_ok)); c = int(np.sum(~a_ok & b_ok))
    return b, c, (binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0)
ok = {k: np.array(v) == yarr for k, v in preds.items()}
tests = [("base M2 vs base M1", ok["base|M2 bidirectional"], ok["base|M1 one-direction"]),
         ("base M2 vs base M3", ok["base|M2 bidirectional"], ok["base|M3 bidirectional + Stage 3"]),
         ("base M2 vs large M2", ok["base|M2 bidirectional"], ok["large|M2 bidirectional"])]
yc_ = yarr == "contradiction"
tests.append(("contradiction: base M2 vs B1 cosine", (np.array(preds["base|M2 bidirectional"]) == "contradiction") == yc_, (cos >= thr) == yc_))
mc = pd.DataFrame([dict(test=t, a_only_correct=r[0], b_only_correct=r[1], p_value=r[2]) for t, (a_, b_) in [(t, (a, b)) for t, a, b in tests] for r in [mcnemar(a_, b_)]])
mc.to_csv(os.path.join(OUT, "e3_mcnemar.csv"), index=False); print(mc.to_string())

# ---------- ALICE (external, logic-style requirements; 3 classes) ----------
al = pd.concat([pd.read_csv(os.path.join(data.DATA_DIR, "train", "alice.csv")), pd.read_csv(os.path.join(data.DATA_DIR, "test", "alice.csv"))])
def amap(c):
    c = str(c)
    if "subaltern" in c: return "refinement"
    if c in ("0", "Neutral"): return "neutral"
    return "contradiction"
al["y"] = al.Class.map(amap)
ap = list(zip(al.Text1.astype(str).str.strip(), al.Text2.astype(str).str.strip()))
SA = cached_scores(ap, MODELS["base"])
arows = []
for nm, fn in (("M1", lambda s, a, b: nli.decide_one_direction(s)), ("M2", lambda s, a, b: nli.decide(s)),
               ("M3n", lambda s, a, b: constraints.fuse(nli.decide(s), constraints.check(a, b), ("numeric",)))):
    p_ = [fn(s, a, b) for s, (a, b) in zip(SA, ap)]
    p_ = ["neutral" if q == "duplicate" else q for q in p_] if False else p_
    labs = ["refinement", "contradiction", "neutral"]
    arows.append(dict(variant=nm, n=len(ap), macro_f1=f1_score(al.y, p_, labels=labs, average="macro"),
                      **{f"F1_{l}": v for l, v in zip(labs, f1_score(al.y, p_, labels=labs, average=None))},
                      acc_contradiction=np.mean([q == "contradiction" for q, t in zip(p_, al.y) if t == "contradiction"])))
pd.DataFrame(arows).to_csv(os.path.join(OUT, "e3_alice.csv"), index=False)
print(al.y.value_counts().to_dict()); print(pd.DataFrame(arows).round(3).to_string())
