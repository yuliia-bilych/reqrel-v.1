"""Builds examples.json for the interface: for each corpus, two sets of 6 requirements = 3 real requirements
plus one PTS variant each (contradiction, duplicate, refinement). Combinations are chosen to have few errors of the method,
so that the demo shows all relation types; remaining errors are kept."""
import json,sys,itertools,random; sys.path.insert(0,'.'); sys.path.insert(0,'data')
from reqrel import retrieval,nli,constraints
from reqrel.cache import cached_scores
from paraphrase_testset_seeds import SEEDS
TITLE = {"uav": "Керування безпілотниками (UAV / Dronology)", "ibmuav": "Безпілотник для спостереження (IBM-UAV)",
         "worldvista": "Медична інформаційна система (WorldVista)", "pure": "Керування опаленням (PURE / THEMAS)",
         "opencoss": "Платформа сертифікації безпеки (OpenCOSS)"}
random.seed(1)
def evaluate(reqs, truth):
    cand=sorted(retrieval.topk_pairs(retrieval.similarity(reqs,"hybrid_cc"),3))
    sc=cached_scores([(reqs[i],reqs[j]) for i,j in cand],"cross-encoder/nli-deberta-v3-base")
    err=0
    for (i,j),s in zip(cand,sc):
        l=constraints.fuse(nli.decide(s),constraints.check(reqs[i],reqs[j]),("numeric",))
        t=truth.get((min(i,j),max(i,j)),'neutral')
        if l!=t: err+=1
    miss=sum(1 for p in truth if p not in set(cand))
    return err+miss
out={}
for c in TITLE:
    ids=[i for i,s in enumerate(SEEDS) if s[0]==c]
    res=[]
    for combo in random.sample(list(itertools.combinations(ids,3)),min(60,len(list(itertools.combinations(ids,3))))):
        reqs=[];truth={}
        for k,i in enumerate(combo):
            s=SEEDS[i]; lab=('contradiction','duplicate','refinement')[k]; v=(s[4],s[2],s[3])[k]
            reqs+=[s[1].replace("  "," "),v]; truth[(2*k,2*k+1)]=lab
        perm=list(range(6)); random.shuffle(perm)
        R=[reqs[p] for p in perm]; inv={p:n for n,p in enumerate(perm)}
        T={(min(inv[a],inv[b]),max(inv[a],inv[b])):l for (a,b),l in truth.items()}
        res.append((evaluate(R,T),combo,R))
    res.sort(key=lambda x:x[0]); used=set(); n=0
    for e,combo,R in res:
        if used & set(combo): continue
        used|=set(combo); n+=1
        out[f"{TITLE[c]}, набір {n}"]=R; print(c,n,'errors',e)
        if n==2: break
json.dump(out,open(os.path.join(ROOT,'examples.json'),'w',encoding='utf-8'),ensure_ascii=False,indent=1)
