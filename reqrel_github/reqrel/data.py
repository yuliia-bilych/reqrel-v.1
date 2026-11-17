"""Loading of the public requirement-conflict corpora (processed version from the
Gechuyan GitHub repository, see README) into a set of requirements + labelled pairs."""
from __future__ import annotations
import os
from dataclasses import dataclass, field
import pandas as pd

CORPORA = ["uav", "worldvista", "pure", "opencoss", "ibmuav"]
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


@dataclass
class Corpus:
    name: str
    reqs: list[str]                                   # unique requirements, sorted
    conflicts: set[tuple[int, int]] = field(default_factory=set)   # (i<j) labelled Conflict
    labelled: dict[tuple[int, int], int] = field(default_factory=dict)  # (i<j) -> 1 conflict / 0 neutral
    test_pairs: dict[tuple[int, int], int] = field(default_factory=dict)  # official test split
    train_pairs: dict[tuple[int, int], int] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.reqs)

    @property
    def all_pairs(self) -> int:
        return self.n * (self.n - 1) // 2


def _read(path: str) -> pd.DataFrame:
    d = pd.read_csv(path)[["Text1", "Text2", "Class"]]
    d["Text1"] = d["Text1"].astype(str).str.strip()
    d["Text2"] = d["Text2"].astype(str).str.strip()
    return d


def load_corpus(name: str, data_dir: str = DATA_DIR) -> Corpus:
    tr = _read(os.path.join(data_dir, "train", f"{name}.csv"))
    te = _read(os.path.join(data_dir, "test", f"{name}.csv"))
    reqs = sorted(set(tr.Text1) | set(tr.Text2) | set(te.Text1) | set(te.Text2))
    idx = {r: i for i, r in enumerate(reqs)}
    c = Corpus(name, reqs)
    for df, target in ((tr, c.train_pairs), (te, c.test_pairs)):
        for a, b, cls in df.itertuples(index=False):
            i, j = idx[a], idx[b]
            if i == j:
                continue
            p = (min(i, j), max(i, j))
            y = int(cls == "Conflict")
            target[p] = max(target.get(p, 0), y)
            c.labelled[p] = max(c.labelled.get(p, 0), y)
            if y:
                c.conflicts.add(p)
    # a pair must not be in both splits
    for p in list(c.train_pairs):
        if p in c.test_pairs:
            del c.train_pairs[p]
    return c


def load_all() -> dict[str, Corpus]:
    return {n: load_corpus(n) for n in CORPORA}
