"""Stage 3: explicit checks of numbers/units, modality and negation for a candidate pair,
and fusion with the NLI label. Rule-based, deterministic, applied only to Stage-1 candidates."""
from __future__ import annotations
import re
from dataclasses import dataclass

WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
            "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30,
            "sixty": 60, "hundred": 100, "thousand": 1000}
UPPER = r"(?:less than|fewer than|under|below|within|at most|up to|no more than|not exceed(?:ing)?|maximum of|max(?:imum)?)"
LOWER = r"(?:at least|atleast|higher than|longer than|more than|greater than|over|above|no less than|minimum of|min(?:imum)?)"
NUM = r"\b(?:\d+(?:[.,]\d+)?|" + "|".join(WORD_NUM) + r")\b"
QTY = re.compile(rf"(?:(?P<cmp>{UPPER}|{LOWER})\s+)?(?P<num>{NUM})\s*(?P<unit>%|[a-zA-Z]+\b)?", re.I)
UNIT_ALIASES = {"hour": "h", "hours": "h", "hr": "h", "hrs": "h", "h": "h",
                "minute": "min", "minutes": "min", "min": "min", "mins": "min",
                "second": "s", "seconds": "s", "sec": "s", "secs": "s", "s": "s",
                "millisecond": "ms", "milliseconds": "ms", "ms": "ms",
                "day": "d", "days": "d", "percent": "%", "%": "%"}
STRONG = re.compile(r"\b(shall|must|will|is required to|are required to|has to|have to)\b", re.I)
WEAK = re.compile(r"\b(should|may|can|could|might|is allowed to|optionally)\b", re.I)
NEG = re.compile(r"\b(not|never|no|cannot|can't|won't|neither|nor|without)\b|n't\b", re.I)


@dataclass
class Quantity:
    value: float
    unit: str
    bound: str  # "<=", ">=", "="


def quantities(text: str) -> list[Quantity]:
    out = []
    for m in QTY.finditer(text):
        raw = m.group("num").lower().replace(",", ".")
        val = float(WORD_NUM[raw]) if raw in WORD_NUM else float(raw)
        unit = (m.group("unit") or "").lower()
        unit = UNIT_ALIASES.get(unit, unit)
        cmp_ = (m.group("cmp") or "").lower()
        bound = "<=" if re.fullmatch(UPPER, cmp_ or "x", re.I) else ">=" if re.fullmatch(LOWER, cmp_ or "x", re.I) else "="
        out.append(Quantity(val, unit, bound))
    return out


def modality(text: str) -> str:
    if STRONG.search(text):
        return "strong"
    if WEAK.search(text):
        return "weak"
    return "none"


def negated(text: str) -> bool:
    return bool(NEG.search(text))


UNITS = {"h", "min", "s", "ms", "d", "%", "miles", "mile", "km", "kilometers", "kilometres", "m", "meters",
         "metres", "feet", "ft", "knots", "kg", "g", "hz", "khz", "db", "mb", "gb", "kb", "users", "uavs",
         "drones", "units", "orders", "entries", "characters", "levels", "story", "stories", "degrees", "v",
         "years", "months", "weeks", "times", "attempts", "seconds", "percent"}


def interval_relation(x: Quantity, y: Quantity) -> str:
    """Logical relation between two constraints on the same quantity:
    'contradiction' (no common value), 'refinement' (one implies the other) or 'compatible'."""
    if x.bound == "=" and y.bound == "=":
        return "contradiction" if x.value != y.value else "compatible"
    if x.bound == y.bound:                                   # both upper or both lower bounds
        # logically one bound implies the other, but in requirements two different limits of the same
        # parameter may be a capability vs. a limit; such cases are left to the NLI model
        return "compatible"
    if {x.bound, y.bound} == {"<=", ">="}:
        lo, up = (x, y) if x.bound == ">=" else (y, x)
        return "contradiction" if lo.value > up.value else "compatible"
    eq, bd = (x, y) if x.bound == "=" else (y, x)            # equality vs bound
    if bd.bound == "<=":
        return "contradiction" if eq.value > bd.value else "refinement"
    return "contradiction" if eq.value < bd.value else "refinement"


def numeric_relation(qa: list[Quantity], qb: list[Quantity]) -> str:
    """Compares the constraints of two requirements unit by unit; identical constraints are matched
    first, a single remaining constraint per side is compared by interval logic."""
    worst = "compatible"
    for u in {q.unit for q in qa} & {q.unit for q in qb}:
        if u not in UNITS:
            continue
        xa = [(q.value, q.bound) for q in qa if q.unit == u]; xb = [(q.value, q.bound) for q in qb if q.unit == u]
        ra = [q for q in xa if q not in xb]; rb = [q for q in xb if q not in xa]
        if len(ra) == 1 and len(rb) == 1:
            rel = interval_relation(Quantity(ra[0][0], u, ra[0][1]), Quantity(rb[0][0], u, rb[0][1]))
            if rel == "contradiction":
                return "contradiction"
            if rel == "refinement":
                worst = "refinement"
    return worst


@dataclass
class ConstraintFlags:
    numeric_mismatch: bool
    quantity_added: bool
    modality_mismatch: bool
    negation_mismatch: bool
    numeric_refinement: bool = False


def check(a: str, b: str) -> ConstraintFlags:
    qa, qb = quantities(a), quantities(b)
    rel = numeric_relation(qa, qb)
    mismatch = rel == "contradiction"
    added = (len([q for q in qa if q.unit in UNITS]) == 0) != (len([q for q in qb if q.unit in UNITS]) == 0)
    ma, mb = modality(a), modality(b)
    mod_mis = {ma, mb} == {"strong", "weak"}
    neg_mis = negated(a) != negated(b)
    return ConstraintFlags(mismatch, added, mod_mis, neg_mis, rel == "refinement")


ALL_RULES = ("numeric", "negation", "modality", "added")


def fuse(nli_label: str, f: ConstraintFlags, rules=ALL_RULES) -> str:
    """Stage-3 correction of the Stage-2 label (rules of Table 2.1; `rules` selects a subset)."""
    if nli_label == "neutral":          # Stage 3 corrects the type of a relation, it never creates one
        return nli_label
    if "numeric" in rules and f.numeric_mismatch:
        return "contradiction"
    if nli_label in ("duplicate", "refinement") and (("negation" in rules and f.negation_mismatch) or
                                                    ("modality" in rules and f.modality_mismatch)):
        return "contradiction"
    if "added" in rules and nli_label == "duplicate" and f.quantity_added:
        return "refinement"
    return nli_label
