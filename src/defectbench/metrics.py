"""Benchmark metrics shared by all layers.

Classification layer: accuracy / precision / recall / F1 for the binary
normal-vs-defect task, plus per-category breakdown.

VLM layers additionally measure structured-output quality: JSON validity,
per-field accuracy against ground truth and character error rate (CER).
"""

import json
from dataclasses import dataclass, field


@dataclass
class ClassificationReport:
    accuracy: float
    precision: float
    recall: float
    f1: float
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    per_category: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "confusion": {"tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn},
            "per_category": {
                k: {m: round(v, 4) for m, v in d.items()} for k, d in self.per_category.items()
            },
        }


def classification_report(y_true: list[bool], y_pred: list[bool], categories: list[str]) -> ClassificationReport:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("y_true and y_pred must be non-empty lists of equal length")

    tp = fp = tn = fn = 0
    per_cat: dict[str, list[tuple[bool, bool]]] = {}
    for truth, pred, cat in zip(y_true, y_pred, categories, strict=True):
        if truth and pred:
            tp += 1
        elif not truth and pred:
            fp += 1
        elif not truth and not pred:
            tn += 1
        else:
            fn += 1
        per_cat.setdefault(cat, []).append((truth, pred))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(y_true)

    cat_report = {}
    for cat, pairs in per_cat.items():
        ctp = sum(1 for t, p in pairs if t and p)
        cfp = sum(1 for t, p in pairs if not t and p)
        cfn = sum(1 for t, p in pairs if t and not p)
        cprec = ctp / (ctp + cfp) if ctp + cfp else 0.0
        crec = ctp / (ctp + cfn) if ctp + cfn else 0.0
        cf1 = 2 * cprec * crec / (cprec + crec) if cprec + crec else 0.0
        cacc = (ctp + sum(1 for t, p in pairs if not t and not p)) / len(pairs)
        cat_report[cat] = {"accuracy": cacc, "precision": cprec, "recall": crec, "f1": cf1}

    return ClassificationReport(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        per_category=cat_report,
    )


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate via Levenshtein distance, normalized by reference length."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    prev = list(range(len(hypothesis) + 1))
    for i, ref_char in enumerate(reference, start=1):
        curr = [i]
        for j, hyp_char in enumerate(hypothesis, start=1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ref_char != hyp_char)))
        prev = curr
    return prev[-1] / len(reference)


def _norm(value: object) -> str:
    return str(value).strip().lower()


def parse_json_output(raw: str) -> dict | None:
    """Parse a model reply into a dict, tolerating markdown fences and prose around JSON."""
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def json_validity_rate(raw_outputs: list[str]) -> float:
    if not raw_outputs:
        raise ValueError("raw_outputs must be non-empty")
    valid = sum(1 for out in raw_outputs if parse_json_output(out) is not None)
    return valid / len(raw_outputs)


def field_accuracy(
    parsed: list[dict | None],
    expected: list[dict],
    field: str,
) -> float:
    """Share of samples where parsed[field] matches expected[field] (case-insensitive)."""
    if len(parsed) != len(expected) or not expected:
        raise ValueError("parsed and expected must be non-empty lists of equal length")
    hits = 0
    for pred, truth in zip(parsed, expected, strict=True):
        if pred is None:
            continue
        if field not in truth:
            continue
        if _norm(pred.get(field)) == _norm(truth[field]):
            hits += 1
    scored = sum(1 for truth in expected if field in truth)
    return hits / scored if scored else 0.0
