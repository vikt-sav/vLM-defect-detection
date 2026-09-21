"""Full local eval of defect-qwen-vl (v5) on the 419-image eval set via Ollama.

Writes progress to artifacts/eval_progress.jsonl (one line per image) and
artifacts/eval_report_local.json at the end. Designed to run unattended overnight:
per-image errors are logged and skipped; the script survives, reports what it got.
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, r"C:\AI\vlm-defect-benchmark-v5\src")

from defectbench.config import get_settings  # noqa: E402
from defectbench.vlm import OllamaVLM  # noqa: E402

settings = get_settings()
settings.artifacts_dir = Path(r"C:\AI\vlm-defect-benchmark-v5\artifacts")
settings.data_dir = Path(r"C:\AI\vlm-defect-benchmark-v5\data")
ART = settings.artifacts_dir
eval_manifest = ART / "eval_manifest.jsonl"

records = [json.loads(line) for line in eval_manifest.open(encoding="utf-8") if line.strip()]
print(f"eval images: {len(records)}")

vlm = OllamaVLM(settings.ollama_host, "defect-qwen-vl", timeout=600.0)
assert vlm.health(), "Ollama not reachable"
assert vlm.model_available(), "defect-qwen-vl not available"
print("ollama ok, model:", vlm.model)

progress_path = ART / "eval_progress.jsonl"
done_ids = set()
if progress_path.exists():
    for line in progress_path.open(encoding="utf-8"):
        if line.strip():
            done_ids.add(json.loads(line)["idx"])
print(f"resuming: {len(done_ids)} already done")

def norm(x):
    return str(x).strip().lower()

t0 = time.time()
with progress_path.open("a", encoding="utf-8") as prog:
    for rec in records:
        idx = rec["path"]
        if idx in done_ids:
            continue
        entry = {"idx": idx, "category": rec["category"], "is_defect": rec["is_defect"], "defect_type": rec["defect_type"]}
        t1 = time.time()
        try:
            result, raw = vlm.inspect_image(Path(rec["path"]))
            entry["latency"] = round(time.time() - t1, 1)
            entry["raw_ok"] = result is not None
            entry["pred"] = result.to_dict() if result else None
        except Exception as exc:
            entry["error"] = str(exc)[:200]
        prog.write(json.dumps(entry, ensure_ascii=False) + "\n")
        prog.flush()
        done_n = len(done_ids)
        if (done_n + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"{done_n + 1}/{len(records)} done ({elapsed / 60:.0f} min)")

# aggregate
entries = [json.loads(line) for line in progress_path.open(encoding="utf-8") if line.strip()]
ok = [e for e in entries if e.get("pred")]
json_valid = sum(1 for e in entries if e.get("raw_ok")) / max(1, len(entries))
tp = sum(1 for e in ok if e["is_defect"] and e["pred"]["is_defect"])
fp = sum(1 for e in ok if not e["is_defect"] and e["pred"]["is_defect"])
tn = sum(1 for e in ok if not e["is_defect"] and not e["pred"]["is_defect"])
fn = sum(1 for e in ok if e["is_defect"] and not e["pred"]["is_defect"])
precision = tp / (tp + fp) if tp + fp else 0.0
recall = tp / (tp + fn) if tp + fn else 0.0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

exact = sum(
    1
    for e in ok
    if e["is_defect"] and norm(e["pred"]["defect_type"]) == norm(e["defect_type"])
) / max(1, sum(1 for e in ok if e["is_defect"]))
fuzzy = 0
defects = [e for e in ok if e["is_defect"]]
for e in defects:
    p, t = norm(e["pred"]["defect_type"]), norm(e["defect_type"])
    if p == t or (len(t) > 3 and t in p) or (len(p) > 3 and p in t):
        fuzzy += 1
fuzzy = fuzzy / max(1, len(defects))

severity_acc = sum(1 for e in ok if norm(e["pred"].get("severity", "")) == norm(
    "none" if not e["is_defect"] else "major")) / max(1, len(ok))

report = {
    "model": "defect-qwen-vl (v5: Qwen2.5-VL-3B + QLoRA, 6 categories, 504 train records)",
    "n_eval": len(records),
    "n_scored": len(ok),
    "json_validity": round(json_valid, 4),
    "accuracy": round((tp + tn) / max(1, len(ok)), 4),
    "precision": round(precision, 4),
    "recall": round(recall, 4),
    "f1": round(f1, 4),
    "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    "defect_type_exact_acc": round(exact, 4),
    "defect_type_fuzzy_acc": round(fuzzy, 4),
    "severity_acc_v5_scheme": round(severity_acc, 4),
    "avg_latency_s": round(sum(e.get("latency", 0) for e in ok) / max(1, len(ok)), 1),
    "errors": len(entries) - len(ok),
}
(ART / "eval_report_local.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2, ensure_ascii=False))
