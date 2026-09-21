"""Run the three-layer benchmark on the reference eval set.

Layers:
- baseline: ResNet18+logreg (artifacts/baseline.pkl, optional)
- vlm: zero-shot Qwen2.5-VL via Ollama (optional, needs a running Ollama)

Writes artifacts/benchmark_report.json and prints a markdown comparison table.
"""

import argparse
import json
import time
from pathlib import Path

from defectbench.config import get_settings
from defectbench.dataset import load_manifest
from defectbench.metrics import (
    classification_report,
    field_accuracy,
    json_validity_rate,
)


def _fuzzy_defect_match(pred: dict | None, expected_type: str) -> bool:
    """Exact match, or containment match either way (e.g. 'scratch' vs 'scratches')."""
    from defectbench.metrics import _norm

    if pred is None:
        return False
    p, e = _norm(pred.get("defect_type")), _norm(expected_type)
    return p == e or (len(e) > 3 and e in p) or (len(p) > 3 and p in e)


def run_baseline(manifest, settings) -> dict | None:
    model_path = settings.artifacts_dir / "baseline.pkl"
    if not model_path.exists():
        print("[baseline] model not found, skipping (run scripts/train_baseline.py)")
        return None
    from defectbench.baseline import BaselineClassifier

    model = BaselineClassifier.load(model_path)
    paths = [s.path for s in manifest]
    truths = [s.is_defect for s in manifest]

    start = time.perf_counter()
    preds, _ = model.predict(paths)
    elapsed = time.perf_counter() - start

    report = classification_report(truths, preds, [s.category for s in manifest]).to_dict()
    report["avg_latency_s"] = round(elapsed / len(paths), 3)
    print(f"[baseline] accuracy={report['accuracy']} f1={report['f1']} latency={report['avg_latency_s']}s/img")
    return report


def run_vlm(manifest, settings, max_samples: int | None) -> dict | None:
    from defectbench.vlm import OllamaVLM

    vlm = OllamaVLM(settings.ollama_host, settings.vlm_model, settings.vlm_timeout)
    if not vlm.health():
        print("[vlm] Ollama unreachable, skipping layer 2")
        return None
    if not vlm.model_available():
        print(f"[vlm] model '{settings.vlm_model}' not pulled, run: ollama pull {settings.vlm_model}")
        return None

    samples = manifest if max_samples is None else manifest[:max_samples]
    truths, preds, parsed, raws, latencies = [], [], [], [], []
    for i, sample in enumerate(samples, start=1):
        start = time.perf_counter()
        try:
            result, raw = vlm.inspect_image(Path(sample.path))
        except Exception as exc:
            print(f"[vlm] {i}/{len(samples)} request failed: {exc}")
            continue
        latencies.append(time.perf_counter() - start)
        raws.append(raw)
        parsed.append(result.to_dict() if result else None)
        preds.append(bool(result.is_defect) if result else False)
        truths.append(sample.is_defect)
        status = "ok" if result else "INVALID JSON"
        print(f"[vlm] {i}/{len(samples)} {sample.category}/{sample.defect_type} -> {status} ({latencies[-1]:.1f}s)")

    if not raws:
        print("[vlm] no successful responses")
        return None

    report = classification_report(truths, preds, [s.category for s in samples[: len(raws)]]).to_dict()
    report["avg_latency_s"] = round(sum(latencies) / len(latencies), 2)
    report["n_scored"] = len(raws)
    report["json_validity"] = round(json_validity_rate(raws), 4)

    expected_types = [s.defect_type for s in samples[: len(raws)]]
    expected_dicts = [{"defect_type": t} for t in expected_types]
    report["defect_type_exact_acc"] = round(field_accuracy(parsed, expected_dicts, "defect_type"), 4)
    report["defect_type_fuzzy_acc"] = round(
        sum(1 for p, t in zip(parsed, expected_types, strict=True) if _fuzzy_defect_match(p, t)) / len(expected_types),
        4,
    )
    print(
        f"[vlm] accuracy={report['accuracy']} f1={report['f1']} json={report['json_validity']} "
        f"defect_type(fuzzy)={report['defect_type_fuzzy_acc']} latency={report['avg_latency_s']}s/img"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-vlm-samples", type=int, default=30, help="Cap VLM requests (slow on CPU)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    settings = get_settings()
    manifest = load_manifest(settings.artifacts_dir / "eval_manifest.jsonl")
    if not manifest:
        raise SystemExit("Eval manifest is empty. Run scripts/build_eval_set.py first.")
    print(f"Eval manifest: {len(manifest)} images")

    report = {
        "dataset": {"n_images": len(manifest), "categories": sorted({s.category for s in manifest})},
        "baseline": run_baseline(manifest, settings),
        "vlm": run_vlm(manifest, settings, args.max_vlm_samples),
    }
    if report["vlm"]:
        report["vlm"]["model"] = settings.vlm_model

    out_path = args.out or settings.artifacts_dir / "benchmark_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved: {out_path}")

    print("\n| layer | accuracy | f1 | json | defect_type | s/img |")
    print("|---|---|---|---|---|---|")
    for name, key in [("baseline (CV)", "baseline"), ("vlm", "vlm")]:
        layer = report[key]
        if layer:
            extra = f" ({layer.get('model', '')})" if layer.get("model") else ""
            print(
                f"| {name}{extra} | {layer['accuracy']} | {layer['f1']} | "
                f"{layer.get('json_validity', '-')} | {layer.get('defect_type_fuzzy_acc', '-')} | {layer['avg_latency_s']} |"
            )


if __name__ == "__main__":
    main()
