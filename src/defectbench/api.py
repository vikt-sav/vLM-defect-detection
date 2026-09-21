"""FastAPI service exposing both inspection layers.

    POST /inspect   -> baseline prediction + VLM structured JSON
    GET  /health    -> service, Ollama and model status
"""

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from defectbench.config import get_settings
from defectbench.vlm import OllamaVLM, VLMResult

app = FastAPI(title="defectbench", version="0.1.0")

_settings = get_settings()
_baseline = None
_vlm: OllamaVLM | None = None


def get_vlm() -> OllamaVLM:
    global _vlm
    if _vlm is None:
        _vlm = OllamaVLM(_settings.ollama_host, _settings.vlm_model, _settings.vlm_timeout)
    return _vlm


def get_baseline():
    global _baseline
    if _baseline is None:
        model_path = _settings.artifacts_dir / "baseline.pkl"
        if not model_path.exists():
            return None
        from defectbench.baseline import BaselineClassifier

        _baseline = BaselineClassifier.load(model_path)
    return _baseline


@app.get("/health")
def health() -> dict:
    vlm = get_vlm()
    ollama_up = vlm.health()
    return {
        "status": "ok",
        "baseline_loaded": get_baseline() is not None,
        "ollama": ollama_up,
        "vlm_model_available": vlm.model_available() if ollama_up else False,
        "vlm_model": vlm.model,
    }


@app.post("/inspect")
async def inspect(
    file: UploadFile = File(...),
    layers: str = Query("baseline,vlm", description="Comma-separated: baseline,vlm"),
    ground_truth_defect_type: str | None = Query(None, description="Optional expected defect type for scoring"),
) -> dict:
    requested = {layer.strip() for layer in layers.split(",") if layer.strip()}
    unknown = requested - {"baseline", "vlm"}
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown layers: {sorted(unknown)}")

    suffix = Path(file.filename or "image.png").suffix or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        result: dict = {"layers": sorted(requested)}

        if "baseline" in requested:
            baseline = get_baseline()
            if baseline is None:
                result["baseline"] = {
                    "error": "baseline model not found; run scripts/train_baseline.py first"
                }
            else:
                preds, proba = baseline.predict([str(tmp_path)])
                result["baseline"] = {"is_defect": bool(preds[0]), "defect_probability": float(proba[0])}

        if "vlm" in requested:
            vlm = get_vlm()
            if not vlm.health():
                result["vlm"] = {"error": "Ollama is unreachable"}
            else:
                vlm_result, raw = vlm.inspect_image(tmp_path)
                result["vlm"] = vlm_result.to_dict() if vlm_result else {
                    "error": "invalid VLM output",
                    "raw": raw,
                }

        if ground_truth_defect_type is not None:
            result["field_match"] = {
                "vlm": _match(result.get("vlm"), ground_truth_defect_type),
            }
        return result
    finally:
        tmp_path.unlink(missing_ok=True)


def _match(vlm_output: dict | None, expected: str) -> bool | None:
    from defectbench.metrics import _norm

    if not vlm_output or "defect_type" not in vlm_output:
        return None
    return _norm(vlm_output["defect_type"]) == _norm(expected)


__all__ = ["VLMResult", "app"]
