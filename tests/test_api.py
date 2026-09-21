from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

import defectbench.api as api_module
from defectbench.api import app

client = TestClient(app)


class StubVLM:
    """Stands in for OllamaVLM so tests never touch the network."""

    def __init__(self, healthy: bool = False, model_available: bool = False):
        self._healthy = healthy
        self._model_available = model_available
        self.model = "stub:1b"

    def health(self) -> bool:
        return self._healthy

    def model_available(self) -> bool:
        return self._model_available


def png_bytes(color=(255, 0, 0), size=(32, 32)) -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_health_with_vlm_down(monkeypatch):
    monkeypatch.setattr(api_module, "get_vlm", lambda: StubVLM(healthy=False))
    monkeypatch.setattr(api_module, "get_baseline", lambda: None)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["ollama"] is False
    assert data["vlm_model_available"] is False
    assert data["baseline_loaded"] is False


def test_health_with_vlm_up(monkeypatch):
    monkeypatch.setattr(api_module, "get_vlm", lambda: StubVLM(healthy=True, model_available=True))
    monkeypatch.setattr(api_module, "get_baseline", lambda: None)
    response = client.get("/health")
    data = response.json()
    assert data["ollama"] is True
    assert data["vlm_model_available"] is True
    assert data["baseline_loaded"] is False


def test_inspect_unknown_layer_rejected():
    response = client.post(
        "/inspect",
        files={"file": ("img.png", png_bytes(), "image/png")},
        params={"layers": "baseline,yolo"},
    )
    assert response.status_code == 400


def test_inspect_baseline_missing_model(monkeypatch):
    monkeypatch.setattr(api_module, "get_baseline", lambda: None)
    response = client.post(
        "/inspect",
        files={"file": ("img.png", png_bytes(), "image/png")},
        params={"layers": "baseline"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "error" in data["baseline"]


def test_inspect_baseline_with_model(monkeypatch):
    class StubBaseline:
        def predict(self, paths):
            return [True], [0.93]

    monkeypatch.setattr(api_module, "get_baseline", lambda: StubBaseline())
    response = client.post(
        "/inspect",
        files={"file": ("img.png", png_bytes(), "image/png")},
        params={"layers": "baseline"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["baseline"] == {"is_defect": True, "defect_probability": 0.93}


def test_inspect_vlm_unreachable(monkeypatch):
    monkeypatch.setattr(api_module, "get_vlm", lambda: StubVLM(healthy=False))
    response = client.post(
        "/inspect",
        files={"file": ("img.png", png_bytes(), "image/png")},
        params={"layers": "vlm"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "error" in data["vlm"]


def test_inspect_field_match_param(monkeypatch):
    monkeypatch.setattr(api_module, "get_vlm", lambda: StubVLM(healthy=False))
    response = client.post(
        "/inspect",
        files={"file": ("img.png", png_bytes(), "image/png")},
        params={"layers": "vlm", "ground_truth_defect_type": "good"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["field_match"]["vlm"] is None  # no VLM output to compare
