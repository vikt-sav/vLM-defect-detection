import pytest

from defectbench.vlm import RESULT_SCHEMA, OllamaVLM, VLMResult


def test_result_schema_fields():
    assert set(RESULT_SCHEMA["required"]) == {"defect_type", "is_defect", "location", "severity", "confidence"}


def test_parse_reply_valid():
    raw = '{"defect_type": "scratch", "is_defect": true, "location": "top", "severity": "minor", "confidence": 0.8}'
    result = OllamaVLM.parse_reply(raw)
    assert isinstance(result, VLMResult)
    assert result.defect_type == "scratch"
    assert result.is_defect is True


def test_parse_reply_fenced():
    raw = '```json\n{"defect_type": "good", "is_defect": false, "location": "none", "severity": "none", "confidence": 0.99}\n```'
    result = OllamaVLM.parse_reply(raw)
    assert result is not None
    assert result.defect_type == "good"


def test_parse_reply_invalid_json():
    assert OllamaVLM.parse_reply("I see a broken bottle") is None


def test_parse_reply_missing_fields():
    assert OllamaVLM.parse_reply('{"defect_type": "crack"}') is None


def test_result_defaults_and_dict():
    result = VLMResult()
    assert result.to_dict()["severity"] == "none"


def test_health_unreachable():
    vlm = OllamaVLM("http://localhost:1", "qwen2.5vl:3b")
    assert vlm.health() is False
    assert vlm.model_available() is False


@pytest.mark.parametrize("model,expected", [("qwen2.5vl:3b", True), ("nonexistent:1b", False)])
def test_model_available(monkeypatch, model, expected):
    import httpx

    payload = {"models": [{"name": "qwen2.5vl:3b"}, {"name": "llama3:8b"}]}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return payload

    monkeypatch.setattr(httpx, "get", lambda *a, **kw: FakeResponse())
    vlm = OllamaVLM("http://fake", model)
    assert vlm.model_available() is expected
