"""Layer 2: zero-shot VLM inspection via Ollama.

Sends the image together with a strict instruction to a Qwen2.5-VL model
served by Ollama and validates the structured JSON reply.
"""

import base64
from pathlib import Path

import httpx
from pydantic import BaseModel, Field, ValidationError

SYSTEM_PROMPT = (
    "You are an industrial quality-control inspector. You analyze device and product "
    "images and report defects. Answer strictly in JSON matching the requested schema. "
    "Field meanings: defect_type is a short defect class name in English, or 'good' when "
    "no defect is visible; location describes where the defect is (e.g. 'bottom left, "
    "near the cap') or 'none'; severity is one of 'none', 'minor', 'major', 'critical'; "
    "confidence is a number between 0 and 1."
)

USER_PROMPT = (
    "Inspect this image. Report whether the object is normal or defective, the defect "
    "type, its location, severity and your confidence. Reply with JSON only."
)

RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "defect_type": {"type": "string"},
        "is_defect": {"type": "boolean"},
        "location": {"type": "string"},
        "severity": {"type": "string", "enum": ["none", "minor", "major", "critical"]},
        "confidence": {"type": "number"},
    },
    "required": ["defect_type", "is_defect", "location", "severity", "confidence"],
}


class VLMResult(BaseModel):
    defect_type: str = "unknown"
    is_defect: bool = False
    location: str = "none"
    severity: str = Field(default="none")
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return self.model_dump()


def encode_image(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


class OllamaVLM:
    """Thin async-capable client over the Ollama /api/chat endpoint."""

    def __init__(self, host: str, model: str, timeout: float = 180.0):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def health(self) -> bool:
        try:
            response = httpx.get(f"{self.host}/api/tags", timeout=5.0)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def model_available(self) -> bool:
        try:
            response = httpx.get(f"{self.host}/api/tags", timeout=5.0)
            response.raise_for_status()
        except httpx.HTTPError:
            return False
        names = [m.get("name", "") for m in response.json().get("models", [])]
        return any(n == self.model or n.split(":")[0] == self.model.split(":")[0] for n in names)

    def inspect_image(self, image_path: Path) -> tuple[VLMResult | None, str]:
        """Return (validated result, raw reply). Result is None on invalid output."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_PROMPT,
                    "images": [encode_image(image_path)],
                },
            ],
            "stream": False,
            "format": RESULT_SCHEMA,
            "options": {"temperature": 0.0},
        }
        response = httpx.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
        response.raise_for_status()
        raw: str = response.json()["message"]["content"]
        return self.parse_reply(raw), raw

    @staticmethod
    def parse_reply(raw: str) -> VLMResult | None:
        from defectbench.metrics import parse_json_output

        data = parse_json_output(raw)
        if data is None:
            return None
        if not set(RESULT_SCHEMA["required"]) <= set(data):
            return None  # strict: benchmark counts incomplete replies as invalid
        try:
            return VLMResult(**data)
        except (ValidationError, TypeError):
            return None
