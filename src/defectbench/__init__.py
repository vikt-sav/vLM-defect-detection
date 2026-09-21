"""defectbench: three-layer device defect inspection benchmark.

Layer 1: classical CV baseline (pretrained CNN features + linear classifier).
Layer 2: VLM zero-shot (Qwen2.5-VL via Ollama) producing structured JSON.
Layer 3: QLoRA fine-tuned VLM (see notebooks/qlora_colab.ipynb).

All layers are compared on the same reference eval set built from MVTec AD.
"""

__version__ = "0.1.0"
