"""Layer 1: classical CV baseline.

Pretrained ResNet18 (ImageNet) feature extraction + logistic regression head.
Heavy torch imports are lazy so that tests and the API run without torch.
"""

import pickle
from pathlib import Path

import numpy as np


class BaselineClassifier:
    """Binary normal-vs-defect classifier over frozen CNN features."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.clf = None
        self.feature_dim: int | None = None

    # ---- features ---------------------------------------------------------
    @staticmethod
    def _build_extractor():
        import torch
        from torchvision import models, transforms

        weights = models.ResNet18_Weights.IMAGENET1K_V1
        backbone = models.resnet18(weights=weights)
        backbone.fc = torch.nn.Identity()
        backbone.eval()
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return backbone, preprocess, torch

    def extract_features(self, image_paths: list[str]) -> np.ndarray:
        backbone, preprocess, torch = self._build_extractor()
        feats = []
        with torch.no_grad():
            for path in image_paths:
                from PIL import Image

                img = Image.open(path).convert("RGB")
                batch = preprocess(img).unsqueeze(0)
                out = backbone(batch)
                feats.append(out.squeeze(0).numpy())
        features = np.stack(feats)
        self.feature_dim = int(features.shape[1])
        return features

    # ---- training / inference --------------------------------------------
    def train(self, image_paths: list[str], labels: list[bool]) -> dict:
        from sklearn.linear_model import LogisticRegression

        features = self.extract_features(image_paths)
        self.clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.clf.fit(features, np.asarray(labels, dtype=int))
        train_acc = float(self.clf.score(features, np.asarray(labels, dtype=int)))
        return {"train_accuracy": train_acc, "n_samples": len(labels)}

    def predict(self, image_paths: list[str]) -> tuple[list[bool], np.ndarray]:
        if self.clf is None:
            raise RuntimeError("Model is not trained or loaded")
        features = self.extract_features(image_paths)
        proba = self.clf.predict_proba(features)[:, 1]
        preds = proba >= self.threshold
        return preds.tolist(), proba

    # ---- persistence -------------------------------------------------------
    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"clf": self.clf, "threshold": self.threshold, "feature_dim": self.feature_dim}, f)
        return path

    @classmethod
    def load(cls, path: Path) -> "BaselineClassifier":
        with path.open("rb") as f:
            state = pickle.load(f)
        instance = cls(threshold=state["threshold"])
        instance.clf = state["clf"]
        instance.feature_dim = state["feature_dim"]
        return instance
