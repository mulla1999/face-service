"""
Face engine: detection -> alignment -> ArcFace embedding -> cosine similarity.

Uses the InsightFace 'buffalo_l' model pack:
  - RetinaFace (SCRFD) detector  : finds faces + 5 landmarks
  - ArcFace ResNet-50 recognizer : 512-d identity embedding
Nothing is hardcoded per person. The network was trained on millions of
identities and generalizes to unseen faces. Identity = direction of the
512-d vector; similarity = cosine between vectors.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class DetectedFace:
    embedding: np.ndarray        # L2-normalized 512-d vector
    bbox: list[float]            # [x1, y1, x2, y2]
    det_score: float             # detector confidence 0..1


class Detector(Protocol):
    """What the API needs from an engine (lets tests use a fake without the models)."""

    def detect(self, image_bgr: np.ndarray, max_faces: int = 16) -> list[DetectedFace]: ...

    def primary_face(self, image_bgr: np.ndarray) -> DetectedFace | None: ...


class FaceEngine:
    """Thread-safe singleton wrapper around InsightFace."""

    _instance: FaceEngine | None = None
    _lock = threading.Lock()

    def __init__(self, det_size: tuple[int, int] = (640, 640), ctx_id: int | None = None):
        from insightface.app import FaceAnalysis  # heavy import: only when the real engine loads

        # ctx_id=-1 -> CPU; 0 -> first GPU (requires onnxruntime-gpu)
        ctx_id = int(os.getenv("FACE_CTX_ID", "-1")) if ctx_id is None else ctx_id
        self.analyzer = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"])
        self.analyzer.prepare(ctx_id=ctx_id, det_size=det_size)
        self._infer_lock = threading.Lock()

    @classmethod
    def get(cls) -> FaceEngine:
        with cls._lock:
            if cls._instance is None:
                cls._instance = FaceEngine()
            return cls._instance

    # ------------------------------------------------------------------ #

    def detect(self, image_bgr: np.ndarray, max_faces: int = 16) -> list[DetectedFace]:
        """Detect all faces in a BGR image and return normalized embeddings, largest first."""
        with self._infer_lock:  # one inference at a time on the shared ONNX session
            faces = self.analyzer.get(image_bgr, max_num=max_faces)
        out = [
            DetectedFace(
                embedding=normalize(f.embedding),
                bbox=[float(v) for v in f.bbox],
                det_score=float(f.det_score),
            )
            for f in faces
        ]
        out.sort(key=face_area, reverse=True)  # largest face first (most likely the subject)
        return out

    def primary_face(self, image_bgr: np.ndarray) -> DetectedFace | None:
        faces = self.detect(image_bgr, max_faces=4)
        return faces[0] if faces else None


def normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    return vec / (np.linalg.norm(vec) + 1e-10)


def face_area(face: DetectedFace) -> float:
    x1, y1, x2, y2 = face.bbox
    return (x2 - x1) * (y2 - y1)


# ---------------------------------------------------------------------- #
# Similarity & calibration
# ---------------------------------------------------------------------- #

# Cosine-similarity decision thresholds for ArcFace buffalo_l.
# These are model-level calibration constants (like a ruler's markings),
# not per-person hardcoding. Tune on your own data if you have it:
# raise MATCH for fewer false accepts, lower it for fewer false rejects.
THRESH_MATCH = 0.40       # >= this  -> same person (high confidence)
THRESH_UNCERTAIN = 0.28   # between  -> borderline / ask for better photo


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # inputs are already L2-normalized


def similarity_to_percent(sim: float) -> float:
    """
    Map cosine similarity to a human-friendly 0-100% scale.

    Empirically for ArcFace: genuine pairs cluster ~0.45-0.85,
    impostor pairs cluster ~ -0.1-0.25. We map [-0.1, 0.9] -> [0, 100]
    with a piecewise curve so that THRESH_UNCERTAIN lands at 45% and
    THRESH_MATCH at 70%.
    """
    if sim <= THRESH_UNCERTAIN:
        p = np.interp(sim, [-0.10, THRESH_UNCERTAIN], [0.0, 45.0])          # impostor zone
    elif sim <= THRESH_MATCH:
        p = np.interp(sim, [THRESH_UNCERTAIN, THRESH_MATCH], [45.0, 70.0])  # uncertain zone
    else:
        p = np.interp(sim, [THRESH_MATCH, 0.90], [70.0, 100.0])             # genuine zone
    return round(float(np.clip(p, 0.0, 100.0)), 2)


def verdict(sim: float) -> str:
    if sim >= THRESH_MATCH:
        return "same_person"
    if sim >= THRESH_UNCERTAIN:
        return "uncertain"
    return "different_person"
