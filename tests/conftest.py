"""Test fixtures: a deterministic fake engine so the API runs without downloading the models."""
from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.engine import DetectedFace, normalize
from app.main import Settings, create_app
from app.store import EMB_DIM, IdentityStore

BLUE, RED, DIM_BLUE = (255, 0, 0), (0, 0, 255), (40, 0, 0)  # BGR


def solid(color: tuple[int, int, int], width: int = 64, height: int = 64) -> np.ndarray:
    img = np.zeros((height, width, 3), np.uint8)
    img[:] = color
    return img


def jpeg(img: np.ndarray) -> bytes:
    return cv2.imencode(".png", img)[1].tobytes()  # lossless, so colors survive the round trip


def side_by_side(*colors: tuple[int, int, int]) -> bytes:
    return jpeg(np.hstack([solid(c) for c in colors]))


def frame_b64(img: np.ndarray) -> str:
    return base64.b64encode(jpeg(img)).decode()


class FakeEngine:
    """Each 64px-wide tile of an image is one 'face'; its colour is its identity.

    A black tile has no face. A dark tile is a face the detector is unsure of.
    """

    def detect(self, image_bgr: np.ndarray, max_faces: int = 16) -> list[DetectedFace]:
        faces = []
        for i in range(image_bgr.shape[1] // 64):
            tile = image_bgr[:, i * 64:(i + 1) * 64]
            mean = tile.reshape(-1, 3).mean(axis=0)
            if mean.max() < 5:
                continue
            emb = np.zeros(EMB_DIM, np.float32)
            emb[:3] = mean
            faces.append(DetectedFace(normalize(emb), [i * 64.0, 0.0, i * 64.0 + 64, 64.0],
                                      0.9 if mean.max() > 100 else 0.4))
        return faces[:max_faces]

    def primary_face(self, image_bgr: np.ndarray) -> DetectedFace | None:
        faces = self.detect(image_bgr, max_faces=4)
        return faces[0] if faces else None


@pytest.fixture
def make_client(tmp_path):
    clients = []

    def make(**settings) -> TestClient:
        store = IdentityStore(str(tmp_path / "identities.db"))
        client = TestClient(create_app(Settings(**settings), engine=FakeEngine(), store=store))
        client.__enter__()
        clients.append((client, store))
        return client

    yield make
    for client, store in clients:
        client.__exit__(None, None, None)
        store.close()


@pytest.fixture
def client(make_client) -> TestClient:
    return make_client()
