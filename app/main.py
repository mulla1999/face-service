"""
Face verification & identification API.

Endpoints
---------
POST   /compare            1:1  — two images, same person or not, with %
POST   /enroll             add a person to the gallery (name + photo)
POST   /identify           1:N  — one image, who is this among enrolled people
GET    /people             list enrolled people
DELETE /people/{id}        remove an enrolled sample
WS     /ws/identify        real-time: stream JPEG frames, get identities back
GET    /health             liveness probe (no auth)

Run
---
    uvicorn app.main:app --host 127.0.0.1 --port 9999 --workers 1

Keep one worker per process: the ONNX session is shared. Scale with more
processes/containers behind a load balancer instead.

Configuration (environment variables)
-------------------------------------
FACE_DB_PATH        SQLite file for enrolled embeddings (default identities.db)
FACE_API_KEY        when set, every endpoint except /health requires it
                    (header X-API-Key; WebSocket: first message)
FACE_CORS_ORIGINS   comma-separated origins allowed to call the API from a
                    browser on another origin (default: none, same origin only)
FACE_MAX_IMAGE_MB   upload size limit in MB (default 10)
FACE_CTX_ID         -1 = CPU (default), 0 = first GPU
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import hmac
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .engine import (
    THRESH_MATCH,
    THRESH_UNCERTAIN,
    DetectedFace,
    Detector,
    FaceEngine,
    cosine_similarity,
    similarity_to_percent,
    verdict,
)
from .store import IdentityStore

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("face-service")

MAX_NAME_LENGTH = 100
MAX_SIDE_PX = 1920          # the detector works at 640px; larger images only cost latency
WS_AUTH_TIMEOUT_S = 5.0
CLIENT_DIR = Path(__file__).resolve().parent.parent / "client"


@dataclass(frozen=True)
class Settings:
    db_path: str = "identities.db"
    api_key: str | None = None
    cors_origins: tuple[str, ...] = ()
    max_image_bytes: int = 10 * 1024 * 1024

    @classmethod
    def from_env(cls) -> Settings:
        origins = tuple(o.strip().rstrip("/") for o in os.getenv("FACE_CORS_ORIGINS", "").split(",") if o.strip())
        return cls(
            db_path=os.getenv("FACE_DB_PATH", "identities.db"),
            api_key=os.getenv("FACE_API_KEY") or None,
            cors_origins=origins,
            max_image_bytes=int(float(os.getenv("FACE_MAX_IMAGE_MB", "10")) * 1024 * 1024),
        )


def create_app(
    settings: Settings | None = None,
    engine: Detector | None = None,
    store: IdentityStore | None = None,
) -> FastAPI:
    """Build the app. Tests pass a fake engine and a temporary store; production loads the models."""
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.engine is None:
            log.info("Loading face models (first run downloads ~300MB)...")
            app.state.engine = await run_in_threadpool(FaceEngine.get)
        owns_store = app.state.store is None
        if owns_store:
            app.state.store = IdentityStore(settings.db_path)
        log.info("Ready. Auth %s.", "on" if settings.api_key else "off (set FACE_API_KEY to require a key)")
        yield
        if owns_store:
            app.state.store.close()
            app.state.store = None

    app = FastAPI(title="Face Identity Service", version="1.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.engine = engine
    app.state.store = store

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["X-API-Key"],
        )

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    def require_key(request: Request) -> None:
        expected = settings.api_key
        if expected and not hmac.compare_digest(request.headers.get("x-api-key", "").encode(), expected.encode()):
            raise HTTPException(401, "Missing or wrong API key (send it in the X-API-Key header)")

    def decode_image(data: bytes) -> np.ndarray:
        if len(data) > settings.max_image_bytes:
            raise HTTPException(413, f"Image too large (max {settings.max_image_bytes // (1024 * 1024)}MB)")
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
        if img is None:
            raise HTTPException(400, "Could not decode image — send JPEG/PNG/WebP")
        h, w = img.shape[:2]
        if max(h, w) > MAX_SIDE_PX:
            scale = MAX_SIDE_PX / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        return img

    async def read_image(upload: UploadFile) -> np.ndarray:
        # read one byte past the limit so oversized uploads are rejected without buffering them whole
        return decode_image(await upload.read(settings.max_image_bytes + 1))

    async def require_face(img: np.ndarray, label: str) -> DetectedFace:
        # inference is CPU-bound: run it off the event loop so other requests and streams keep flowing
        face = await run_in_threadpool(app.state.engine.primary_face, img)
        if face is None:
            raise HTTPException(422, f"No face detected in {label}. Use a clearer, front-facing photo.")
        return face

    def origin_allowed(origin: str | None, host: str | None) -> bool:
        """Block cross-site WebSocket hijacking: browsers send Origin, and only our own page may connect."""
        if not origin:
            return True  # non-browser client
        return urlsplit(origin).netloc == host or origin.rstrip("/") in settings.cors_origins

    auth = [Depends(require_key)]

    # ------------------------------------------------------------------ #
    # 1:1 verification
    # ------------------------------------------------------------------ #

    @app.post("/compare", dependencies=auth)
    async def compare(image_a: UploadFile = File(...), image_b: UploadFile = File(...)):
        t0 = time.perf_counter()
        face_a = await require_face(await read_image(image_a), "image A")
        face_b = await require_face(await read_image(image_b), "image B")
        sim = cosine_similarity(face_a.embedding, face_b.embedding)
        return {
            "verdict": verdict(sim),
            "match_percent": similarity_to_percent(sim),
            "cosine_similarity": round(sim, 4),
            "thresholds": {"match": THRESH_MATCH, "uncertain": THRESH_UNCERTAIN},
            "faces": {
                "a": {"bbox": face_a.bbox, "det_score": round(face_a.det_score, 3)},
                "b": {"bbox": face_b.bbox, "det_score": round(face_b.det_score, 3)},
            },
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    # ------------------------------------------------------------------ #
    # enrollment + 1:N identification
    # ------------------------------------------------------------------ #

    @app.post("/enroll", dependencies=auth)
    async def enroll(name: str = Form(...), image: UploadFile = File(...)):
        name = " ".join(name.split())
        if not name or len(name) > MAX_NAME_LENGTH:
            raise HTTPException(422, f"Name must be 1-{MAX_NAME_LENGTH} characters.")
        face = await require_face(await read_image(image), "enrollment photo")
        if face.det_score < 0.6:
            raise HTTPException(422, "Face too unclear to enroll — use a sharper photo.")
        identity_id = await run_in_threadpool(app.state.store.enroll, name, face.embedding)
        return {"enrolled": True, "name": name, "identity_id": identity_id,
                "tip": "Enroll 2-3 photos per person (different angles/lighting) for best accuracy."}

    @app.post("/identify", dependencies=auth)
    async def identify(image: UploadFile = File(...), top_k: int = Query(3, ge=1, le=10)):
        t0 = time.perf_counter()
        img = await read_image(image)
        faces = await run_in_threadpool(app.state.engine.detect, img)
        if not faces:
            raise HTTPException(422, "No face detected.")
        results = []
        for face in faces:
            matches = app.state.store.search(face.embedding, top_k=top_k)
            best = matches[0] if matches else None
            results.append({
                "bbox": face.bbox,
                "det_score": round(face.det_score, 3),
                "identity": (
                    {
                        "name": best[0],
                        "match_percent": similarity_to_percent(best[2]),
                        "cosine_similarity": round(best[2], 4),
                        "confident": best[2] >= THRESH_MATCH,
                    }
                    if best and best[2] >= THRESH_UNCERTAIN
                    else None  # unknown person
                ),
                "candidates": [{"name": n, "similarity": round(s, 4)} for n, _, s in matches],
            })
        return {"faces": results, "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}

    @app.get("/people", dependencies=auth)
    def people():
        return {"people": app.state.store.list_people()}

    @app.delete("/people/{identity_id}", dependencies=auth)
    def delete_identity(identity_id: str):
        if not app.state.store.remove(identity_id):
            raise HTTPException(404, "Identity not found")
        return {"deleted": identity_id}

    # ------------------------------------------------------------------ #
    # real-time identification over WebSocket
    # ------------------------------------------------------------------ #

    @app.websocket("/ws/identify")
    async def ws_identify(ws: WebSocket):
        """
        Client streams frames as base64 JPEG strings; server replies per frame:
            {"faces": [{"bbox": [...], "name": "...", "match_percent": ...}], "ts": ...}
        When FACE_API_KEY is set, the first message must be the key.
        Send ~3-8 FPS: identity doesn't change 30x/second.
        """
        if not origin_allowed(ws.headers.get("origin"), ws.headers.get("host")):
            await ws.close(code=1008)
            return
        await ws.accept()
        max_frame_chars = settings.max_image_bytes * 4 // 3 + 4
        try:
            if settings.api_key:
                try:
                    key = await asyncio.wait_for(ws.receive_text(), WS_AUTH_TIMEOUT_S)
                except TimeoutError:
                    key = ""
                if not hmac.compare_digest(key.encode(), settings.api_key.encode()):
                    await ws.send_json({"error": "unauthorized"})
                    await ws.close(code=1008)
                    return
                await ws.send_json({"ok": "authorized"})
            while True:
                b64 = await ws.receive_text()
                if len(b64) > max_frame_chars:
                    await ws.send_json({"error": "frame_too_large"})
                    continue
                try:
                    img = decode_image(base64.b64decode(b64, validate=True))
                except (binascii.Error, ValueError, HTTPException):
                    await ws.send_json({"error": "bad_frame"})
                    continue
                faces = await run_in_threadpool(app.state.engine.detect, img, 8)
                payload = []
                for face in faces:
                    matches = app.state.store.search(face.embedding, top_k=1)
                    name, pct = None, None
                    if matches and matches[0][2] >= THRESH_UNCERTAIN:
                        name, pct = matches[0][0], similarity_to_percent(matches[0][2])
                    payload.append({"bbox": [round(v, 1) for v in face.bbox], "name": name, "match_percent": pct})
                await ws.send_json({"faces": payload, "ts": time.time()})
        except WebSocketDisconnect:
            pass

    @app.get("/health")
    def health():
        return {"status": "ok", "auth_required": bool(settings.api_key), "enrolled": app.state.store.count()}

    # serve the web UI (client/) from the same port; mounted last so API routes win
    if CLIENT_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(CLIENT_DIR), html=True), name="ui")

    return app


app = create_app()
