# Face Identity Service

[![CI](https://github.com/mulla1999/face-service/actions/workflows/ci.yml/badge.svg)](https://github.com/mulla1999/face-service/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-009688)
![License: MIT](https://img.shields.io/badge/code-MIT-green)

Face verification (1:1) and real-time identification (1:N) built on
**InsightFace ArcFace** (~99.8% on the LFW benchmark). Nothing is hardcoded per
person: any face becomes a 512-d identity vector, and matching is pure geometry
(cosine similarity), so it copes with age gaps, new looks, glasses, beards and
lighting changes.

- **Compare** two photos: same person, uncertain or different, with a calibrated match %.
- **Enroll** people, then **identify** everyone in a group photo against them (FAISS vector search).
- **Live**: stream a webcam over WebSocket and get named boxes back at ~5 FPS.
- One container serves the API and a web UI; enrolled identities persist in SQLite.

## How identification works

```
photo ──► detect face + 5 landmarks (RetinaFace/SCRFD)
      ──► align & crop to canonical pose
      ──► ArcFace ResNet-50 ──► 512-d embedding (identity signature)
      ──► cosine similarity vs. another embedding (1:1)
          or FAISS nearest-neighbour search over enrolled people (1:N)
      ──► calibrated threshold ──► same_person / uncertain / different_person
```

## Project structure

```
face-service/
├── app/
│   ├── engine.py      # detection -> alignment -> ArcFace embedding + thresholds
│   ├── store.py       # FAISS vector index + SQLite persistence (1:N search)
│   └── main.py        # FastAPI: /compare /enroll /identify /people /ws/identify
├── client/
│   ├── index.html     # upload UI (compare / enroll / identify)
│   └── live.html      # real-time webcam identification
├── tests/             # API, store and calibration tests (fake engine, no model download)
├── Dockerfile         # API + UI image, models baked in, non-root, healthcheck
└── docker-compose.yml # one-command run on 127.0.0.1:9999
```

## Run with Docker (recommended)

```bash
docker compose up --build
```

Then open **http://localhost:9999**. The web UI and API share the port, and
enrolled identities persist in the `face_data` volume. The first build
downloads the ~300MB model pack once and bakes it into the image.

The port is bound to this machine only. To use it from other devices, put an
HTTPS reverse proxy (nginx, Caddy) in front and set `FACE_API_KEY`.

## Run without Docker

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 9999
# the first run downloads the model pack to ~/.insightface
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `FACE_API_KEY` | *(off)* | When set, every endpoint except `/health` requires it in the `X-API-Key` header. The WebSocket expects it as its first message. The web UI has a field for it. |
| `FACE_CORS_ORIGINS` | *(none)* | Comma-separated origins allowed to call the API from another site. The bundled UI is same-origin and needs none. |
| `FACE_DB_PATH` | `identities.db` | SQLite file for enrolled embeddings. |
| `FACE_MAX_IMAGE_MB` | `10` | Upload size limit. |
| `FACE_CTX_ID` | `-1` | `-1` = CPU, `0` = first GPU (needs `onnxruntime-gpu`). |

Keep secrets in the environment or a secrets manager, never in the repository.

## API

### 1:1 — same person or not?
```bash
curl -F "image_a=@old_photo.jpg" -F "image_b=@new_photo.jpg" http://localhost:9999/compare
```
```json
{
  "verdict": "same_person",
  "match_percent": 84.5,
  "cosine_similarity": 0.6212,
  "faces": {"a": {"bbox": [...], "det_score": 0.91}, "b": {...}},
  "latency_ms": 210.4
}
```

### Enroll (2–3 photos per person recommended)
```bash
curl -F "name=Priya" -F "image=@priya1.jpg" http://localhost:9999/enroll
```

### 1:N — who is in this photo?
```bash
curl -F "image=@group.jpg" "http://localhost:9999/identify?top_k=3"
```

### Manage and monitor
```bash
curl http://localhost:9999/people                       # list enrolled people
curl -X DELETE http://localhost:9999/people/{identity_id}
curl http://localhost:9999/health
```
With `FACE_API_KEY` set, add `-H "X-API-Key: $FACE_API_KEY"`.

### Real-time WebSocket
`WS /ws/identify`: send base64 JPEG frames and receive
`{"faces":[{"bbox":[...], "name":"...", "match_percent":...}]}` per frame.
`client/live.html` is a ready-made client.

## Engineering notes

- **Non-blocking inference.** Model calls run in a thread pool, so one slow
  image never stalls other requests or live streams. A lock serialises access
  to the shared ONNX session.
- **Thread-safe index.** Searches and deletes share one lock, and a delete
  rebuilds the FAISS index from SQLite.
- **Input limits.** Uploads are capped before they are buffered, large images
  are downscaled, names are validated and `top_k` is bounded.
- **Browser safety.** Enrolled names are rendered as text, never HTML. The
  WebSocket rejects connections from other sites, and CORS is off unless
  configured.
- **Tests without models.** The test suite swaps in a deterministic fake engine,
  so `pytest` and CI run in seconds without the 300MB download.

```bash
pip install -r requirements-dev.txt
ruff check . && pytest -q
```

## Accuracy and thresholds

- The thresholds in `app/engine.py` (`THRESH_MATCH=0.40`, `THRESH_UNCERTAIN=0.28`)
  are model-level calibration for ArcFace buffalo_l, not per-person values.
  Raise `THRESH_MATCH` for stricter security (fewer false accepts) and lower it
  for a friendlier experience (fewer false rejects).
- The same person across a 10+ year age gap typically still scores 0.45–0.70.
- The hardest cases are identical twins, toddler-vs-adult comparisons, extreme
  profile angles and faces smaller than ~40px.

## Production checklist

- [ ] **GPU**: swap `onnxruntime` for `onnxruntime-gpu`, set `FACE_CTX_ID=0` and
      uncomment the GPU block in `docker-compose.yml` (~10–20x faster).
- [ ] **Scale out**: run several containers behind a load balancer, one uvicorn
      worker each.
- [ ] **Vector DB**: beyond ~1M enrolled vectors, replace FAISS FlatIP with
      Qdrant, pgvector or Milvus (the `store.py` interface stays the same).
- [ ] **TLS**: terminate HTTPS/WSS at a reverse proxy. The live page switches to
      `wss://` automatically.
- [ ] **Liveness / anti-spoofing**: if this gates anything security-sensitive,
      add a liveness model. Embedding similarity alone cannot tell a printed
      photo from a real face.

## Licences

- **Code** in this repository: [MIT](LICENSE).
- **Models**: the InsightFace library is MIT-licensed, but its pretrained model
  packs (including `buffalo_l`, downloaded at build time) are released for
  **non-commercial research use only**. Commercial use needs a licence from
  InsightFace or a different model.

## Legal and ethical use

Face data is **biometric data**. GDPR (EU), BIPA (Illinois), India's DPDP Act
and similar laws generally require **explicit consent** before enrolling or
matching someone's face, plus secure storage and deletion rights. Only enroll
people who have agreed, secure the database volume, and never use a match score
as the sole basis for consequential decisions (access denial, accusations, law
enforcement). Keep a human review step.
