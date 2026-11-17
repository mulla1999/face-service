# ---------------------------------------------------------------- #
# Face Identity Service — API + Web UI on port 9999
# ---------------------------------------------------------------- #
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app \
    && mkdir -p /data && chown app:app /data

WORKDIR /srv

COPY requirements-base.txt requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

USER app

# Pre-download the model pack (~300MB) at build time -> fast cold starts.
# Lands in /home/app/.insightface. Model licence: non-commercial research only (see README).
RUN python -c "from insightface.app import FaceAnalysis; \
    FaceAnalysis(name='buffalo_l', allowed_modules=['detection','recognition']).prepare(ctx_id=-1)"

COPY --chown=app:app app ./app
COPY --chown=app:app client ./client

# identities.db lives in /data so it persists via a volume
ENV FACE_DB_PATH=/data/identities.db \
    PYTHONUNBUFFERED=1
VOLUME ["/data"]

EXPOSE 9999
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD curl -sf http://localhost:9999/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9999", "--workers", "1"]
