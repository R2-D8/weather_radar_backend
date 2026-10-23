FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install .

# Same uid/gid as the dedicated host user that deploy/create-user.sh creates, so files in the
# ./data volume have one owner on both sides. The container never runs as root.
RUN groupadd --system --gid 10001 radar \
    && useradd --system --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin radar \
    && mkdir /data && chown radar:radar /data
USER radar

ENV RADAR_DATA_DIR=/data \
    RADAR_HOST=0.0.0.0 \
    RADAR_PORT=8080
EXPOSE 8080

# /health answers 503 until frames exist and when they are stale; urlopen raises on 503.
HEALTHCHECK --interval=60s --timeout=5s --start-period=90s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=4)"]

CMD ["python", "-m", "weather_radar"]
