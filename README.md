# weather-radar

Radar tile server for the weather app. It downloads the Italian Civil Protection (DPC) rain-rate
radar frames, keeps the last 6 hours, and serves them as small map tiles over HTTP.

The app that uses it is in `../android/weather`; the plan and decisions are in
`../android/weather/docs/plan.md` (Radar architecture, Phase 4). The server is optional for the
app: it is off by default there.

## API

| Request | Answer |
|---|---|
| `GET /frames` | `{"frames": [{"id": 1792791900, "time": "2026-10-23T21:45:00Z"}, ...], "step_seconds": 300}`, oldest first |
| `GET /tiles/{frame}/{palette}/{z}/{x}/{y}.png` | a map tile; `?size=512` for 512 px tiles. Zoom 0 to 8. |
| `GET /palettes` | colour palettes with their legend bands (rain rate in mm/h) |
| `GET /health` | status, frame count, newest frame age, ingest status; HTTP 503 when starting or stale |
| `GET /attribution` | the credit the app must show |

A tile of a past frame never changes, so it is sent with `Cache-Control: immutable` and phones can
keep it; the newest frame gets a 60 second cache. Tiles with no rain are one shared 116 byte image.
Changing the colours of an existing palette would leave stale tiles on phones, so give new colours a
new palette name.

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `RADAR_DATA_DIR` | `data` | where frames are stored (`/data` in Docker) |
| `RADAR_HISTORY_HOURS` | `6` | how long frames are kept |
| `RADAR_HOST` / `RADAR_PORT` | `127.0.0.1` / `8080` | listen address (the container uses `0.0.0.0` inside) |
| `RADAR_POLL_SECONDS` | `30` | how often to re-check while a new frame is overdue; otherwise polls are timed to when DPC publishes the next frame (about 1 to 3 tiny requests per 5 minutes) |
| `RADAR_TILE_CACHE_MIB` | `32` | memory for rendered tiles |
| `RADAR_MAX_ZOOM` | `8` | highest zoom served (the DPC grid is 1 km, about zoom 8) |

## Development

```bash
python3 -m venv .venv && ./.venv/bin/pip install -e '.[dev]'
./.venv/bin/python -m pytest          # tests, using a real sample frame in tests/data
./.venv/bin/ruff check . && ./.venv/bin/ruff format --check .
./.venv/bin/mypy
RADAR_HISTORY_HOURS=1 ./.venv/bin/python -m weather_radar   # runs on http://127.0.0.1:8080
```

## Deployment (home server `red8`, over Tailscale)

The service runs in Docker as its own unprivileged user `weather-radar` (uid/gid 10001: no sudo,
no docker group, no login shell), with a read-only root filesystem, no capabilities, a memory
limit, and its port published only on the Tailscale address `100.95.20.125`.

1. **Once**, create the user (needs root, so you run it): copy `deploy/create-user.sh` to
   `/tmp/weather-radar-create-user.sh` on the server and run `sudo bash /tmp/weather-radar-create-user.sh`.
   It writes `/tmp/weather-radar-create-user.out`.
2. From the PC: `deploy/deploy.sh` (rsync to `/srv/stacks/weather-radar`, then `docker compose up -d --build`).
3. Check from the PC (Tailscale must be on): `curl http://100.95.20.125:8080/health`.

## Data and licence

Radar data: Radar-DPC, Presidenza del Consiglio dei Ministri, Dipartimento della Protezione Civile,
CC BY-SA 4.0. Keep the credit (`/attribution`) visible in the app.
