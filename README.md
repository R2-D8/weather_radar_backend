# weather-radar

Radar tile server for the weather app. It downloads the Italian Civil Protection (DPC) rain-rate
radar frames, keeps the last 6 hours, and serves them as small map tiles over HTTP, plus a short
forecast (up to 30 minutes) made by moving the newest frame along its measured motion (averaged over the last two 10-minute steps)
(`src/weather_radar/nowcast.py`; why only 30 minutes: see the backtest in the app's plan).
It also uses DWD's ICON models (ICON-D2, 2 km, for the Alps, northern Italy and central Europe,
faded into ICON-EU, 7 km, elsewhere; `icon.py`, `modeldata.py`) for:

- **hourly temperature maps** (24 h back, 48 h ahead), corrected for the real ground height with
  open terrain tiles (`temperature.py`, `terrain.py`);
- **model rain frames** for the next 12 hours, shown after the radar's 30-minute nowcast
  (`precip.py`);
- **rain or snow**: radar and model tiles can draw snow and mixed precipitation in their own
  colours, from the model snow line and the real ground height (`precip.py`, `palettes.py`).

Design and measurements: the app's plan, Phases 5c and 5d.

The app that uses it is in `../android/weather`; the plan and decisions are in
`../android/weather/docs/plan.md` (Radar architecture, Phase 4). The server is optional for the
app: it is off by default there.

## API

| Request | Answer |
|---|---|
| `GET /frames` | `{"frames": [{"id": 1792791900, "time": "2026-10-23T21:45:00Z"}, ...], "step_seconds": 300, "max_zoom": 8, "tile_size": 256}`, oldest first |
| `GET /tiles/{frame}/{palette}/{z}/{x}/{y}.png` | a map tile; `?size=512` for 512 px tiles. Zoom 0 to 8. |
| `GET /forecast/{base}/{lead}/{palette}/{z}/{x}/{y}.png` | a tile of the short rain forecast made from frame `{base}`, `{lead}` = 10, 20 or 30 minutes ahead. `/frames` lists them under `forecast` (with `forecast_base`) once two frames 10 minutes apart exist. Immutable per base. |
| `GET /coverage?lat=..&lon=..` | `{"covered": true}` when the newest frame has radar data at that point (dry areas count as covered), so the app can decide whether this server is useful for a place |
| `GET /temperature` | hourly temperature maps: `{"hours": [{"id": 1793656800, "time": "...", "run": 1793620800, "version": "1793620800-1793642400"}, ...], "now": ..., "max_zoom": 8, "tile_size": 256, "palette": "temp2c", "unit": "°C", "bands": [{"from": -30.0, "color": "#..."}, ...], "attribution": "..."}`, oldest first. `run` is the ICON-EU run, `version` names the ICON-EU and (if the hour has one) ICON-D2 runs: `{eu}` or `{eu}-{d2}`. |
| `GET /temperature/{version}/{valid}/{palette}/{z}/{x}/{y}.png` | a 256 px temperature tile of hour `{valid}`, 2 °C colour bands, height-corrected. `{version}` as listed (a plain ICON-EU run gives ICON-EU only). Immutable (a newer run has new URLs). 503 if the terrain tile it needs cannot be fetched. |
| `GET /model` | model rain frames: `{"hours": [{"id": 1793656800, "time": "...", "version": "{eu}-{d2}"}, ...], "step_seconds": 3600, "max_zoom": 8, "tile_size": 256, "attribution": "..."}`: the hours after now, up to `RADAR_MODEL_RAIN_HOURS` ahead. Each is the precipitation in the hour **ending** at `id`. |
| `GET /model/{version}/{valid}/{palette}/{z}/{x}/{y}.png` | a 256 px tile of that hour's precipitation (mm in the hour, drawn as mm/h with the radar palettes; a `-snow` palette shows snow and mixed). Immutable. |
| `GET /temperature/point?lat=..&lon=..&valid=..` | `{"temperature": 7.8, "valid": ..., "run": ..., "ground_m": 133}` at a point (`null` outside the model area) |
| `GET /palettes` | colour palettes with their legend bands (rain rate in mm/h). When the server has the snow line: `"snow_suffix": "-snow"` and the `mixed` and `snow` bands. Appending the suffix to a palette name (`classic-snow`) in any rain tile URL draws snow and mixed precipitation in their own colours; if the snow line or terrain is missing for a radar tile, it is drawn as rain with a short cache time. |
| `GET /health` | status, frame count, newest frame age, ingest status, and per model field (`models`): hours stored, newest run, last error; HTTP 503 when starting or stale (radar only) |
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
| `RADAR_TEMPERATURE` | `1` | `0` switches the temperature maps off (no DWD or terrain downloads) |
| `RADAR_TEMPERATURE_PAST_HOURS` / `_FUTURE_HOURS` | `24` / `48` | the hours kept around the current one; about 220 MB a day of downloads with the defaults |
| `RADAR_TEMPERATURE_POLL_SECONDS` | `600` | how often to look for new model runs (all model fields) |
| `RADAR_ICON_D2` | `1` | `0`: ICON-EU only (saves about 295 MB a day of downloads) |
| `RADAR_MODEL_RAIN_HOURS` | `12` | hours of model rain after the radar; `0` switches it off (saves about 145 MB a day) |
| `RADAR_SNOW` | `1` | `0`: no rain/snow colours, no snow line downloads (saves about 65 MB a day) |

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
no docker group, no login shell; "no login" means its shell is `nologin`, it has no password and no
SSH keys, so nobody can sign in as it), with a read-only root filesystem, no capabilities, a memory
limit, and its port published only on the Tailscale address `100.95.20.125`.

1. **Once**, create the user (needs root, so you run it): copy `deploy/create-user.sh` to
   `/tmp/weather-radar-create-user.sh` on the server and run `sudo bash /tmp/weather-radar-create-user.sh`.
   It writes `/tmp/weather-radar-create-user.out`.
2. From the PC: `deploy/deploy.sh` (rsync to `/srv/stacks/weather-radar`, then `docker compose up -d --build`).
3. Check from the PC (Tailscale must be on): `curl http://100.95.20.125:8080/health`.

Operating it (on the server, in `/srv/stacks/weather-radar`): `docker compose logs -f` to watch, `docker compose restart` to restart,
`docker compose down` to stop. The `./data` folder (stored frames) belongs to the service user and
is not readable by `r2d8`; deleting it only costs a re-download of the last 6 hours.

## Data and licence

Radar data: Radar-DPC, Presidenza del Consiglio dei Ministri, Dipartimento della Protezione Civile,
CC BY-SA 4.0. Keep the credit (`/attribution`) visible in the app.

Model data (temperature, precipitation, snow line): Deutscher Wetterdienst (DWD), ICON-D2 and
ICON-EU, CC BY 4.0 (open data server,
https://opendata.dwd.de). Ground height: Mapzen/Tilezen Terrain Tiles on AWS (SRTM, GMTED2010,
ETOPO1, EU-DEM and others; attribution list:
https://github.com/tilezen/joerd/blob/master/docs/attribution.md). Both credits are in the app.

Stored data: `./data/frames` (radar, 6 h), `./data/temperature` (ICON-EU temperature, one float16
grid per hour, about 135 MB), `./data/icon-d2/t_2m` (about 130 MB), `./data/icon-d2/tot_prec` and
`./data/icon-eu/tot_prec` (about 25 MB each), `./data/icon-eu/snowlmt` (about 40 MB),
`./data/terrain` (terrain tiles, kept for good, only areas that were viewed).

Downloads with the defaults, per day: DPC radar about 150 MB, ICON-EU temperature about 220 MB,
ICON-D2 temperature about 185 MB, ICON-D2 rain about 110 MB, ICON-EU snow line about 65 MB,
ICON-EU rain about 35 MB: about 765 MB in total.
