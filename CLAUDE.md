# CLAUDE.md

## Project
Radar tile server for the user's personal weather app (`../android/weather`). Python 3.12+, aiohttp,
numpy, pyproj, Pillow. It ingests Italian DPC SRI rain-rate frames and serves map tiles. See
[README.md](README.md). The plan and decisions live in `../android/weather/docs/plan.md` (Radar
architecture and Phase 4); read them before changing behaviour.

## Rules
1. **Keep docs updated.** When behaviour, API, configuration or deployment changes, update this
   repo's README and `../android/weather/docs/plan.md` (and `docs/server.md` there) in the same piece
   of work.
2. **Ask, don't assume** when there is more than one reasonable option.
3. **Written from scratch.** The old proof of concept was deleted on purpose; only the findings in
   the plan are used.
4. **Server changes are risky.** The server (`ssh r2d8@red8`) is only reachable over Tailscale. Never
   touch its network, Wi-Fi, Tailscale or SSH setup. Claude has no sudo there: privileged steps are
   scripts in `deploy/` that the user copies to `/tmp` and runs with `sudo`. Ask before anything
   destructive, and before the first deploy or any change to what is published.
5. **The service runs as its own unprivileged user** (`weather-radar`, uid/gid 10001, not in the
   sudo or docker groups). Never run it as `r2d8` or root, and never publish its port on all
   interfaces: only on `100.95.20.125`.
6. **Be polite to DPC.** Polls are timed to when the next frame is due (a few tiny requests per 5 minutes), at most 12 frame downloads per poll.
7. **Palettes are immutable.** Phones cache tiles for good; new colours need a new palette name.
8. **Keep the GitHub repo updated.** The repo is public: https://github.com/R2-D8/weather_radar_backend (`origin`, branch `main`). After each finished piece of work run the checks (tests, ruff, mypy), commit with a clear message and `git push`. The user asked for this explicitly, for this repo only. Never commit secrets, tokens or personal data; the Tailscale address `100.95.20.125` is a private address and is fine.
9. The user is new to Android but an experienced programmer: explain Android-specific things, not
   Python basics.

## Commands
- Tests: `./.venv/bin/python -m pytest`
- Lint/format: `./.venv/bin/ruff check . && ./.venv/bin/ruff format --check .`
- Types: `./.venv/bin/mypy`
- Run locally: `RADAR_HISTORY_HOURS=1 ./.venv/bin/python -m weather_radar`
- Deploy: `deploy/deploy.sh` (after `deploy/create-user.sh` has been run once on the server)
- Never use `pkill -f` with a pattern that also appears in your own command line; it kills the shell.
