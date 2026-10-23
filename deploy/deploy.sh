#!/usr/bin/env bash
# Copies the project to the home server and (re)starts the stack. Run from the PC.
#
# Needs deploy/create-user.sh to have been run once on the server (as root) first.
# The ./data directory on the server belongs to the service user and is never touched.
set -euo pipefail

HOST="${HOST:-r2d8@red8}"
DIR=/srv/stacks/weather-radar
cd "$(dirname "$0")/.."

rsync -a --delete \
    --exclude 'data/' --exclude '.venv/' --exclude '.git/' --exclude 'tests/' \
    --exclude '__pycache__/' --exclude '*.egg-info/' --exclude '.*_cache/' \
    ./ "$HOST:$DIR/"

ssh "$HOST" "cd $DIR && docker compose up -d --build && docker compose ps"
