#!/usr/bin/env bash
# Creates the unprivileged user the radar backend runs as, and its data directory.
#
# Run on the home server as root:   sudo bash /tmp/weather-radar-create-user.sh
# Safe to run again. It only creates a user, one directory and its permissions; it does not touch
# the network, Tailscale, SSH or any existing user. The result is written to
# /tmp/weather-radar-create-user.out so it can be read without sudo.
set -u
OUT=/tmp/weather-radar-create-user.out
exec > >(tee "$OUT") 2>&1

NAME=weather-radar
ID=10001
STACK=/srv/stacks/weather-radar
DEPLOYER=r2d8

if [ "$(id -u)" -ne 0 ]; then
    echo "Run this with sudo."
    exit 1
fi

if id "$NAME" >/dev/null 2>&1; then
    echo "user $NAME already exists"
else
    # Refuse to reuse the id for something else.
    if getent passwd "$ID" >/dev/null || getent group "$ID" >/dev/null; then
        echo "uid or gid $ID is already taken by another account: stopping, nothing changed"
        exit 1
    fi
    # System account: no password, no login shell, no home directory, own group only.
    useradd --system --uid "$ID" --user-group --no-create-home \
        --home-dir /nonexistent --shell /usr/sbin/nologin "$NAME"
    echo "created user $NAME ($ID)"
fi

# The stack directory belongs to the person who deploys; only ./data belongs to the service.
mkdir -p "$STACK/data"
chown "$DEPLOYER:$DEPLOYER" "$STACK"
chown "$NAME:$NAME" "$STACK/data"
chmod 750 "$STACK/data"

echo
echo "== checks =="
id "$NAME"
getent passwd "$NAME"
if id -nG "$NAME" | tr ' ' '\n' | grep -qxE 'sudo|docker|adm|root'; then
    echo "PROBLEM: $NAME is in a privileged group"
else
    echo "ok: $NAME is not in the sudo, docker, adm or root groups"
fi
if sudo -l -U "$NAME" 2>&1 | grep -qi "not allowed"; then
    echo "ok: $NAME has no sudo rights"
else
    echo "CHECK: sudo -l -U $NAME says:"; sudo -l -U "$NAME" 2>&1
fi
echo
ls -ld "$STACK" "$STACK/data"
echo
echo "done"
