#!/usr/bin/env sh
set -eu

export LOCAL_UID="${LOCAL_UID:-$(id -u 2>/dev/null || echo 1000)}"
export LOCAL_GID="${LOCAL_GID:-$(id -g 2>/dev/null || echo 1000)}"

docker compose -f docker-compose.publication.yml up --build publication-test
