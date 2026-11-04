#!/usr/bin/env bash
set -euo pipefail
umask 077
cd "$(dirname "$0")/.."
device="${1:-cpu}"
if [[ "$device" != "cpu" && "$device" != "cuda" ]]; then
    echo 'Usage: bash server/run.sh [cpu|cuda]' >&2
    exit 2
fi
exec .venv-sdv/bin/python -m revision.server_run --device "$device"
