#!/usr/bin/env bash
set -Eeuo pipefail
exec python3 -m verl.trainer.main_ppo "$@"
