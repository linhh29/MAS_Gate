#!/usr/bin/env bash
# Compatibility wrapper. MAS-Gate implementation lives in verl/mas_gate_rl.
set -euo pipefail
MAS_GATE_ROOT=${MAS_GATE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)}
exec "${MAS_GATE_ROOT}/verl/mas_gate_rl/run_mas_gate_grpo.sh" "$@"
