#!/usr/bin/env bash
# Default to the latest unified inter-node quantification; set this variable explicitly to select a legacy mode.
export QUANTIFICATION_INTER_NODE_MODE="${QUANTIFICATION_INTER_NODE_MODE:-unified_baseline}"
# Shared process limits for MAS-Gate only. Source this from every MAS-Gate
# launcher before starting Ray/vLLM/PyTorch children.
#
# Core dumps from many distributed workers can silently fill /tmp after a
# crash, so they are always disabled. Keep the very high container open-file
# limit for vLLM HTTP concurrency and Ray IPC. Unsupported limits are left at
# the container defaults rather than making a launcher fail.
ulimit -c 0

_mas_gate_hard_nofile=$(ulimit -Hn 2>/dev/null || true)
if [[ "${_mas_gate_hard_nofile}" == "unlimited" ]]; then
  ulimit -n 1048576 2>/dev/null || true
elif [[ "${_mas_gate_hard_nofile}" =~ ^[0-9]+$ ]]; then
  ulimit -n "${_mas_gate_hard_nofile}" 2>/dev/null || true
fi

# Do not cap worker count or address space below the container hard limits.
ulimit -u unlimited 2>/dev/null || true
ulimit -v unlimited 2>/dev/null || true
ulimit -f unlimited 2>/dev/null || true

if [[ "${MAS_GATE_PRINT_ULIMITS:-1}" == "1" ]]; then
  echo "[mas_gate_rl limits] core=$(ulimit -c) nofile=$(ulimit -n) nproc=$(ulimit -u) vmem=$(ulimit -v)"
fi
unset _mas_gate_hard_nofile
