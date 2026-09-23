#!/usr/bin/env bash
# Foreground entry point for an allocated cluster job (Slurm/PBS/etc.).
set -euo pipefail

: "${NEQUIP_MODEL_PATH:?Set an absolute path to the compiled NequIP potential}"
: "${IONIC_MD_TEMPERATURE_K:?Set the elevated screening temperature in kelvin}"
export LAMMPS_COMMAND="${LAMMPS_COMMAND:-lmp}"
if [[ "$NEQUIP_MODEL_PATH" != /* || ! -f "$NEQUIP_MODEL_PATH" ]]; then
    printf '%s\n' 'NEQUIP_MODEL_PATH must name an existing absolute file path.' >&2
    exit 1
fi
if ! command -v "$LAMMPS_COMMAND" >/dev/null; then
    printf 'LAMMPS executable not found: %s\n' "$LAMMPS_COMMAND" >&2
    exit 1
fi

matinvent_project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$matinvent_project_dir"
exec python -u main.py \
    "expname=${EXPNAME:-ca_nequip_md}" \
    pipeline=mat_invent model=mattergen_ionic reward=ionic_conductor logger=csv \
    "device=${DEVICE:-cuda:0}" \
    "model.sample_cfg.properties_to_condition_on.chemical_system=${CHEMICAL_SYSTEM:-Ca-P-S}" \
    "$@"
