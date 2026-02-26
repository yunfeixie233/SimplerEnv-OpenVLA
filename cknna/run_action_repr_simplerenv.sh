#!/bin/bash
# =============================================================================
# Action Representation CKNNA Pipeline for SimplerEnv-OpenVLA Models
# =============================================================================
#
# Phase 2b: Extract action representations (feats_action) for SpatialVLA,
#           Pi0 lerobot, and OpenVLA-7B.
# Phase 3b: Compute CKNNA(feats_A, feats_action) for each model.
#
# Reuses Phase 1 data (symlinked from StarVLA) and Phase 2 feats_A.
#
# Conda environments:
#   - spatialvla_env: for SpatialVLA
#   - pi0fast_env:    for Pi0 lerobot
#   - openvla_env:    for OpenVLA-7B
#
# Usage:
#   bash cknna/run_action_repr_simplerenv.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SIMPLER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK="$(cd "${SIMPLER_ROOT}/.." && pwd)"
DATA_DIR="${SCRIPT_DIR}/cknna_data"
CONDA="${CONDA_ROOT:-${WORK}/conda}"
COMPUTE_CKNNA="${SCRIPT_DIR}/compute_cknna.py"

echo "============================================="
echo "  SimplerEnv Action Repr CKNNA Pipeline"
echo "============================================="

# Verify Phase 1 data
if [ ! -f "${DATA_DIR}/feats_B.pt" ] || [ ! -f "${DATA_DIR}/metadata.json" ]; then
    echo "ERROR: Phase 1 data not found at ${DATA_DIR}/"
    exit 1
fi
echo "Phase 1 data verified."
echo ""


# =========================================================================
# Phase 2b-a: SpatialVLA action representation extraction
# =========================================================================
echo "============================================="
echo "  Phase 2b-a: SpatialVLA action repr"
echo "============================================="

SPATIALVLA_OUTPUT="${DATA_DIR}/spatialvla-sft-bridge"
if [ -f "${SPATIALVLA_OUTPUT}/feats_action.pt" ]; then
    echo "  feats_action.pt already exists, skipping."
else
    source "${CONDA}/bin/activate" spatialvla_env
    python "${SCRIPT_DIR}/extract_action_repr_spatialvla.py" \
        --ckpt IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge \
        --data_dir "${DATA_DIR}" \
        --output_dir "${SPATIALVLA_OUTPUT}" \
        --seed 42
fi
echo ""


# =========================================================================
# Phase 2b-b: Pi0 lerobot action representation extraction
# =========================================================================
echo "============================================="
echo "  Phase 2b-b: Pi0 lerobot action repr"
echo "============================================="

PI0_OUTPUT="${DATA_DIR}/pi0-lerobot-bridge"
if [ -f "${PI0_OUTPUT}/feats_action.pt" ]; then
    echo "  feats_action.pt already exists, skipping."
else
    source "${CONDA}/bin/activate" pi0fast_env
    python "${SCRIPT_DIR}/extract_action_repr_pi0_lerobot.py" \
        --ckpt_path HaomingSong/lerobot-pi0-bridge \
        --data_dir "${DATA_DIR}" \
        --output_dir "${PI0_OUTPUT}" \
        --seed 42
fi
echo ""


# =========================================================================
# Phase 2b-c: OpenVLA-7B action representation extraction
# =========================================================================
echo "============================================="
echo "  Phase 2b-c: OpenVLA-7B action repr"
echo "============================================="

OPENVLA_OUTPUT="${DATA_DIR}/openvla-7b-bridge"
if [ -f "${OPENVLA_OUTPUT}/feats_action.pt" ]; then
    echo "  feats_action.pt already exists, skipping."
else
    source "${CONDA}/bin/activate" openvla_env
    python "${SCRIPT_DIR}/extract_action_repr_openvla.py" \
        --ckpt "${SIMPLER_ROOT}/checkpoints/openvla-7b" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${OPENVLA_OUTPUT}" \
        --seed 42
fi
echo ""


# =========================================================================
# Phase 3b: Compute CKNNA(feats_A, feats_action) per model
# =========================================================================
echo "============================================="
echo "  Phase 3b: CKNNA(VLM, action_repr)"
echo "============================================="

source "${CONDA}/bin/activate" pi0fast_env

for dir in "${SPATIALVLA_OUTPUT}" "${PI0_OUTPUT}" "${OPENVLA_OUTPUT}"; do
    name=$(basename "${dir}")
    fa="${dir}/feats_A.pt"
    fact="${dir}/feats_action.pt"

    if [ ! -f "${fa}" ]; then
        echo "[${name}] feats_A.pt missing, skipping"
        continue
    fi
    if [ ! -f "${fact}" ]; then
        echo "[${name}] feats_action.pt missing, skipping"
        continue
    fi

    echo ""
    echo "--- ${name}: CKNNA(VLM, action_repr) ---"
    python "${COMPUTE_CKNNA}" \
        --feats_A "${fa}" \
        --feats_B "${fact}" \
        --topk 5 10 20 \
        --also_mutual_knn \
        --output "${dir}/cknna_action_repr.json"
done


echo ""
echo "============================================="
echo "  Pipeline complete"
echo "============================================="
echo "Per-model results:"
for dir in "${SPATIALVLA_OUTPUT}" "${PI0_OUTPUT}" "${OPENVLA_OUTPUT}"; do
    name=$(basename "${dir}")
    if [ -f "${dir}/cknna_action_repr.json" ]; then
        echo "  ${name}: ${dir}/cknna_action_repr.json"
    fi
done
