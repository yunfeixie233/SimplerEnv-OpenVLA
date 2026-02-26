#!/bin/bash
# =============================================================================
# CKNNA Pipeline for SimplerEnv-OpenVLA Models on WidowX Bridge Data
# =============================================================================
#
# Extracts VLM features for SpatialVLA, Pi0 lerobot, and OpenVLA-7B, then
# computes CKNNA scores across all models for cross-codebase comparison.
#
# Phase 1 data (images, feats_B, metadata) is reused from StarVLA via symlink.
# Phase 2 extracts feats_A for each SimplerEnv model.
# Phase 3 computes CKNNA across all available feats_A (SimplerEnv + StarVLA).
#
# Conda environments:
#   - spatialvla_env: transformers 4.47.0 -- for SpatialVLA (PaLiGemma2)
#   - pi0fast_env:    transformers 4.53.3 -- for Pi0 lerobot (PaliGemma)
#   - openvla_env:    transformers 4.40.1, timm 0.9.16 -- for OpenVLA-7B (Llama-2)
#
# Prerequisites:
#   - cknna_data/ symlink -> <WORK>/starVLA/cknna/cknna_data/
#   - StarVLA feats_A already computed (Phase 2 of starVLA pipeline)
#   - conda envs spatialvla_env, pi0fast_env, and openvla_env exist
#
# Usage:
#   bash cknna/run_cknna_simplerenv.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SIMPLER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK="$(cd "${SIMPLER_ROOT}/.." && pwd)"
DATA_DIR="${SCRIPT_DIR}/cknna_data"
CONDA="${CONDA_ROOT:-${WORK}/conda}"
RESULTS_FILE="${DATA_DIR}/cknna_results_all_bridge.json"

echo "============================================="
echo "  SimplerEnv CKNNA Pipeline"
echo "============================================="
echo "Script dir : ${SCRIPT_DIR}"
echo "Data dir   : ${DATA_DIR}"
echo "Results    : ${RESULTS_FILE}"
echo ""

# -- Verify Phase 1 data exists (from StarVLA) --
if [ ! -f "${DATA_DIR}/feats_B.pt" ] || [ ! -f "${DATA_DIR}/metadata.json" ]; then
    echo "ERROR: Phase 1 data not found at ${DATA_DIR}/"
    echo "Expected: feats_B.pt, metadata.json, images/"
    echo "Make sure the symlink to starVLA/cknna/cknna_data exists."
    exit 1
fi

NUM_SAMPLES=$(python3 -c "import json; print(json.load(open('${DATA_DIR}/metadata.json'))['num_samples'])")
echo "Phase 1 data verified: ${NUM_SAMPLES} samples (symlinked from StarVLA)."
echo ""


# =========================================================================
# Phase 2a: SpatialVLA feature extraction
# =========================================================================
echo "============================================="
echo "  Phase 2a: SpatialVLA  (spatialvla_env)"
echo "============================================="

SPATIALVLA_OUTPUT="${DATA_DIR}/spatialvla-sft-bridge"
if [ -f "${SPATIALVLA_OUTPUT}/feats_A.pt" ]; then
    echo "  feats_A.pt already exists at ${SPATIALVLA_OUTPUT}/, skipping."
else
    echo "  Activating spatialvla_env (transformers 4.47.0)"
    source "${CONDA}/bin/activate" spatialvla_env
    python "${SCRIPT_DIR}/extract_features_spatialvla.py" \
        --ckpt IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge \
        --data_dir "${DATA_DIR}" \
        --output_dir "${SPATIALVLA_OUTPUT}" \
        --unnorm_key "bridge_orig/1.0.0"
fi
echo ""


# =========================================================================
# Phase 2b: Pi0 lerobot feature extraction
# =========================================================================
echo "============================================="
echo "  Phase 2b: Pi0 lerobot  (pi0fast_env)"
echo "============================================="

PI0_OUTPUT="${DATA_DIR}/pi0-lerobot-bridge"
if [ -f "${PI0_OUTPUT}/feats_A.pt" ]; then
    echo "  feats_A.pt already exists at ${PI0_OUTPUT}/, skipping."
else
    echo "  Activating pi0fast_env (transformers 4.53.3 custom fork)"
    source "${CONDA}/bin/activate" pi0fast_env
    python "${SCRIPT_DIR}/extract_features_pi0_lerobot.py" \
        --ckpt_path HaomingSong/lerobot-pi0-bridge \
        --data_dir "${DATA_DIR}" \
        --output_dir "${PI0_OUTPUT}"
fi
echo ""


# =========================================================================
# Phase 2c: OpenVLA-7B feature extraction
# =========================================================================
echo "============================================="
echo "  Phase 2c: OpenVLA-7B  (openvla_env)"
echo "============================================="

OPENVLA_OUTPUT="${DATA_DIR}/openvla-7b-bridge"
if [ -f "${OPENVLA_OUTPUT}/feats_A.pt" ]; then
    echo "  feats_A.pt already exists at ${OPENVLA_OUTPUT}/, skipping."
else
    echo "  Activating openvla_env (transformers 4.40.1, timm 0.9.16)"
    source "${CONDA}/bin/activate" openvla_env
    python "${SCRIPT_DIR}/extract_features_openvla.py" \
        --ckpt "${SIMPLER_ROOT}/checkpoints/openvla-7b" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${OPENVLA_OUTPUT}"
fi
echo ""


# =========================================================================
# Phase 3: Compute CKNNA across ALL models
# =========================================================================
echo "============================================="
echo "  Phase 3: Compute CKNNA (all models)"
echo "============================================="

FEATS_A_PATHS=""

# SimplerEnv models
for dir in "${DATA_DIR}/spatialvla-sft-bridge" "${DATA_DIR}/pi0-lerobot-bridge" "${DATA_DIR}/openvla-7b-bridge"; do
    if [ -f "${dir}/feats_A.pt" ]; then
        FEATS_A_PATHS="${FEATS_A_PATHS} ${dir}/feats_A.pt"
        echo "  $(basename ${dir})"
    else
        echo "  $(basename ${dir}) -- MISSING, skipped"
    fi
done

if [ -z "${FEATS_A_PATHS}" ]; then
    echo "ERROR: No feats_A.pt files found."
    exit 1
fi
echo ""

source "${CONDA}/bin/activate" pi0fast_env
python "${SCRIPT_DIR}/compute_cknna.py" \
    --feats_A ${FEATS_A_PATHS} \
    --feats_B "${DATA_DIR}/feats_B.pt" \
    --topk 5 10 20 \
    --also_mutual_knn \
    --output "${RESULTS_FILE}"


echo ""
echo "============================================="
echo "  Pipeline complete"
echo "============================================="
echo "Results: ${RESULTS_FILE}"
echo ""
echo "Models evaluated:"
for f in ${FEATS_A_PATHS}; do
    echo "  - $(basename $(dirname ${f}))"
done
