#!/bin/bash
source /home/ubuntu/verl/conda/bin/activate simpler_env
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=/home/ubuntu/verl/conda/envs/simpler_env/lib:$LD_LIBRARY_PATH
export DISPLAY=:1
export CUDA_VISIBLE_DEVICES=0

RUNS=/home/ubuntu/verl/openvla/runs
PREFIX=openvla-7b+bridge_orig+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug
POLICY=openvla
TEMP=0
LOG_BASE=/tmp/simplerenv_sweep

cd /home/ubuntu/verl/SimplerEnv-OpenVLA

run_task() {
    local ckpt=$1
    local label=$2
    local env=$3
    local robot=$4
    local scene=$5
    local overlay=$6
    local rx=$7
    local ry=$8
    local max_steps=$9

    echo "--- [${label}] ${env} ---"
    python simpler_env/main_inference.py \
      --policy-model $POLICY --ckpt-path $ckpt --action-ensemble-temp $TEMP \
      --logging-dir ${LOG_BASE}/${label} \
      --robot $robot --policy-setup widowx_bridge \
      --control-freq 5 --sim-freq 500 --max-episode-steps $max_steps \
      --env-name $env --scene-name $scene \
      --rgb-overlay-path $overlay \
      --robot-init-x $rx $rx 1 --robot-init-y $ry $ry 1 \
      --obj-variation-mode episode --obj-episode-range 0 24 \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      2>&1 | grep "Average success"
}

for STEPS in 5000 10000 15000; do
    CKPT=${RUNS}/${PREFIX}--${STEPS}_chkpt
    LABEL=step${STEPS}
    echo ""
    echo "=============================="
    echo "  CHECKPOINT: ${STEPS} steps"
    echo "=============================="
    run_task $CKPT $LABEL \
        PutCarrotOnPlateInScene-v0 widowx bridge_table_1_v1 \
        ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png \
        0.147 0.028 60
    run_task $CKPT $LABEL \
        StackGreenCubeOnYellowCubeBakedTexInScene-v0 widowx bridge_table_1_v1 \
        ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png \
        0.147 0.028 60
    run_task $CKPT $LABEL \
        PutSpoonOnTableClothInScene-v0 widowx bridge_table_1_v1 \
        ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png \
        0.147 0.028 60
    run_task $CKPT $LABEL \
        PutEggplantInBasketScene-v0 widowx_sink_camera_setup bridge_table_1_v2 \
        ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png \
        0.127 0.06 120
    echo "  [${STEPS}] DONE"
done

echo ""
echo "=============================="
echo "  ALL CHECKPOINTS DONE"
echo "=============================="
