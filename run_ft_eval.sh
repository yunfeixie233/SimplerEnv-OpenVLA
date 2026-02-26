#!/bin/bash
set -e
source /home/ubuntu/verl/conda/bin/activate simpler_env
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=/home/ubuntu/verl/conda/envs/simpler_env/lib:$LD_LIBRARY_PATH
export DISPLAY=:1
export CUDA_VISIBLE_DEVICES=0

CKPT=/home/ubuntu/verl/openvla/runs/openvla-7b+bridge_orig+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--200000_chkpt
LOG_DIR=/tmp/simplerenv_ft_full
POLICY=openvla
TEMP=0

cd /home/ubuntu/verl/SimplerEnv-OpenVLA

echo "=== Task 1/4: PutCarrotOnPlate ==="
python simpler_env/main_inference.py \
  --policy-model $POLICY --ckpt-path $CKPT --action-ensemble-temp $TEMP --logging-dir $LOG_DIR \
  --robot widowx --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 60 \
  --env-name PutCarrotOnPlateInScene-v0 --scene-name bridge_table_1_v1 \
  --rgb-overlay-path ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png \
  --robot-init-x 0.147 0.147 1 --robot-init-y 0.028 0.028 1 \
  --obj-variation-mode episode --obj-episode-range 0 24 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  2>&1 | tail -3
echo ""

echo "=== Task 2/4: StackGreenCubeOnYellowCube ==="
python simpler_env/main_inference.py \
  --policy-model $POLICY --ckpt-path $CKPT --action-ensemble-temp $TEMP --logging-dir $LOG_DIR \
  --robot widowx --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 60 \
  --env-name StackGreenCubeOnYellowCubeBakedTexInScene-v0 --scene-name bridge_table_1_v1 \
  --rgb-overlay-path ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png \
  --robot-init-x 0.147 0.147 1 --robot-init-y 0.028 0.028 1 \
  --obj-variation-mode episode --obj-episode-range 0 24 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  2>&1 | tail -3
echo ""

echo "=== Task 3/4: PutSpoonOnTableCloth ==="
python simpler_env/main_inference.py \
  --policy-model $POLICY --ckpt-path $CKPT --action-ensemble-temp $TEMP --logging-dir $LOG_DIR \
  --robot widowx --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 60 \
  --env-name PutSpoonOnTableClothInScene-v0 --scene-name bridge_table_1_v1 \
  --rgb-overlay-path ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png \
  --robot-init-x 0.147 0.147 1 --robot-init-y 0.028 0.028 1 \
  --obj-variation-mode episode --obj-episode-range 0 24 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  2>&1 | tail -3
echo ""

echo "=== Task 4/4: PutEggplantInBasket ==="
python simpler_env/main_inference.py \
  --policy-model $POLICY --ckpt-path $CKPT --action-ensemble-temp $TEMP --logging-dir $LOG_DIR \
  --robot widowx_sink_camera_setup --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 120 \
  --env-name PutEggplantInBasketScene-v0 --scene-name bridge_table_1_v2 \
  --rgb-overlay-path ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png \
  --robot-init-x 0.127 0.127 1 --robot-init-y 0.06 0.06 1 \
  --obj-variation-mode episode --obj-episode-range 0 24 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  2>&1 | tail -3
echo ""

echo "=== DONE ==="
