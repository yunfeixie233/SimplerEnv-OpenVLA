"""
Extract VLM backbone features (feats_A) from GR00T N1.6 for CKNNA.

Architecture:
    Gr00tN1d6 = EagleBackbone (Eagle-Block2A-2B-v2: SigLIP2 + Qwen3-1.7B, truncated to 16 layers)
              + Gr00tN1d6ActionHead (AlternateVLDiT, 32 layers)

Extraction point:
    backbone_features from EagleBackbone.forward()
    = eagle_model.hidden_states[-1] after truncation to 16 layers
    = POST-norm (Qwen3 final RMSNorm applied)
    = (B, seq_len, 2048)

    State enters ONLY via the action head (state_encoder), NOT the backbone.
    So backbone_features contains pure vision+text features.
    NOTE: top 4 LLM layers (12-15) are fine-tuned during training, but inference is the same.

Pooling: masked mean-pool over non-padding tokens (using backbone_attention_mask)

Requires:
    PYTHONPATH must point to gr00t_1p6/Isaac-GR00T

Usage:
    PYTHONPATH=/home/ubuntu/verl/gr00t_1p6/Isaac-GR00T python extract_features_groot_n16.py \
        --ckpt /home/ubuntu/verl/GR00T-N1.6-bridge \
        --data_dir ./cknna_data \
        --output_dir ./cknna_data/groot-n16-bridge
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType, VLAStepData
from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype


def masked_mean_pool(hidden_states, attention_mask):
    h = hidden_states.float()
    m = attention_mask.unsqueeze(-1).float()
    return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)


def build_bridge_obs(image_np, task_description):
    """Build N1.6-format observation dict for a single sample (B=1, T=1).

    State is dummy zeros since it only enters the action head, not the backbone.
    """
    return {
        "video": {
            "image_0": image_np[np.newaxis, np.newaxis],
        },
        "state": {
            "x": np.zeros((1, 1, 1), dtype=np.float32),
            "y": np.zeros((1, 1, 1), dtype=np.float32),
            "z": np.zeros((1, 1, 1), dtype=np.float32),
            "roll": np.zeros((1, 1, 1), dtype=np.float32),
            "pitch": np.zeros((1, 1, 1), dtype=np.float32),
            "yaw": np.zeros((1, 1, 1), dtype=np.float32),
            "pad": np.zeros((1, 1, 1), dtype=np.float32),
            "gripper": np.zeros((1, 1, 1), dtype=np.float32),
        },
        "language": {
            "annotation.human.action.task_description": [[task_description]],
        },
    }


def process_single_obs(policy, obs):
    """Run one observation through the N1.6 processor pipeline.

    Replicates what Gr00tPolicy._get_action does internally up to collation.
    """
    unbatched = policy._unbatch_observation(obs)
    vla_step = policy._to_vla_step_data(unbatched[0])
    messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step}]
    processed = policy.processor(messages)
    collated = policy.collate_fn([processed])
    collated = _rec_to_dtype(collated, dtype=torch.bfloat16)
    return collated["inputs"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to GR00T N1.6 checkpoint directory (e.g. nvidia/GR00T-N1.6-bridge)")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume_from", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=500)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, "metadata.json")) as f:
        metadata = json.load(f)
    num_samples = metadata["num_samples"]
    task_descriptions = metadata["task_descriptions"]
    images_dir = os.path.join(args.data_dir, "images")

    print(f"Loading GR00T N1.6: {args.ckpt}")
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.OXE_WIDOWX,
        model_path=args.ckpt,
        device=args.device,
    )
    model = policy.model
    hidden_size = 2048

    print(f"  select_layer: {model.config.select_layer}")
    print(f"  backbone_embedding_dim: {model.config.backbone_embedding_dim}")
    print(f"  use_alternate_vl_dit: {model.config.use_alternate_vl_dit}")
    print(f"  Samples to process: {num_samples}")

    partial_path = os.path.join(args.output_dir, "feats_A_partial.pt")
    if args.resume_from > 0 and os.path.exists(partial_path):
        partial_tensor = torch.load(partial_path, weights_only=True)
        feats_list = list(partial_tensor[:args.resume_from])
        print(f"  Resuming from sample {args.resume_from} ({len(feats_list)} loaded)")
    else:
        feats_list = []
        args.resume_from = 0

    t0 = time.time()
    for i in range(args.resume_from, num_samples):
        img = np.array(
            Image.open(os.path.join(images_dir, f"{i:06d}.png")).convert("RGB")
        )
        task = task_descriptions[i]

        obs = build_bridge_obs(img, task)
        inputs = process_single_obs(policy, obs)

        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            backbone_inputs, _ = model.prepare_input(inputs)
            backbone_outputs = model.backbone(backbone_inputs)

        backbone_features = backbone_outputs["backbone_features"]
        backbone_mask = backbone_outputs["backbone_attention_mask"]

        feat = masked_mean_pool(backbone_features, backbone_mask).squeeze(0).cpu()
        feats_list.append(feat)

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            done = i + 1 - args.resume_from
            rate = done / elapsed if elapsed > 0 else 0
            eta = (num_samples - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i+1}/{num_samples}]  shape=({hidden_size},)  "
                f"rate={rate:.1f}/s  ETA={eta/60:.1f}min"
            )

        if (i + 1) % args.save_every == 0:
            torch.save(torch.stack(feats_list), partial_path)

    feats_A = torch.stack(feats_list)
    feats_A_path = os.path.join(args.output_dir, "feats_A.pt")
    torch.save(feats_A, feats_A_path)

    if os.path.exists(partial_path):
        os.remove(partial_path)

    extraction_meta = {
        "model": "GR00T-N1.6-bridge",
        "checkpoint_path": args.ckpt,
        "extraction_point": (
            "EagleBackbone.hidden_states[-1] after truncation to 16 layers "
            "(POST-norm Qwen3 RMSNorm, Eagle-Block2A-2B-v2)"
        ),
        "pooling": "masked_mean_pool (backbone_attention_mask, vision+text tokens)",
        "hidden_size": hidden_size,
        "feats_A_shape": list(feats_A.shape),
        "num_samples": num_samples,
        "state_in_feats_A": False,
        "state_note": "State enters only action_head.state_encoder, not backbone",
        "vlm_backbone": "Eagle-Block2A-2B-v2 (SigLIP2 + Qwen3-1.7B, 16 layers, top 4 tuned)",
    }
    with open(os.path.join(args.output_dir, "extraction_metadata.json"), "w") as f:
        json.dump(extraction_meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n=== GR00T N1.6 Backbone Extraction Complete ===")
    print(f"  feats_A: {feats_A_path}  shape={tuple(feats_A.shape)}")
    print(f"  Time: {elapsed/60:.1f} min  ({elapsed/num_samples:.2f} s/sample)")


if __name__ == "__main__":
    main()
