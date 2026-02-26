"""
Phase 2: Extract VLM backbone features (feats_A) from GR00T N1.5 for CKNNA.

Architecture:
    GR00T_N1_5 = EagleBackbone (NVEagle: SigLIP2 + Qwen3-1.7B, truncated to 12 layers)
               + FlowmatchingActionHead (DiT)

Extraction point:
    backbone_features from EagleBackbone.forward()
    = eagle_model.hidden_states[12] (POST-norm, after Qwen3 final RMSNorm of truncated model)
    = (B, seq_len, 2048)
    project_to_dim=null means no linear projection is applied (Identity).

    State enters ONLY via the action head (state_proj), NOT the backbone.
    So backbone_features contains pure vision+text features.

Pooling: masked mean-pool over non-padding tokens (using backbone_attention_mask)

Alignment with other models:
    - Same semantic target as StarVLA/SpatialVLA/Pi0/OpenVLA:
      post-VLM condition embedding, before action decoder.
    - POST-norm consistent across all models.
    - Masked mean-pool consistent across all models.
    - No state leakage in feats_A.

Requires:
    transformers==4.51.3, flash-attn, Isaac-GR00T on PYTHONPATH

Usage:
    PYTHONPATH=/home/ubuntu/verl/Isaac-GR00T python extract_features_groot_n15.py \
        --ckpt /home/ubuntu/verl/GR00T-N1.5-Lerobot-SimplerEnv-BridgeV2 \
        --data_dir ./cknna_data \
        --output_dir ./cknna_data/groot-n15-bridge
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image

from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy


def masked_mean_pool(hidden_states, attention_mask):
    """Mean-pool hidden states over valid (non-padding) tokens.

    Matches the StarVLA/SpatialVLA/OpenVLA extraction scripts exactly.

    Args:
        hidden_states: (B, seq_len, D)
        attention_mask: (B, seq_len) int or bool

    Returns:
        pooled: (B, D) float32
    """
    h = hidden_states.float()
    m = attention_mask.unsqueeze(-1).float()
    return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)


def build_bridge_batch(image_np, task_description):
    """Construct an observation dict matching GR00T's Bridge format.

    Uses dummy state values since state enters only the action head, not
    the backbone. The backbone output is identical regardless of state.

    Args:
        image_np: (H, W, 3) uint8 numpy array
        task_description: str

    Returns:
        batch dict ready for Gr00tPolicy.apply_transforms()
    """
    img = image_np[None]  # (1, H, W, C) -- T=1
    zeros = np.zeros((1, 1))  # (T=1, D=1) for each scalar state key
    return {
        "video.image_0": img,
        "state.x": zeros.copy(),
        "state.y": zeros.copy(),
        "state.z": zeros.copy(),
        "state.roll": zeros.copy(),
        "state.pitch": zeros.copy(),
        "state.yaw": zeros.copy(),
        "state.pad": zeros.copy(),
        "state.gripper": zeros.copy(),
        "annotation.human.action.task_description": [task_description],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Extract GR00T N1.5 backbone features for CKNNA."
    )
    parser.add_argument(
        "--ckpt", type=str, required=True,
        help="Path to GR00T N1.5 checkpoint directory (e.g. ShuaiYang03/GR00T-N1.5-Lerobot-SimplerEnv-BridgeV2)",
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Phase 1 data directory (images/, metadata.json)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save feats_A.pt")
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

    print(f"Loading GR00T N1.5: {args.ckpt}")

    data_config = DATA_CONFIG_MAP["bridge"]
    modality_config = data_config.modality_config()
    transforms = data_config.transform()

    policy = Gr00tPolicy(
        model_path=args.ckpt,
        modality_config=modality_config,
        modality_transform=transforms,
        embodiment_tag="oxe",
        device=args.device,
    )
    model = policy.model

    backbone_cfg = model.config.backbone_cfg
    hidden_size = 2048
    print(f"  select_layer: {backbone_cfg['select_layer']}")
    print(f"  project_to_dim: {backbone_cfg['project_to_dim']}")
    print(f"  backbone hidden_size: {hidden_size}")
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

        batch = build_bridge_batch(img, task)

        # The batch has state with 2 dims => _check_state_is_batched returns False
        # => get_action would unsqueeze. Replicate that here.
        from gr00t.model.policy import unsqueeze_dict_values
        batch_unsqueezed = unsqueeze_dict_values(batch)

        normalized_input = policy.apply_transforms(batch_unsqueezed)

        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            backbone_inputs, _ = model.prepare_input(normalized_input)
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
        "model": "GR00T-N1.5-Lerobot-SimplerEnv-BridgeV2",
        "checkpoint_path": args.ckpt,
        "extraction_point": (
            "EagleBackbone.backbone_features = "
            "eagle_model.hidden_states[12] (POST-norm Qwen3 RMSNorm, "
            "truncated 12-layer LLM, no linear projection)"
        ),
        "pooling": "masked_mean_pool (backbone_attention_mask, vision+text tokens)",
        "hidden_size": hidden_size,
        "feats_A_shape": list(feats_A.shape),
        "num_samples": num_samples,
        "state_in_feats_A": False,
        "state_note": "State enters only action_head.state_proj, not backbone",
        "vlm_backbone": "NVEagle (SigLIP2-400M + Qwen3-1.7B, 12 layers)",
    }
    with open(os.path.join(args.output_dir, "extraction_metadata.json"), "w") as f:
        json.dump(extraction_meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n=== GR00T N1.5 Phase 2 Complete ===")
    print(f"  feats_A: {feats_A_path}  shape={tuple(feats_A.shape)}")
    print(f"  Time: {elapsed/60:.1f} min  ({elapsed/num_samples:.2f} s/sample)")


if __name__ == "__main__":
    main()
