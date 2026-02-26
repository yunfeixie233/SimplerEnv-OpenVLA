"""
Phase 2: Extract VLM features (feats_A) from SpatialVLA for CKNNA.

Loads the SpatialVLA checkpoint, runs a forward pass with
output_hidden_states=True on each (image, task) pair from Phase 1 data,
and saves the masked-mean-pooled last hidden state as feats_A.

Architecture: PaLiGemma2 (SigLIP vision + ZoeDepth + Gemma2 LM, hidden_size=2304)
Extraction point: hidden_states[-1] (POST-norm, after final Gemma2 RMSNorm)
Pooling: masked mean-pool over non-padding tokens (matches StarVLA pipeline)

The extraction is purely read-only: no hooks, no model mutation, no side effects.
output_hidden_states=True is a standard HuggingFace parameter that includes
intermediate hidden states in the return value without affecting the computation.

Requires: conda env with transformers>=4.47.0, torch>=2.1

Usage:
    python extract_features_spatialvla.py \
        --ckpt IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge \
        --data_dir ./cknna_data \
        --output_dir ./cknna_data/spatialvla-sft-bridge
"""

import argparse
import json
import os
import time

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


def masked_mean_pool(hidden_states, attention_mask):
    """Mean-pool hidden states over valid (non-padding) tokens.

    Matches the StarVLA extract_features_starvla.py implementation exactly.

    Args:
        hidden_states: (B, seq_len, D)
        attention_mask: (B, seq_len) int or bool

    Returns:
        pooled: (B, D) float32
    """
    h = hidden_states.float()
    m = attention_mask.unsqueeze(-1).float()
    return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Extract SpatialVLA features for CKNNA.")
    parser.add_argument("--ckpt", type=str, default="IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Phase 1 data directory (images/, metadata.json)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save feats_A.pt")
    parser.add_argument("--unnorm_key", type=str, default="bridge_orig/1.0.0",
                        help="Dataset key for intrinsic camera params (ZoeDepth)")
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

    print(f"Loading SpatialVLA: {args.ckpt}")
    processor = AutoProcessor.from_pretrained(args.ckpt, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.ckpt,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).eval().to(args.device)

    hidden_size = model.config.text_config.hidden_size
    print(f"  hidden_size: {hidden_size}")
    print(f"  use_vision_zoe: {getattr(model.config, 'use_vision_zoe', 'N/A')}")
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
        img = Image.open(os.path.join(images_dir, f"{i:06d}.png")).convert("RGB")
        task = task_descriptions[i]

        inputs = processor(
            images=[img],
            text=task,
            unnorm_key=args.unnorm_key,
            return_tensors="pt",
            do_normalize=False,
        )
        inputs = inputs.to(dtype=torch.bfloat16, device=args.device)

        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=True)

        last_hidden = outputs.hidden_states[-1]
        mask = inputs["attention_mask"]
        feat = masked_mean_pool(last_hidden, mask).squeeze(0).cpu()
        feats_list.append(feat)

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            done = i + 1 - args.resume_from
            rate = done / elapsed if elapsed > 0 else 0
            eta = (num_samples - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{num_samples}]  shape=({hidden_size},)  "
                  f"rate={rate:.1f}/s  ETA={eta/60:.1f}min")

        if (i + 1) % args.save_every == 0:
            torch.save(torch.stack(feats_list), partial_path)

    feats_A = torch.stack(feats_list)
    feats_A_path = os.path.join(args.output_dir, "feats_A.pt")
    torch.save(feats_A, feats_A_path)

    if os.path.exists(partial_path):
        os.remove(partial_path)

    extraction_meta = {
        "model": args.ckpt,
        "extraction_point": "hidden_states[-1] (post-norm Gemma2)",
        "pooling": "masked_mean_pool",
        "hidden_size": hidden_size,
        "feats_A_shape": list(feats_A.shape),
        "num_samples": num_samples,
        "unnorm_key": args.unnorm_key,
    }
    with open(os.path.join(args.output_dir, "extraction_metadata.json"), "w") as f:
        json.dump(extraction_meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n=== SpatialVLA Phase 2 Complete ===")
    print(f"  feats_A: {feats_A_path}  shape={tuple(feats_A.shape)}")
    print(f"  Time: {elapsed/60:.1f} min  ({elapsed/num_samples:.2f} s/sample)")


if __name__ == "__main__":
    main()
