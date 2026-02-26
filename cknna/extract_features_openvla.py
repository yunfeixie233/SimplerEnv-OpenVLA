"""
Phase 2: Extract VLM features (feats_A) from OpenVLA-7B for CKNNA.

Loads the OpenVLA checkpoint, runs a forward pass with
output_hidden_states=True on each (image, task) pair from Phase 1 data,
and saves the masked-mean-pooled last hidden state as feats_A.

Architecture: Prismatic VLM (DINOv2 + SigLIP fused vision + Llama-2-7b LM, hidden_size=4096)
Extraction point: hidden_states[-1] (POST-norm, after final Llama-2 RMSNorm)
Pooling: masked mean-pool over non-padding tokens in the multimodal sequence
         (includes both projected vision patches and text tokens)

The multimodal sequence is: [BOS, vision_patches, text_tokens_after_BOS]
where vision patches are always valid (mask=1). The multimodal_attention_mask
is reconstructed from the original attention_mask since it is built internally
by PrismaticForConditionalGeneration.forward().

Prompt format: "In: What action should the robot take to {instruction}?\\nOut:"

Requires: conda env with transformers==4.40.1, timm==0.9.16, torch>=2.1

Usage:
    python extract_features_openvla.py \
        --ckpt $WORK/SimplerEnv-OpenVLA/checkpoints/openvla-7b \
        --data_dir ./cknna_data \
        --output_dir ./cknna_data/openvla-7b-bridge
"""

import argparse
import json
import os
import time

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


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
    parser = argparse.ArgumentParser(description="Phase 2: Extract OpenVLA features for CKNNA.")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_ckpt = os.path.join(script_dir, "..", "checkpoints", "openvla-7b")
    parser.add_argument("--ckpt", type=str,
                        default=default_ckpt,
                        help="HuggingFace model ID or local path")
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

    print(f"Loading OpenVLA: {args.ckpt}")
    processor = AutoProcessor.from_pretrained(args.ckpt, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.ckpt,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval().to(args.device)

    hidden_size = model.config.text_config.hidden_size
    print(f"  hidden_size: {hidden_size}")
    print(f"  vision_backbone_id: {model.config.vision_backbone_id}")
    print(f"  llm_backbone_id: {model.config.llm_backbone_id}")
    print(f"  use_fused_vision_backbone: {model.config.use_fused_vision_backbone}")
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

        prompt = f"In: What action should the robot take to {task}?\nOut:"

        inputs = processor(prompt, img)
        input_ids = inputs["input_ids"].to(args.device)
        attention_mask = inputs["attention_mask"].to(args.device)
        pixel_values = inputs["pixel_values"].to(dtype=torch.bfloat16, device=args.device)

        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                output_hidden_states=True,
            )

        last_hidden = outputs.hidden_states[-1]

        # Reconstruct multimodal_attention_mask:
        # forward() builds: [input_embeds[:,:1,:], patch_embeds, input_embeds[:,1:,:]]
        # So multimodal_seq_len = input_ids_seq_len + num_patches
        text_seq_len = input_ids.shape[1]
        multimodal_seq_len = last_hidden.shape[1]
        num_patches = multimodal_seq_len - text_seq_len

        patch_mask = torch.ones(
            (1, num_patches), dtype=attention_mask.dtype, device=attention_mask.device
        )
        multimodal_mask = torch.cat(
            [attention_mask[:, :1], patch_mask, attention_mask[:, 1:]], dim=1
        )

        feat = masked_mean_pool(last_hidden, multimodal_mask).squeeze(0).cpu()
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
        "extraction_point": "hidden_states[-1] (post-norm Llama-2 RMSNorm)",
        "pooling": "masked_mean_pool (multimodal sequence: vision patches + text)",
        "hidden_size": hidden_size,
        "feats_A_shape": list(feats_A.shape),
        "num_samples": num_samples,
        "prompt_template": "In: What action should the robot take to {task}?\\nOut:",
    }
    with open(os.path.join(args.output_dir, "extraction_metadata.json"), "w") as f:
        json.dump(extraction_meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n=== OpenVLA Phase 2 Complete ===")
    print(f"  feats_A: {feats_A_path}  shape={tuple(feats_A.shape)}")
    print(f"  Time: {elapsed/60:.1f} min  ({elapsed/num_samples:.2f} s/sample)")


if __name__ == "__main__":
    main()
