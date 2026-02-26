"""
Phase 2: Extract VLM features (feats_A) from CogACT for CKNNA.

CogACT shares the same Prismatic VLM backbone as OpenVLA:
  DINOv2 + SigLIP fused vision + Llama-2-7b LM, hidden_size=4096

Extraction point: hidden_states[-1] (POST-norm, after final Llama-2 RMSNorm)
Pooling: masked mean-pool over the full multimodal sequence
         (vision patches + text tokens, including the cognition token)

Prompt format: same as CogACT predict_action():
  prompt_builder.add_turn("human", "What action should the robot take to {task}?")
  + appended tokens [29871, 2] (empty-space token + cognition/EOS token)

The multimodal sequence is: [BOS, vision_patches, text_tokens_after_BOS]
where vision patches are always valid (mask=1). The multimodal_attention_mask
is reconstructed identically to extract_features_openvla.py.

State: NOT in feats_A. CogACT's proprioceptive state is never fed into
the VLM; the DiT action model receives state implicitly via the cognition
feature only. This matches all other models in the CKNNA comparison.

Requires: cogact conda env

Usage:
    python extract_features_cogact.py \
        --ckpt CogACT/CogACT-Base \
        --data_dir ./cknna_data \
        --output_dir ./cknna_data/cogact-base-bridge
"""

import argparse
import json
import os
import time

import torch
from PIL import Image
from transformers import LlamaTokenizerFast

from vla import load_vla


def masked_mean_pool(hidden_states, attention_mask):
    """Mean-pool hidden states over valid (non-padding) tokens.

    Matches StarVLA and OpenVLA extraction scripts exactly.
    """
    h = hidden_states.float()
    m = attention_mask.unsqueeze(-1).float()
    return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Extract CogACT features for CKNNA.")
    parser.add_argument("--ckpt", type=str, default="CogACT/CogACT-Base",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--action_model_type", type=str, default="DiT-B",
                        help="DiT variant: DiT-Small, DiT-B (Base), DiT-L (Large)")
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

    print(f"Loading CogACT: {args.ckpt} (action_model_type={args.action_model_type})")
    vla = load_vla(
        args.ckpt,
        load_for_training=False,
        action_model_type=args.action_model_type,
    )
    vla.vlm = vla.vlm.to(torch.bfloat16)
    vla = vla.to(args.device).eval()

    vlm = vla.vlm
    tokenizer = vlm.llm_backbone.tokenizer
    image_transform = vlm.vision_backbone.image_transform
    hidden_size = vlm.llm_backbone.llm.config.hidden_size

    print(f"  hidden_size: {hidden_size}")
    print(f"  tokenizer type: {type(tokenizer).__name__}")
    print(f"  Samples to process: {num_samples}")

    partial_path = os.path.join(args.output_dir, "feats_A_partial.pt")
    if args.resume_from > 0 and os.path.exists(partial_path):
        partial_tensor = torch.load(partial_path, weights_only=True)
        feats_list = list(partial_tensor[:args.resume_from])
        print(f"  Resuming from sample {args.resume_from} ({len(feats_list)} loaded)")
    else:
        feats_list = []
        args.resume_from = 0

    autocast_dtype = vlm.llm_backbone.half_precision_dtype

    t0 = time.time()
    for i in range(args.resume_from, num_samples):
        img = Image.open(os.path.join(images_dir, f"{i:06d}.png")).convert("RGB")
        task = task_descriptions[i]

        prompt_builder = vlm.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=f"What action should the robot take to {task.lower()}?")
        prompt_text = prompt_builder.get_prompt()

        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(args.device)
        assert isinstance(tokenizer, LlamaTokenizerFast), f"Expected LlamaTokenizerFast, got {type(tokenizer)}"
        input_ids = torch.cat(
            (input_ids, torch.tensor([[29871, 2]], dtype=torch.long, device=args.device)), dim=1
        )
        attention_mask = torch.ones_like(input_ids)

        pixel_values = image_transform(img)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(args.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(args.device) for k, v in pixel_values.items()}

        with torch.inference_mode(), torch.autocast("cuda", dtype=autocast_dtype):
            output = vlm(
                input_ids=input_ids,
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )

        last_hidden = output.hidden_states[-1]

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
        "action_model_type": args.action_model_type,
        "extraction_point": "hidden_states[-1] (post-norm Llama-2 RMSNorm, same as OpenVLA)",
        "pooling": "masked_mean_pool (multimodal: vision patches + text + cognition token)",
        "hidden_size": hidden_size,
        "feats_A_shape": list(feats_A.shape),
        "num_samples": num_samples,
        "prompt_template": "CogACT prompt_builder + appended [29871, 2]",
        "state_in_feats_A": False,
    }
    with open(os.path.join(args.output_dir, "extraction_metadata.json"), "w") as f:
        json.dump(extraction_meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n=== CogACT Phase 2 Complete ===")
    print(f"  feats_A: {feats_A_path}  shape={tuple(feats_A.shape)}")
    print(f"  Time: {elapsed/60:.1f} min  ({elapsed/num_samples:.2f} s/sample)")


if __name__ == "__main__":
    main()
