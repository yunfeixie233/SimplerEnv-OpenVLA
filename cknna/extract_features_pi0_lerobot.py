"""
Phase 2: Extract VLM features (feats_A) from Pi0 lerobot for CKNNA.

Loads the HaomingSong/lerobot-pi0-bridge checkpoint, runs a prefill-only
forward pass (vision + language, no action decoding), captures features
via a forward hook on language_model.norm, and saves the masked-mean-pooled
output as feats_A.

Architecture: PaliGemma (SigLIP vision + Gemma LM, hidden_size=2048)
Extraction point: Hook on paligemma_with_expert.paligemma.language_model.norm
                  (POST-norm, output of final GemmaModel RMSNorm)
Pooling: masked mean-pool over non-padding prefix tokens (matches StarVLA)

The hook is read-only: detach() creates a new tensor, the original output
passes through unchanged. No model state is mutated.

The prefill-only approach is used instead of select_action because:
  1. select_action has an action queue -- without reset() the hook never fires
  2. Prefill-only is ~10x faster (no denoising loop)
  3. Prefill gives access to prefix_pad_masks for masked mean-pool

Requires: conda env pi0fast_env (transformers==4.53.3 custom fork)

Usage:
    python extract_features_pi0_lerobot.py \
        --ckpt_path HaomingSong/lerobot-pi0-bridge \
        --data_dir ./cknna_data \
        --output_dir ./cknna_data/pi0-lerobot-bridge
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image


def masked_mean_pool(hidden_states, mask):
    """Mean-pool hidden states over valid (non-padding) tokens.

    Matches the StarVLA extract_features_starvla.py implementation exactly.

    Args:
        hidden_states: (B, seq_len, D)
        mask: (B, seq_len) bool or int

    Returns:
        pooled: (B, D) float32
    """
    h = hidden_states.float()
    m = mask.unsqueeze(-1).float()
    return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)


def resolve_hook_module(model):
    """Resolve the norm module for the forward hook.

    Handles different possible module layouts in the HaomingSong checkpoint.
    The norm layer is at the end of the GemmaModel, capturing POST-norm output.
    """
    candidates = [
        "paligemma_with_expert.paligemma.language_model.norm",
        "paligemma_with_expert.paligemma.language_model.model.norm",
    ]
    for path in candidates:
        obj = model
        for attr in path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            print(f"  Hook path resolved: {path}")
            return obj
    raise AttributeError(
        f"Cannot resolve norm module. Tried: {candidates}. "
        f"Check the checkpoint's model structure."
    )


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Extract Pi0 lerobot features for CKNNA.")
    parser.add_argument("--ckpt_path", type=str, default="HaomingSong/lerobot-pi0-bridge",
                        help="HuggingFace model ID or local path to Pi0 checkpoint")
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

    # ---- Load Pi0 via checkpoint's bundled code + compat shim ----
    print(f"Loading Pi0 from: {args.ckpt_path}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    from _lerobot_common_shim import install_shim
    install_shim()

    from huggingface_hub import snapshot_download
    local_ckpt = snapshot_download(args.ckpt_path)

    sys.path.insert(0, local_ckpt)
    from modeling_pi0 import PI0Policy, make_att_2d_masks

    if "predict_action_chunk" in getattr(PI0Policy, "__abstractmethods__", set()):
        PI0Policy.predict_action_chunk = lambda self, batch, **kw: None
        PI0Policy.__abstractmethods__ = PI0Policy.__abstractmethods__ - {"predict_action_chunk"}

    policy = PI0Policy.from_pretrained(local_ckpt)
    policy.to(args.device)
    policy.eval()
    model = policy.model

    # ---- Resolve hidden size ----
    norm_module = resolve_hook_module(model)
    weight_shape = list(norm_module.weight.shape)
    hidden_size = weight_shape[0]
    print(f"  hidden_size: {hidden_size}")
    print(f"  Samples to process: {num_samples}")

    # ---- Set up tokenizer ----
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(local_ckpt)
    tokenizer_max_length = policy.config.tokenizer_max_length
    print(f"  tokenizer_max_length: {tokenizer_max_length}")

    # ---- Register forward hook ----
    hook_output = {}

    def hook_fn(module, inp, out):
        if isinstance(out, tuple):
            hook_output["feat"] = out[0].detach()
        else:
            hook_output["feat"] = out.detach()

    handle = norm_module.register_forward_hook(hook_fn)

    # ---- Detect image key from config ----
    img_keys = list(policy.config.image_features.keys())
    img_key = img_keys[0]
    print(f"  Image key from config: {img_key}  (total cameras: {len(img_keys)})")

    # ---- Detect model precision ----
    weight_dtype = model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
    print(f"  Model dtype: {weight_dtype}")

    # ---- Resume support ----
    partial_path = os.path.join(args.output_dir, "feats_A_partial.pt")
    if args.resume_from > 0 and os.path.exists(partial_path):
        partial_tensor = torch.load(partial_path, weights_only=True)
        feats_list = list(partial_tensor[:args.resume_from])
        print(f"  Resuming from sample {args.resume_from} ({len(feats_list)} loaded)")
    else:
        feats_list = []
        args.resume_from = 0

    OBS_LANGUAGE_TOKENS = "observation.language.tokens"
    OBS_LANGUAGE_ATTENTION_MASK = "observation.language.attention_mask"

    t0 = time.time()
    for i in range(args.resume_from, num_samples):
        img_np = np.array(Image.open(os.path.join(images_dir, f"{i:06d}.png")).convert("RGB"))
        task = task_descriptions[i]

        image_tensor = torch.from_numpy(img_np / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(args.device)

        tokenized = tokenizer(
            [task],
            max_length=tokenizer_max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        lang_tokens = tokenized["input_ids"].to(args.device)
        lang_masks = tokenized["attention_mask"].to(dtype=torch.bool, device=args.device)

        batch = {
            img_key: image_tensor,
            OBS_LANGUAGE_TOKENS: lang_tokens,
            OBS_LANGUAGE_ATTENTION_MASK: lang_masks,
        }

        images, img_masks = policy.prepare_images(batch)

        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks,
        )

        if weight_dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        with torch.inference_mode():
            model.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=False,
            )

        feat = masked_mean_pool(hook_output["feat"], prefix_pad_masks).squeeze(0).cpu()
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

    handle.remove()

    feats_A = torch.stack(feats_list)
    feats_A_path = os.path.join(args.output_dir, "feats_A.pt")
    torch.save(feats_A, feats_A_path)

    if os.path.exists(partial_path):
        os.remove(partial_path)

    extraction_meta = {
        "model": args.ckpt_path,
        "extraction_point": "hook on language_model.norm (post-norm GemmaModel)",
        "pooling": "masked_mean_pool (prefix_pad_masks)",
        "hidden_size": hidden_size,
        "feats_A_shape": list(feats_A.shape),
        "num_samples": num_samples,
        "tokenizer_max_length": tokenizer_max_length,
    }
    with open(os.path.join(args.output_dir, "extraction_metadata.json"), "w") as f:
        json.dump(extraction_meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n=== Pi0 Lerobot Phase 2 Complete ===")
    print(f"  feats_A: {feats_A_path}  shape={tuple(feats_A.shape)}")
    print(f"  Time: {elapsed/60:.1f} min  ({elapsed/num_samples:.2f} s/sample)")


if __name__ == "__main__":
    main()
