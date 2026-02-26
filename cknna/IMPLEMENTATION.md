# CKNNA Feature Extraction for SimplerEnv-OpenVLA Models

Implementation guide for computing CKNNA on SpatialVLA and Pi0 lerobot using
shared Bridge data from the StarVLA pipeline.

## Directory Layout

```
SimplerEnv-OpenVLA/cknna/
  cknna_data/               -> symlink to <WORK>/starVLA/cknna/cknna_data/
    feats_B.pt              (5000, 7)  -- shared proprioceptive states
    metadata.json           -- task descriptions and sample indices
    images/                 -- 5000 PNGs at 256x256
    Qwen-GR00T-Bridge/feats_A.pt      -- StarVLA model (already computed)
    Qwen-FAST-Bridge-RT-1/feats_A.pt  -- StarVLA model (already computed)
    ... (other StarVLA models)
  extract_features_spatialvla.py
  extract_features_pi0_lerobot.py
  compute_cknna.py          -- copied from lerobot/cknna/
  run_cknna_simplerenv.sh   -- orchestration script
  IMPLEMENTATION.md         -- this file
```

## Models and Extraction Points

### SpatialVLA (spatialvla-4b-224-sft-bridge)

| Property | Value |
|----------|-------|
| Architecture | PaLiGemma2: SigLIP vision + ZoeDepth depth + Gemma2 LM |
| Checkpoint | IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge |
| Extraction point | `outputs.hidden_states[-1]` (POST-norm) |
| Mechanism | `output_hidden_states=True` (standard HuggingFace, no hooks) |
| Pooling | Masked mean-pool over non-padding tokens (attention_mask) |
| feat_A dim | 2304 |
| State in feat_A | NO (no proprioceptive input to VLM) |
| Conda env | spatialvla_env (transformers>=4.47.0) |

How `hidden_states[-1]` is POST-norm:
  Gemma2Model.forward() calls self.norm(hidden_states) AFTER all decoder layers,
  then appends the normed result to all_hidden_states. Index -1 is this final entry.
  (Verified in modeling_gemma2.py lines 481-484.)

Key implementation details:
  - `do_normalize=False` is explicitly passed to the processor (prevents double normalization)
  - `unnorm_key="bridge_orig/1.0.0"` selects the correct ZoeDepth intrinsic parameters
  - ZoeDepth cannot be skipped (model was trained with it; skipping would change features)
  - Processor returns {input_ids, attention_mask, pixel_values, intrinsic}
  - Model forward accepts all processor outputs via **inputs unpacking

### Pi0 lerobot (lerobot-pi0-bridge)

| Property | Value |
|----------|-------|
| Architecture | PaliGemma: SigLIP vision + Gemma LM + Gemma expert |
| Checkpoint | HaomingSong/lerobot-pi0-bridge |
| Extraction point | Forward hook on language_model.norm (POST-norm) |
| Mechanism | register_forward_hook on GemmaModel.norm, prefill-only forward |
| Pooling | Masked mean-pool over non-padding prefix tokens (prefix_pad_masks) |
| feat_A dim | 2048 |
| State in feat_A | NO (state enters only expert via state_proj/suffix_embs) |
| Conda env | pi0fast_env (transformers==4.53.3 custom fork) |

How the hook captures POST-norm features:
  The hook is registered on GemmaModel.norm (the final RMSNorm layer).
  When the forward pass reaches this layer, the hook captures its output,
  which is the normalized hidden states. This is equivalent to hidden_states[-1]
  in the HuggingFace convention.

Why prefill-only (not select_action):
  1. select_action has an action queue -- without policy.reset() before EACH call,
     subsequent calls pop from the queue and never run the model (hook never fires)
  2. Prefill-only is ~10x faster (no denoising loop iterations)
  3. select_action pads text to tokenizer_max_length. With naive mean(dim=1), padding
     positions dilute features. Prefill-only gives access to prefix_pad_masks for
     proper masked mean-pool.

Key implementation details:
  - Checkpoint loaded via: snapshot_download() + sys.path.insert(0, local_ckpt) +
    from modeling_pi0 import PI0Policy, make_att_2d_masks
  - Text tokenization done MANUALLY (not via lerobot preprocessor pipeline):
    AutoTokenizer.from_pretrained(local_ckpt) + tokenizer([task], max_length=...,
    padding="max_length", return_tensors="pt")
  - Image key auto-detected from policy.config.image_features to handle different
    checkpoint configs (e.g., "observation.images.image_0" vs "observation.images.image")
  - Hook path resolved with fallback: tries language_model.norm first,
    then language_model.model.norm (handles GemmaModel vs GemmaForCausalLM wrapping)
  - embed_prefix() used (NOT embed_prefix_fast() -- that's Pi0Fast only)
  - make_att_2d_masks imported from checkpoint code (NOT from lerobot.policies.pi0_fast)
  - inputs_embeds=[prefix_embs, None] sends only VLM input (None = skip expert)
  - use_cache=False since we don't need KV cache (no subsequent decoding)
  - Dummy state tensor is NOT needed (removed from batch) -- state only enters
    embed_suffix() during denoising, not during prefill

### Cross-Model Alignment Table

| Aspect | SpatialVLA | Pi0 lerobot | StarVLA (reference) |
|--------|------------|-------------|---------------------|
| Data | Bridge 5000 samples (symlink) | Same | Same |
| feats_B | (5000, 7) shared | Same | Same |
| Extraction depth | hidden_states[-1] | hook on norm | hidden_states[-1] |
| Normalization | Post-norm (Gemma2) | Post-norm (Gemma) | Post-norm (Qwen2.5-VL/Qwen3-VL) |
| Pooling | masked_mean_pool | masked_mean_pool | masked_mean_pool |
| Pooling code | identical 3-line impl | identical 3-line impl | identical 3-line impl |
| Modalities in feat_A | vision + text | vision + text | vision + text |
| State in feat_A | NO | NO | NO |
| CKNNA code | compute_cknna.py (copied) | Same | Same |

## Conda Environment Setup

### spatialvla_env (for SpatialVLA)

```bash
CONDA=${CONDA_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)/conda}
$CONDA/bin/conda create -n spatialvla_env python=3.10 -y
source $CONDA/bin/activate spatialvla_env
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install "transformers>=4.47.0" accelerate pillow huggingface_hub
```

### pi0fast_env (for Pi0 lerobot, already exists)

Uses the custom transformers fork at 4.53.3. Already set up for CKNNA on LIBERO.

## Running the Pipeline

### Full pipeline (all phases)

```bash
cd /path/to/SimplerEnv-OpenVLA
bash cknna/run_cknna_simplerenv.sh
```

### Individual phases

```bash
# Phase 2a: SpatialVLA (in spatialvla_env)
source $CONDA/bin/activate spatialvla_env
python cknna/extract_features_spatialvla.py \
    --ckpt IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge \
    --data_dir cknna/cknna_data \
    --output_dir cknna/cknna_data/spatialvla-sft-bridge

# Phase 2b: Pi0 lerobot (in pi0fast_env)
source $CONDA/bin/activate pi0fast_env
python cknna/extract_features_pi0_lerobot.py \
    --ckpt_path HaomingSong/lerobot-pi0-bridge \
    --data_dir cknna/cknna_data \
    --output_dir cknna/cknna_data/pi0-lerobot-bridge

# Phase 3: Compute CKNNA (any env with torch)
python cknna/compute_cknna.py \
    --feats_A cknna/cknna_data/spatialvla-sft-bridge/feats_A.pt \
              cknna/cknna_data/pi0-lerobot-bridge/feats_A.pt \
              cknna/cknna_data/Qwen-GR00T-Bridge/feats_A.pt \
              cknna/cknna_data/Qwen-FAST-Bridge-RT-1/feats_A.pt \
              cknna/cknna_data/Qwen-OFT-Bridge-RT-1/feats_A.pt \
    --feats_B cknna/cknna_data/feats_B.pt \
    --topk 5 10 20 --also_mutual_knn \
    --output cknna/cknna_data/cknna_results_simplerenv.json
```

## Verification Checklist

1. [x] Different architectures handled with correct extraction points
   - SpatialVLA: output_hidden_states[-1] on Gemma2
   - Pi0: hook on GemmaModel.norm with fallback path resolution
2. [x] Extracted features have consistent semantics (post-VLM condition embedding)
3. [x] Vision + text features for all models (no vision-only, no state contamination)
4. [x] Hooks do not affect inference (detach() for Pi0, output_hidden_states for SpatialVLA)
5. [x] Aligned with StarVLA pipeline (same data, pooling, normalization, CKNNA code)
6. [x] Image key auto-detected from Pi0 config (handles different checkpoint conventions)
7. [x] do_normalize=False explicitly passed for SpatialVLA (prevents double normalization)
8. [x] Pi0 tokenization done manually (fixes pseudocode bug A1 from design doc)
9. [x] masked_mean_pool implementation identical across all 3 codebases

## Known Caveats

- SpatialVLA includes ZoeDepth features (3D spatial info) that other models lack.
  This is an architectural confound, acceptable since we compare models as-is.
- Pi0 uses masked mean-pool with prefix_pad_masks, while the lerobot LIBERO CKNNA
  pipeline used naive mean-pool (mask=None). The SimplerEnv approach is more correct
  but creates a methodological gap if cross-referencing with LIBERO results.
- Different tokenizers produce different token counts for the same text, affecting
  mean-pool denominators. This is expected (different VLM architectures).
- Prompt templates differ across codebases (StarVLA uses CoT_prompt, SpatialVLA and
  Pi0 use raw task text). This is an acceptable protocol confound.
