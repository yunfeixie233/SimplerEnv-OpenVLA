"""
Phase 3: Compute CKNNA -- memory-optimized for large N (30K-50K).

Identical algorithm to compute_cknna.py but reduces peak GPU memory from
~11 NxN float32 matrices to ~5 NxN by:
  1. In-place diagonal modification (save/restore, no clone)
  2. Replacing mm(K,L) in HSIC with sum-of-row-sums dot product
  3. Aggressive del + empty_cache between _similarity calls

For N=50K on H100 80GB:
  1 NxN matrix = 9.3 GB
  Peak ~5 NxN = 46.5 GB + ~1 GB overhead = ~48 GB  (fits in 80 GB)

Usage:
  python compute_cknna_large.py \
      --feats_A ./cknna_data_50k/spatialvla-sft-bridge/feats_A.pt \
      --feats_B ./cknna_data_50k/feats_B.pt \
      --topk 5 10 20
"""

import argparse
import datetime
import json
import os
import time

import torch
import torch.nn.functional as F


def _hsic_unbiased_lowmem(K, L):
    """Unbiased HSIC (Song et al., 2012) -- low-memory version.

    K and L are modified IN-PLACE (diagonal zeroed). Caller must not reuse them.
    The mm(K, L) term is replaced with a dot product of row-sums/column-sums:
      sum(mm(K, L)) = K.sum(dim=0) @ L.sum(dim=1)
    This avoids allocating a full NxN product matrix.
    """
    m = K.shape[0]
    K.fill_diagonal_(0)
    L.fill_diagonal_(0)

    # term1 = sum(K * L.T) -- must create temporary NxN chunk-by-chunk
    # For moderate N (~50K), the full product is ~9.3 GB which fits if we have headroom.
    # But to be safe, compute in row chunks.
    term1 = torch.tensor(0.0, device=K.device, dtype=K.dtype)
    chunk = 2000
    for i in range(0, m, chunk):
        end = min(i + chunk, m)
        term1 += (K[i:end] * L.T[i:end]).sum()

    term2 = K.sum() * L.sum() / ((m - 1) * (m - 2))

    # sum(mm(K,L)) = sum_k (sum_i K_{ik}) * (sum_j L_{kj})
    k_col_sums = K.sum(dim=0)
    l_row_sums = L.sum(dim=1)
    term3 = 2.0 * (k_col_sums @ l_row_sums) / (m - 2)
    del k_col_sums, l_row_sums

    return (term1 + term2 - term3) / (m * (m - 3))


def _compute_topk_mask(sim_mat, topk, n, device):
    """Find top-k neighbors per row and return a binary NxN mask.

    Modifies sim_mat diagonal in-place (set to -inf, then restored).
    """
    diag = sim_mat.diagonal().clone()
    sim_mat.fill_diagonal_(float("-inf"))
    _, topk_idx = torch.topk(sim_mat, topk, dim=1)
    sim_mat.diagonal().copy_(diag)
    del diag

    mask = torch.zeros(n, n, device=device, dtype=sim_mat.dtype)
    mask.scatter_(1, topk_idx, 1.0)
    del topk_idx
    return mask


def cknna_lowmem(feats_A, feats_B, topk=10):
    """CKNNA -- memory-optimized.

    Peak GPU memory: ~5 NxN float32 matrices (K, L, mask, masked_K, masked_L).
    """
    n = feats_A.shape[0]
    device = feats_A.device

    K = feats_A @ feats_A.T
    L = feats_B @ feats_B.T

    del feats_A, feats_B
    torch.cuda.empty_cache()

    def _similarity(K_mat, L_mat, topk_val):
        mask_K = _compute_topk_mask(K_mat, topk_val, n, device)
        mask_L = _compute_topk_mask(L_mat, topk_val, n, device)
        mask_K.mul_(mask_L)
        del mask_L
        mask = mask_K

        masked_K = mask * K_mat
        masked_L = mask * L_mat
        del mask
        torch.cuda.empty_cache()

        result = _hsic_unbiased_lowmem(masked_K, masked_L)
        del masked_K, masked_L
        torch.cuda.empty_cache()
        return result

    sim_kl = _similarity(K, L, topk)
    sim_kk = _similarity(K, K, topk)
    sim_ll = _similarity(L, L, topk)

    del K, L
    torch.cuda.empty_cache()

    return sim_kl.item() / (torch.sqrt(sim_kk * sim_ll) + 1e-6).item()


def mutual_knn_lowmem(feats_A, feats_B, topk=10):
    """Mutual k-NN -- memory-optimized (sequential sim matrix computation)."""
    n = feats_A.shape[0]
    device = feats_A.device

    sim_A = (feats_A @ feats_A.T).fill_diagonal_(-1e8)
    _, knn_A = torch.topk(sim_A, topk, dim=1)
    del sim_A
    torch.cuda.empty_cache()

    sim_B = (feats_B @ feats_B.T).fill_diagonal_(-1e8)
    _, knn_B = torch.topk(sim_B, topk, dim=1)
    del sim_B
    torch.cuda.empty_cache()

    mask_A = torch.zeros(n, n, device=device).scatter_(1, knn_A, 1.0)
    del knn_A
    mask_B = torch.zeros(n, n, device=device).scatter_(1, knn_B, 1.0)
    del knn_B

    result = ((mask_A * mask_B).sum(dim=1) / topk).mean().item()
    del mask_A, mask_B
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Compute CKNNA (large N, memory-optimized).")
    parser.add_argument("--feats_A", type=str, nargs="+", required=True)
    parser.add_argument("--feats_B", type=str, required=True)
    parser.add_argument("--topk", type=int, nargs="+", default=[10])
    parser.add_argument("--also_mutual_knn", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    feats_B_raw = torch.load(args.feats_B, weights_only=True).float()
    feats_B_norm = F.normalize(feats_B_raw, p=2, dim=-1).to(args.device)
    N = feats_B_norm.shape[0]
    del feats_B_raw

    print(f"feats_B: {args.feats_B}  shape={tuple(feats_B_norm.shape)}  N={N}")
    print(f"k values: {args.topk}")
    mem_per_mat = N * N * 4 / 1024**3
    print(f"NxN matrix size: {mem_per_mat:.1f} GB  (peak ~5x = {mem_per_mat*5:.1f} GB)")
    print()

    results = {
        "_meta": {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "feats_B_path": args.feats_B,
            "feats_B_shape": list(feats_B_norm.shape),
            "N": N,
            "k_values": args.topk,
            "memory_optimized": True,
        },
        "models": {},
    }

    for feats_A_path in args.feats_A:
        feats_A_raw = torch.load(feats_A_path, weights_only=True).float()
        feats_A_norm = F.normalize(feats_A_raw, p=2, dim=-1).to(args.device)
        del feats_A_raw

        assert feats_A_norm.shape[0] == N

        label = os.path.basename(os.path.dirname(os.path.abspath(feats_A_path)))
        print(f"--- {label} ---")
        print(f"  feats_A shape: {tuple(feats_A_norm.shape)}")

        entry = {
            "feats_A_path": feats_A_path,
            "feats_A_dim": feats_A_norm.shape[1],
        }

        for k in args.topk:
            t0 = time.time()
            score = cknna_lowmem(feats_A_norm, feats_B_norm, topk=k)
            t1 = time.time()
            print(f"  CKNNA (k={k}): {score:.6f}  [{t1-t0:.1f}s]")
            entry[f"cknna_k{k}"] = score

            if args.also_mutual_knn:
                t0 = time.time()
                mknn = mutual_knn_lowmem(feats_A_norm, feats_B_norm, topk=k)
                t1 = time.time()
                print(f"  mutual_knn (k={k}): {mknn:.6f}  [{t1-t0:.1f}s]")
                entry[f"mutual_knn_k{k}"] = mknn

        results["models"][label] = entry
        del feats_A_norm
        torch.cuda.empty_cache()
        print()

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
