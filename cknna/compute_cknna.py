"""
Phase 3: Compute CKNNA between model features and proprioceptive states.

Model-independent. Takes any feats_A (N, D) and feats_B (N, D') saved as .pt
files and computes CKNNA alignment.

The CKNNA implementation is adapted from the Platonic Representation Hypothesis
codebase (Huh et al.): platonic-rep/metrics.py

Usage:
  python compute_cknna.py \
      --feats_A ./cknna_data/feats_A_pointA.pt \
      --feats_B ./cknna_data/feats_B.pt \
      --topk 10

  # Compare two extraction points:
  python compute_cknna.py \
      --feats_A ./cknna_data/feats_A_pointA.pt ./cknna_data/feats_A_pointB.pt \
      --feats_B ./cknna_data/feats_B.pt \
      --topk 10

  # Sweep over k values:
  python compute_cknna.py \
      --feats_A ./cknna_data/feats_A_pointA.pt \
      --feats_B ./cknna_data/feats_B.pt \
      --topk 5 10 20 50
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# CKNNA metric (adapted from platonic-rep/metrics.py, Huh et al.)
# ---------------------------------------------------------------------------

def hsic_unbiased(K, L):
    """Unbiased HSIC estimator (Song et al., 2012, Equation 5)."""
    m = K.shape[0]
    K_tilde = K.clone().fill_diagonal_(0)
    L_tilde = L.clone().fill_diagonal_(0)

    term1 = torch.sum(K_tilde * L_tilde.T)
    term2 = torch.sum(K_tilde) * torch.sum(L_tilde) / ((m - 1) * (m - 2))
    term3 = 2 * torch.sum(torch.mm(K_tilde, L_tilde)) / (m - 2)

    return (term1 + term2 - term3) / (m * (m - 3))


def cknna(feats_A, feats_B, topk=10):
    """Centered Kernel Nearest-Neighbor Alignment.

    Args:
        feats_A: (N, D_A) tensor, L2-normalized
        feats_B: (N, D_B) tensor, L2-normalized
        topk:    number of nearest neighbors

    Returns:
        float, the CKNNA score in [0, 1]
    """
    n = feats_A.shape[0]
    K = feats_A @ feats_A.T
    L = feats_B @ feats_B.T
    device = feats_A.device

    def _similarity(K_mat, L_mat, topk_val):
        K_hat = K_mat.clone().fill_diagonal_(float("-inf"))
        L_hat = L_mat.clone().fill_diagonal_(float("-inf"))

        _, topk_K_idx = torch.topk(K_hat, topk_val, dim=1)
        _, topk_L_idx = torch.topk(L_hat, topk_val, dim=1)

        mask_K = torch.zeros(n, n, device=device).scatter_(1, topk_K_idx, 1)
        mask_L = torch.zeros(n, n, device=device).scatter_(1, topk_L_idx, 1)

        mask = mask_K * mask_L
        return hsic_unbiased(mask * K_mat, mask * L_mat)

    sim_kl = _similarity(K, L, topk)
    sim_kk = _similarity(K, K, topk)
    sim_ll = _similarity(L, L, topk)

    return sim_kl.item() / (torch.sqrt(sim_kk * sim_ll) + 1e-6).item()


def mutual_knn(feats_A, feats_B, topk=10):
    """Mutual k-NN accuracy (fraction of shared neighbors)."""
    n = feats_A.shape[0]
    device = feats_A.device

    sim_A = (feats_A @ feats_A.T).fill_diagonal_(-1e8)
    _, knn_A = torch.topk(sim_A, topk, dim=1)
    del sim_A

    sim_B = (feats_B @ feats_B.T).fill_diagonal_(-1e8)
    _, knn_B = torch.topk(sim_B, topk, dim=1)
    del sim_B

    mask_A = torch.zeros(n, n, device=device).scatter_(1, knn_A, 1.0)
    mask_B = torch.zeros(n, n, device=device).scatter_(1, knn_B, 1.0)

    return ((mask_A * mask_B).sum(dim=1) / topk).mean().item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Compute CKNNA.")
    parser.add_argument("--feats_A", type=str, nargs="+", required=True,
                        help="Path(s) to feats_A .pt file(s)")
    parser.add_argument("--feats_B", type=str, required=True,
                        help="Path to feats_B .pt file")
    parser.add_argument("--topk", type=int, nargs="+", default=[10],
                        help="k values for CKNNA (can specify multiple)")
    parser.add_argument("--also_mutual_knn", action="store_true",
                        help="Also compute mutual k-NN for comparison")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to save results as JSON")
    args = parser.parse_args()

    feats_B_raw = torch.load(args.feats_B, weights_only=True).float()
    feats_B_norm = F.normalize(feats_B_raw, p=2, dim=-1).to(args.device)
    N = feats_B_norm.shape[0]

    print(f"feats_B: {args.feats_B}  shape={tuple(feats_B_raw.shape)}  N={N}")
    print(f"k values: {args.topk}")
    print()

    import datetime

    results = {
        "_meta": {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "feats_B_path": args.feats_B,
            "feats_B_shape": list(feats_B_raw.shape),
            "N": N,
            "k_values": args.topk,
        },
        "models": {},
    }

    for feats_A_path in args.feats_A:
        feats_A_raw = torch.load(feats_A_path, weights_only=True).float()
        feats_A_norm = F.normalize(feats_A_raw, p=2, dim=-1).to(args.device)

        assert feats_A_norm.shape[0] == N, (
            f"feats_A has {feats_A_norm.shape[0]} samples but feats_B has {N}"
        )

        label = os.path.basename(os.path.dirname(os.path.abspath(feats_A_path)))
        print(f"--- {label} ---")
        print(f"  feats_A shape: {tuple(feats_A_raw.shape)}")

        entry = {
            "feats_A_path": feats_A_path,
            "feats_A_dim": feats_A_raw.shape[1],
        }

        for k in args.topk:
            score = cknna(feats_A_norm, feats_B_norm, topk=k)
            print(f"  CKNNA (k={k}): {score:.6f}")
            entry[f"cknna_k{k}"] = score

            if args.also_mutual_knn:
                mknn = mutual_knn(feats_A_norm, feats_B_norm, topk=k)
                print(f"  mutual_knn (k={k}): {mknn:.6f}")
                entry[f"mutual_knn_k{k}"] = mknn

        results["models"][label] = entry
        print()

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
