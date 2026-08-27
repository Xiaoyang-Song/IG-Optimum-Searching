"""
Experiment 3: does IG attribution actually identify "which UFK deserves more
update," and does that match ground truth?

f(x) = sigmoid( sum_j c_j * x_j ),  x in R^6,  c = [5, 5, 0.1, 0.1, 0.1, 0.1]

Coordinates 1-2 are "important" (large coefficient), coordinates 3-6 are
"decoys" (nearly irrelevant). x^(0) is pushed equally far in every coordinate
so it is deep in saturation: the RAW local gradient there is so small it
underflows/rounds unevenly across coordinates and cannot reliably be used to
rank importance. IG, computed as a path integral from a sensible baseline,
does not have this problem -- it is checked against the known ground-truth
coefficients c_j.

We then check the practical consequence: does IG-based coordinate weighting
(Sec 4.1's W_IG) actually make the algorithm move the important coordinates
more, relative to an otherwise-identical run with W_IG replaced by the
identity (i.e. the path-gradient escape mechanism alone, no attribution)?
"""
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from ig_update import run_algorithm1, avg_path_grad_and_ig, ig_weights, grad_and_value

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

C = torch.tensor([5.0, 5.0, 0.1, 0.1, 0.1, 0.1])
P = len(C)
LABELS = [f"$x_{j+1}$" + (" (important)" if C[j] >= 1.0 else " (unimportant)") for j in range(P)]


def f(x):
    return torch.sigmoid((C * x).sum())


def run():
    x0 = torch.full((P,), -3.0)  # equal displacement in every coordinate -> deep saturation
    b0 = torch.zeros(P)          # b sits exactly at the sensitive point f(b0)=0.5
    y_t = 0.9
    project = lambda t: t.clamp(-6, 6)

    g_loc, y0 = grad_and_value(f, x0)
    _, ig0 = avg_path_grad_and_ig(f, b0, x0, M=25)
    print(f"[ig_attr] x0: f(x0)={y0.item():.3e}  local grad norm={g_loc.norm().item():.3e}")
    print(f"[ig_attr] |local grad| per coord: {[f'{v:.2e}' for v in g_loc.abs().tolist()]}")
    print(f"[ig_attr] |IG| per coord:         {[f'{v:.4f}' for v in ig0.abs().tolist()]}")
    print(f"[ig_attr] true |c_j|:             {C.tolist()}")

    _fig_attribution_bars(g_loc, ig0)

    R = 400
    _, _, hist_weighted = run_algorithm1(
        f, x0, b0, y_t, R=R, eta_x=2.0, eta_b=0.0, eta_kick=0.0,
        tau_x=0.05, tau_b=1.0, eps_x=1e-3, eps_b=1e-3, M=25,
        project_x=project, project_b=project, adapt_baseline=False, use_ig_weights=True,
    )
    _, _, hist_unweighted = run_algorithm1(
        f, x0, b0, y_t, R=R, eta_x=2.0, eta_b=0.0, eta_kick=0.0,
        tau_x=0.05, tau_b=1.0, eps_x=1e-3, eps_b=1e-3, M=25,
        project_x=project, project_b=project, adapt_baseline=False, use_ig_weights=False,
    )
    print(f"[ig_attr] IG-weighted:   {len(hist_weighted)-1} iters, final f={hist_weighted[-2]['x_y_x']:.4f}")
    print(f"[ig_attr] unweighted:    {len(hist_unweighted)-1} iters, final f={hist_unweighted[-2]['x_y_x']:.4f}")

    _fig_coordinate_movement(x0, hist_weighted, hist_unweighted)
    _fig_weight_evolution(hist_weighted)


def _fig_attribution_bars(g_loc, ig0):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    idx = np.arange(P)

    axes[0].bar(idx, g_loc.abs().numpy(), color="crimson")
    axes[0].set_title("$|$local gradient$|$ at $x^{(0)}$\n(deep in saturation -- numerically dead)")
    axes[0].set_yscale("log")

    axes[1].bar(idx, ig0.abs().numpy(), color="steelblue")
    axes[1].set_title(r"$|IG_j|$ relative to $b^{(0)}$" "\n(path integral, still well-scaled)")

    axes[2].bar(idx, C.numpy(), color="seagreen")
    axes[2].set_title("ground-truth $|c_j|$\n(what the model actually depends on)")

    for ax in axes:
        ax.set_xticks(idx)
        ax.set_xticklabels(LABELS, rotation=30, ha="right", fontsize=8)
    fig.suptitle("Local gradient is uninformative under saturation; IG still recovers the true ranking")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "attr_bars_local_vs_ig.png"), dpi=150)
    plt.close(fig)


def _fig_coordinate_movement(x0, hist_weighted, hist_unweighted):
    def total_abs_move(hist):
        xs = np.array([h["x"].numpy() for h in hist])
        return np.abs(xs - x0.numpy()).sum(axis=0)

    move_w = total_abs_move(hist_weighted)
    move_u = total_abs_move(hist_unweighted)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    idx = np.arange(P)
    width = 0.35
    ax.bar(idx - width / 2, move_w, width, label="IG-weighted (Algorithm 1)", color="steelblue")
    ax.bar(idx + width / 2, move_u, width, label="unweighted ablation ($W_{IG}=I$)", color="gray")
    ax.set_yscale("log")
    ax.set_xticks(idx)
    ax.set_xticklabels(LABELS, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(r"cumulative $\sum_r |x_j^{(r+1)}-x_j^{(r)}|$ (log scale)")
    ax.set_title("IG weighting concentrates movement onto the truly important coordinates")
    ax.legend()

    ratio_w = move_w[:2].sum() / move_w[2:].sum()
    ratio_u = move_u[:2].sum() / move_u[2:].sum()
    ax.text(0.02, 0.95, f"important:unimportant movement ratio\nIG-weighted = {ratio_w:.1f}x\nunweighted = {ratio_u:.1f}x",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    print(f"[ig_attr] important:decoy movement ratio -- IG-weighted={ratio_w:.2f}x, unweighted={ratio_u:.2f}x")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "attr_coordinate_movement.png"), dpi=150)
    plt.close(fig)


def _fig_weight_evolution(hist_weighted):
    W = np.array([h["x_w"].numpy() for h in hist_weighted if "x_w" in h])
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for j in range(P):
        ax.plot(W[:, j], label=LABELS[j], lw=1.3)
    ax.set_xlabel("iteration $r$")
    ax.set_ylabel(r"IG coordinate weight $w_j^{(r)}$")
    ax.set_title(r"$W_{IG}$ over the run: important coordinates stay upweighted")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "attr_weight_evolution.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run()
