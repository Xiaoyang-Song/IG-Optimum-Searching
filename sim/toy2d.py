"""
Experiment 1: a 2D synthetic saturating target-seeking problem.

f(x1, x2) = mean_k sigmoid(kappa*(u - c_k)),  u = 3*x1 + x2

This is a *staircase* landscape along direction w=(3,1): four logistic risers at
u = -9,-3,3,9 separated by wide flat treads. Any local-gradient method that
lands on a tread saturates (gradient ~0) even though it hasn't reached the
target -- and it will saturate again on the *next* tread after crossing a
riser, so a working escape mechanism must fire repeatedly, not once.

Three methods are compared, all starting from the same x^(0) deep in the
lowest, most-saturated tread:
  - naive GD (no baseline at all),
  - Algorithm 1 with the baseline FROZEN at b^(0) (Sec 4's IG/path-gradient
    machinery, but Sec 5's baseline search disabled),
  - Algorithm 1 with the FULL adaptive baseline (Sec 5 + 5.1).

b^(0) is deliberately placed in the *same* dead flat zone as x^(0) (not on a
riser), so a frozen baseline there is provably useless -- the straight line
from b^(0) to x^(r) never crosses a riser as long as both sit in that zone,
so its path gradient is as dead as the local one. Only the adaptive baseline
can rescue itself, via Sec 5.1's directional probing, and then lead the
vehicle out. This directly tests the claim that *adapting* the baseline (not
just having *some* fixed baseline) is what matters.
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from ig_update import (
    run_algorithm1, run_gd, run_pgd_sign, run_mifgsm, run_adam,
)

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

W = torch.tensor([3.0, 1.0])
CENTERS = torch.tensor([-9.0, -3.0, 3.0, 9.0])
KAPPA = 3.0


def f(x):
    u = (W * x).sum()
    return torch.sigmoid(KAPPA * (u - CENTERS)).mean()


def f_np(x1, x2):
    u = 3 * x1 + x2
    out = np.zeros_like(u)
    for c in CENTERS.tolist():
        out = out + 1.0 / (1.0 + np.exp(-KAPPA * (u - c)))
    return out / len(CENTERS)


def gradnorm_np(x1, x2):
    u = 3 * x1 + x2
    du = np.zeros_like(u)
    for c in CENTERS.tolist():
        p = 1.0 / (1.0 + np.exp(-KAPPA * (u - c)))
        du = du + KAPPA * p * (1 - p)
    du = du / len(CENTERS)
    return np.sqrt((3 * du) ** 2 + (1 * du) ** 2)


def u_of(x):
    return float((W * x).sum())


def run():
    x0 = torch.tensor([-6.0, -2.0])   # u = -20, deep in the lowest tread
    b0 = torch.tensor([-6.5, -2.5])   # u = -22, same dead zone as x0
    y_t = 0.85
    project = lambda t: t.clamp(-10, 12)
    R = 300

    x_gd, hist_gd = run_gd(f, x0, y_t, R=R, eta_x=20.0, eps_x=1e-3, project_x=project)

    x_fx, b_fx, hist_fx = run_algorithm1(
        f, x0, b0, y_t, R=R, eta_x=3.0, eta_b=0.0, eta_kick=0.0,
        tau_x=0.05, tau_b=1.0, eps_x=1e-3, eps_b=1e-3, M=25,
        project_x=project, project_b=project, adapt_baseline=False,
    )

    x_a1, b_a1, hist_a1 = run_algorithm1(
        f, x0, b0, y_t, R=R, eta_x=3.0, eta_b=0.3, eta_kick=0.5,
        tau_x=0.05, tau_b=0.05, eps_x=1e-3, eps_b=1e-3, M=25, delta_b=0.5,
        top_k=2, project_x=project, project_b=project, adapt_baseline=True,
    )

    print(f"[toy2d] naive GD:              {len(hist_gd)-1} iters, final u_x={u_of(x_gd):.2f}, f={hist_gd[-2]['y_x']:.2e}")
    print(f"[toy2d] Algorithm1 (frozen b): {len(hist_fx)-1} iters, final u_x={u_of(x_fx):.2f}, f={hist_fx[-2]['x_y_x']:.2e}")
    print(f"[toy2d] Algorithm1 (adaptive): {len(hist_a1)-1} iters, final u_x={u_of(x_a1):.2f}, f={hist_a1[-2]['x_y_x']:.4f}")
    n_kicks = sum(1 for h in hist_a1 if h.get("b_kicked"))
    print(f"[toy2d] adaptive baseline: {n_kicks} probe-kick steps")

    _fig_heatmap_and_trajectories(x0, b0, hist_gd, hist_fx, hist_a1)
    _fig_convergence(hist_gd, hist_fx, hist_a1)
    _fig_mechanism_multi_escape(hist_a1)
    _fig_vs_baselines(x0, y_t, project, R)


def _fig_vs_baselines(x0, y_t, project, R):
    """Algorithm 1 vs three popular existing adversarial-attack update rules,
    all started from the same deeply-saturated x0, no baseline needed for the
    literature methods (sign/momentum/Adam all act on x alone)."""
    b0 = torch.tensor([0.0, 0.0])  # a reasonably-placed baseline, not the adversarial dead zone

    _, h_gd = run_gd(f, x0, y_t, R=R, eta_x=20.0, eps_x=1e-3, project_x=project)
    _, h_adam = run_adam(f, x0, y_t, R=R, eta_x=2.0, eps_x=1e-3, project_x=project)
    _, h_pgd = run_pgd_sign(f, x0, y_t, R=R, eta_x=0.05, eps_x=1e-3, project_x=project)
    _, h_mifgsm = run_mifgsm(f, x0, y_t, R=R, eta_x=0.05, mu=0.9, eps_x=1e-3, project_x=project)
    _, _, h_a1 = run_algorithm1(
        f, x0, b0, y_t, R=R, eta_x=3.0, eta_b=0.3, eta_kick=0.5,
        tau_x=0.05, tau_b=0.05, eps_x=1e-3, eps_b=1e-3, M=25, delta_b=0.5,
        top_k=2, project_x=project, project_b=project, adapt_baseline=True,
    )

    series = {
        "GD": ("crimson", [abs(h["e_x"]) for h in h_gd if "e_x" in h]),
        "Adam": ("goldenrod", [abs(h["e_x"]) for h in h_adam if "e_x" in h]),
        "PGD / BIM (sign)": ("seagreen", [abs(h["e_x"]) for h in h_pgd if "e_x" in h]),
        "MI-FGSM (momentum-sign)": ("darkviolet", [abs(h["e_x"]) for h in h_mifgsm if "e_x" in h]),
        "Algorithm 1 (IG-guided)": ("steelblue", [abs(h["x_e_x"]) for h in h_a1 if "x_e_x" in h]),
    }
    print("\n[toy2d] === vs. literature baselines (same deeply-saturated x0) ===")
    for name, (_, err) in series.items():
        converged = len(err) < R
        print(f"  {name:26s}: {'converged in ' + str(len(err)) + ' iters' if converged else f'NOT converged after {R} iters'}, "
              f"final |e_x|={err[-1]:.4f}, tail-window range=({min(err[-30:]):.4f},{max(err[-30:]):.4f})")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, (color, err) in series.items():
        ax.semilogy(err, color=color, label=name, lw=1.4)
    ax.axhline(1e-3, color="gray", ls="--", lw=1, label=r"$\epsilon_x$ threshold")
    ax.set_xlabel("iteration $r$")
    ax.set_ylabel(r"$|e_x^{(r)}|$")
    ax.set_title("Algorithm 1 vs. popular adversarial-attack update rules\n"
                 "(sign-based methods escape saturation but never settle precisely)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "toy_vs_baselines.png"), dpi=150)
    plt.close(fig)


def _fig_heatmap_and_trajectories(x0, b0, hist_gd, hist_fx, hist_a1):
    xs1 = np.linspace(-8, 4, 320)
    xs2 = np.linspace(-6, 4, 320)
    X1, X2 = np.meshgrid(xs1, xs2)
    GN = gradnorm_np(X1, X2)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.pcolormesh(X1, X2, GN, shading="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, label=r"$\|\nabla f(x)\|_2$ (saturation map)")

    gd_xy = np.array([h["x"].numpy() for h in hist_gd])
    fx_xy = np.array([h["x"].numpy() for h in hist_fx])
    a1_xy = np.array([h["x"].numpy() for h in hist_a1])
    b_xy = np.array([h["b"].numpy() for h in hist_a1])

    ax.plot(gd_xy[:, 0], gd_xy[:, 1], "o", color="crimson", ms=4, label="naive GD (stuck)")
    ax.plot(fx_xy[:, 0], fx_xy[:, 1], "s", color="orange", ms=4, label="Algorithm 1, frozen baseline (stuck)")
    ax.plot(a1_xy[:, 0], a1_xy[:, 1], "o-", color="white", ms=2.5, lw=1.2, label="Algorithm 1, adaptive: x path")
    ax.plot(b_xy[:, 0], b_xy[:, 1], "d-", color="deepskyblue", ms=2.5, lw=1.2, label="Algorithm 1, adaptive: baseline path")

    ax.scatter(*x0.numpy(), c="cyan", s=110, marker="*", zorder=5, edgecolor="k", label=r"$x^{(0)}$")
    ax.scatter(*b0.numpy(), c="magenta", s=90, marker="D", zorder=5, edgecolor="k", label=r"$b^{(0)}$ (dead zone)")

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Staircase saturation map: GD and frozen baseline both stuck;\nonly the adaptive baseline escapes")
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "toy_heatmap_trajectories.png"), dpi=150)
    plt.close(fig)


def _fig_convergence(hist_gd, hist_fx, hist_a1):
    e_gd = [abs(h["e_x"]) for h in hist_gd if "e_x" in h]
    e_fx = [abs(h["x_e_x"]) for h in hist_fx if "x_e_x" in h]
    e_a1 = [abs(h["x_e_x"]) for h in hist_a1 if "x_e_x" in h]

    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.semilogy(e_gd, color="crimson", label="naive GD (no baseline)")
    ax.semilogy(e_fx, color="darkorange", ls="--", label="Algorithm 1, frozen baseline")
    ax.semilogy(e_a1, color="steelblue", label="Algorithm 1, adaptive baseline")
    ax.axhline(1e-3, color="gray", ls="--", lw=1, label=r"$\epsilon_x$ threshold")
    ax.set_xlabel("iteration $r$")
    ax.set_ylabel(r"$|e_x^{(r)}| = |f(x^{(r)}) - y^t|$")
    ax.set_title("A baseline only helps if it can move:\nsame start, only the adaptive one escapes")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "toy_convergence.png"), dpi=150)
    plt.close(fig)


def _fig_mechanism_multi_escape(hist_a1):
    gloc = [h["x_gloc_norm"] for h in hist_a1 if "x_gloc_norm" in h]
    s = [h["x_s"] for h in hist_a1 if "x_s" in h]
    kick_r = [h["r"] for h in hist_a1 if h.get("b_kicked")]

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.2), sharex=True)
    axes[0].semilogy(gloc, color="darkgreen", lw=1.2)
    axes[0].axhline(0.05, color="gray", ls="--", lw=1, label=r"$\tau_x$")
    for kr in kick_r:
        axes[0].axvline(kr, color="magenta", alpha=0.15, lw=2)
    axes[0].set_ylabel(r"$\|g_{loc,x}^{(r)}\|_2$")
    axes[0].set_title("The vehicle re-saturates multiple times crossing the staircase\n"
                       "(magenta bands = baseline probe-kick iterations)")
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].plot(s, color="purple", lw=1.2)
    for kr in kick_r:
        axes[1].axvline(kr, color="magenta", alpha=0.15, lw=2)
    axes[1].set_ylabel(r"gate $s_x^{(r)}$")
    axes[1].set_xlabel("iteration $r$")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].axhline(0.5, color="gray", ls=":", lw=0.8)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "toy_mechanism_gate.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run()
