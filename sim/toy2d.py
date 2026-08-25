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

    _fig_value_heatmap(x0, b0, y_t)
    _fig_heatmap_and_trajectories(x0, b0, hist_gd, hist_fx, hist_a1)
    _fig_convergence(hist_gd, hist_fx, hist_a1)
    _fig_mechanism_multi_escape(hist_a1)
    _fig_vs_baselines(x0, b0, y_t, project, R)


def _fig_vs_baselines(x0, b0, y_t, project, R):
    """Algorithm 1 vs three popular existing adversarial-attack update rules,
    all started from the same deeply-saturated x0, no baseline needed for the
    literature methods (sign/momentum/Adam all act on x alone). Uses the same
    b0 as the rest of Experiment 1 (the dead zone), not a separately-chosen
    "easy" reference -- so this comparison and the baseline-ablation one are
    directly about the same starting problem, not two different setups."""
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

    _fig_vs_baselines_heatmap(x0, b0, h_gd, h_adam, h_pgd, h_mifgsm, h_a1)


def _fig_vs_baselines_heatmap(x0, b0, h_gd, h_adam, h_pgd, h_mifgsm, h_a1):
    """Same saturation map as toy_heatmap_trajectories.png (same x0, b0), but
    with the literature baselines' own (x1,x2)-space paths overlaid instead
    of the dead-zone baseline-adaptation story -- shows *where* GD/Adam get
    stuck, and the back-and-forth oscillation PGD/MI-FGSM's fixed-size sign
    steps trace out once they reach the target region, right next to
    Algorithm 1's clean approach-and-settle path. The domain is sized to the
    trajectories themselves (with margin) rather than clipped, since
    PGD/MI-FGSM's paths range far past the other two figures' fixed window."""
    # MI-FGSM and PGD take nearly identical paths here (both step (+,+) for
    # almost the whole run), so one would otherwise render directly on top
    # of the other -- draw PGD noticeably thicker/dashed so both remain
    # visible, and use high-contrast colors against the saturation heatmap.
    paths = [
        (h_mifgsm, "x", "-", "magenta", "MI-FGSM", 2.2, 1),
        (h_pgd, "x", "--", "lime", "PGD / BIM (sign)", 3.2, 2),
        (h_gd, "x", "o", "crimson", "naive GD (stuck)", 0, 4),
        (h_adam, "x", "o", "goldenrod", "Adam (stuck)", 0, 4),
        (h_a1, "x", "o-", "white", "Algorithm 1: x path", 1.2, 3),
        (h_a1, "b", "d-", "deepskyblue", "Algorithm 1: baseline path", 1.2, 3),
    ]
    all_xy = np.concatenate([np.array([h[key].numpy() for h in hist]) for hist, key, *_ in paths], axis=0)
    margin = 1.0
    x1_lo, x1_hi = all_xy[:, 0].min() - margin, all_xy[:, 0].max() + margin
    x2_lo, x2_hi = all_xy[:, 1].min() - margin, all_xy[:, 1].max() + margin

    xs1 = np.linspace(x1_lo, x1_hi, 400)
    xs2 = np.linspace(x2_lo, x2_hi, 400)
    X1, X2 = np.meshgrid(xs1, xs2)
    GN = gradnorm_np(X1, X2)

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.pcolormesh(X1, X2, GN, shading="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, label=r"$\|\nabla f(x)\|_2$ (saturation map)")

    for hist, key, style, color, label, lw, z in paths:
        xy = np.array([h[key].numpy() for h in hist])
        ax.plot(xy[:, 0], xy[:, 1], style, color=color, ms=3.5, lw=lw, label=label, alpha=0.9, zorder=z)

    ax.scatter(*x0.numpy(), c="cyan", s=130, marker="*", zorder=5, edgecolor="k", label=r"$x^{(0)}$")
    ax.scatter(*b0.numpy(), c="orange", s=110, marker="D", zorder=5, edgecolor="k", label=r"$b^{(0)}$")

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Where each method's path actually goes (same $x^{(0)}, b^{(0)}$ as Fig. toy_heatmap_trajectories):\n"
                 "GD/Adam stuck at start, PGD/MI-FGSM oscillate far past the target, Algorithm 1 settles")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "toy_vs_baselines_heatmap.png"), dpi=150)
    plt.close(fig)


def _fig_value_heatmap(x0, b0, y_t):
    """Plain f(x1,x2) value map -- what the model actually outputs at each
    point, as opposed to the *saturation* map (||grad f||) in the other
    figure. Four risers (at u=-9,-3,3,9) separate FIVE flat treads (at
    f~0, 0.25, 0.5, 0.75, 1); each tread is labeled with its value, and so
    is the starting point x^(0)."""
    xs1 = np.linspace(-8, 4, 320)
    xs2 = np.linspace(-6, 4, 320)
    X1, X2 = np.meshgrid(xs1, xs2)
    FV = f_np(X1, X2)
    x0n = x0.numpy()

    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.pcolormesh(X1, X2, FV, shading="auto", cmap="cividis", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="$f(x_1,x_2)$")

    riser_cs = ax.contour(X1, X2, FV, levels=[0.125, 0.375, 0.625, 0.875], colors="white",
                           linewidths=0.9)
    # manual label points must lie ON their line (u=3*x1+x2=const), else clabel
    # snaps to whichever contour line is nearest -- compute them exactly.
    ax.clabel(riser_cs, inline=True, fmt=lambda v: f"{v:.3f}", fontsize=8, colors="white",
              manual=[(-3.0, 0.0), (-2.0, 3.0), (1.0, 0.0), (3.667, -2.0)])

    target_cs = ax.contour(X1, X2, FV, levels=[y_t], colors="magenta", linewidths=2)
    ax.clabel(target_cs, inline=True, fmt=lambda v: f"target f={v:.2f}", fontsize=9,
              colors="magenta", manual=[(2.288, 2.0)])

    # label each of the five flat treads at a representative point inside it
    # (verified: u=3*x1+x2 must sit at the tread's center, not just past a riser)
    tread_pts = [(-6.0, 3.0, 0.0), (-1.0, -3.0, 0.25), (1.0, -3.0, 0.5),
                 (3.0, -3.0, 0.75), (3.5, 1.5, 1.0)]
    for tx, ty, tv in tread_pts:
        ax.text(tx, ty, f"f≈{tv:.2f}", color="white", fontsize=9, ha="center",
                 bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.45, lw=0))

    ax.scatter(*x0n, c="cyan", s=110, marker="*", zorder=5, edgecolor="k", label=r"$x^{(0)}$")
    ax.annotate(f"$x^{{(0)}}$: $f(x^{{(0)}})$={f(x0).item():.1e}", xy=x0n,
                xytext=(x0n[0] - 0.3, x0n[1] - 0.9), color="cyan", fontsize=9, ha="right",
                arrowprops=dict(arrowstyle="->", color="cyan", lw=1))
    ax.scatter(*b0.numpy(), c="orange", s=90, marker="D", zorder=5, edgecolor="k", label=r"$b^{(0)}$")

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(r"$f(x_1,x_2)$ values (not saturation): 5 flat treads separated by 4 risers")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "toy_value_heatmap.png"), dpi=150)
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
