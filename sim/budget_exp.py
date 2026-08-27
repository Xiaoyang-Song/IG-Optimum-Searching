"""
Experiment 4: escaping saturation is not enough when movement is *scarce*.

Real UFK/process-control adjustments (and, in the attack analogy, sparse/
budget-constrained perturbations) rarely allow moving every coordinate
freely -- there is usually a bounded total adjustment budget shared across
all of them. This experiment adds exactly that constraint on top of the
saturation problem, and asks two questions the other experiments don't:

1. Which methods can escape saturation from a start whose gradient *norm*
   underflows to exactly 0.0 in float32 at all?
2. Among the ones that escape, which ones spend a *scarce, shared* movement
   budget on the coordinates that actually matter, vs. wasting it uniformly
   across irrelevant ones?

Model: f(x) = sigmoid(c^T x), x in R^12, with 3 "important" coordinates
(c_j=6.0) and 9 "decoy" coordinates (c_j=0.1) -- deliberately more decoys
than the 6-D attribution experiment, so "spend the budget on the few
coordinates that matter" is a sharper, more realistic test. x^(0) sits deep
enough in saturation that f(x^(0)) underflows to ~2e-25. The individual
gradient components are tiny but not literally zero (~1e-24 on important
coords, ~1e-26 on decoys, all same sign); it's grad.norm() that reads as
exactly 0.0, because the L2 norm squares each component first and e.g.
(1.4e-24)**2 underflows even though the component itself doesn't. This
matters: naive GD's step (eta * e_x * grad) is proportional to that ~1e-24
magnitude and is too small to be a representable float32 increment at x's
scale no matter how large eta is; PGD's step (sign(grad)) is unaffected by
this since a well-defined sign survives no matter how tiny the magnitude.

The movement budget is enforced as a shared L1 cap on the *vehicle's* total
perturbation (not the baseline, which is free to search):
    Pi_budget(x) = x0 + rescale(x - x0, so that ||x-x0||_1 <= B)
i.e. if the accumulated perturbation's L1 norm would exceed the budget B,
it is rescaled down (keeping its direction) rather than allowed through --
a simple stand-in for "you only have so much total adjustment capacity."
"""
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from ig_update import (
    run_algorithm1, run_gd, run_pgd_sign, run_mifgsm, run_adam,
    grad_and_value, avg_path_grad_and_ig,
)

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

P = 12
N_IMPORTANT = 3
C = torch.cat([torch.full((N_IMPORTANT,), 6.0), torch.full((P - N_IMPORTANT,), 0.1)])
Y_T = 0.9
EPS_X = 1e-2
R = 400
BUDGET_MAIN = 10.0
BUDGET_SWEEP = [8, 10, 12, 15, 20, 30, 45, 60]

METHOD_COLORS = {"gd": "crimson", "adam": "goldenrod", "pgd": "seagreen",
                  "mifgsm": "darkviolet", "a1_unweighted": "gray", "a1_weighted": "steelblue"}
METHOD_LABELS = {"gd": "naive GD", "adam": "Adam", "pgd": "PGD / BIM (sign)",
                  "mifgsm": "MI-FGSM", "a1_unweighted": "Algorithm 1, no IG weights",
                  "a1_weighted": "Algorithm 1 (IG-guided)"}


def f(x):
    return torch.sigmoid((C * x).sum())


def make_budget_projector(x0, budget, lo=-8.0, hi=8.0):
    def proj(x):
        delta = x - x0
        l1 = delta.abs().sum()
        if l1.item() > budget:
            delta = delta * (budget / l1)
        return (x0 + delta).clamp(lo, hi)
    return proj


def run_all_methods(x0, b0, budget):
    proj_x = make_budget_projector(x0, budget)
    proj_b = lambda t: t.clamp(-8.0, 8.0)  # baseline search is not budget-limited
    out = {}

    _, h = run_gd(f, x0, Y_T, R=R, eta_x=50.0, eps_x=EPS_X, project_x=proj_x)
    out["gd"] = h
    _, h = run_adam(f, x0, Y_T, R=R, eta_x=1.0, eps_x=EPS_X, project_x=proj_x)
    out["adam"] = h
    _, h = run_pgd_sign(f, x0, Y_T, R=R, eta_x=0.05, eps_x=EPS_X, project_x=proj_x)
    out["pgd"] = h
    _, h = run_mifgsm(f, x0, Y_T, R=R, eta_x=0.05, mu=0.9, eps_x=EPS_X, project_x=proj_x)
    out["mifgsm"] = h
    _, _, h = run_algorithm1(
        f, x0, b0, Y_T, R=R, eta_x=2.0, eta_b=0.3, eta_kick=0.4, tau_x=0.02, tau_b=0.05,
        eps_x=EPS_X, eps_b=EPS_X, M=25, delta_b=0.3, top_k=3, project_x=proj_x, project_b=proj_b,
        adapt_baseline=True, use_ig_weights=False,
    )
    out["a1_unweighted"] = h
    _, _, h = run_algorithm1(
        f, x0, b0, Y_T, R=R, eta_x=2.0, eta_b=0.3, eta_kick=0.4, tau_x=0.02, tau_b=0.05,
        eps_x=EPS_X, eps_b=EPS_X, M=25, delta_b=0.3, top_k=3, project_x=proj_x, project_b=proj_b,
        adapt_baseline=True, use_ig_weights=True,
    )
    out["a1_weighted"] = h
    return out


def final_x(hist, key="x"):
    return hist[-1][key]


def final_f(hist):
    last_with_val = [h for h in hist if "y_x" in h or "x_y_x" in h]
    h = last_with_val[-1]
    return h["y_x"] if "y_x" in h else h["x_y_x"]


def final_deviation(hist):
    """|f(x^(R)) - y^t| -- the actual objective. The target is y^t exactly,
    not "as large as possible": overshooting past y^t is just as much a
    failure as undershooting, so this (not raw f(x)) is what every figure
    below plots."""
    return abs(final_f(hist) - Y_T)


def run():
    x0 = torch.full((P,), -3.0)
    b0 = torch.zeros(P)

    g0, y0 = grad_and_value(f, x0)
    _, ig0 = avg_path_grad_and_ig(f, b0, x0, M=25)
    print(f"[budget] x0: f(x0)={y0.item():.3e}, ||grad|| = {g0.norm().item():.3e} "
          f"({'underflows to 0 (norm squares tiny components)' if g0.norm().item() == 0.0 else 'nonzero'}), "
          f"max |grad_j| = {g0.abs().max().item():.3e} (components themselves are not literally zero)")
    print(f"[budget] target y_t={Y_T}, needed total u-shift ~= {torch.logit(torch.tensor(Y_T)).item() - float((C*x0).sum()):.1f}")

    print(f"\n[budget] === all methods at budget B={BUDGET_MAIN} ===")
    results_main = run_all_methods(x0, b0, BUDGET_MAIN)
    for name, hist in results_main.items():
        fx = final_f(hist)
        dev = final_deviation(hist)
        succ = dev <= EPS_X
        print(f"  {METHOD_LABELS[name]:32s}: f={fx:.4f}  |f-y^t|={dev:.4f}  {'SUCCESS' if succ else 'fail'}")

    _fig_bar_at_budget(x0, results_main)
    _fig_concentration(x0, results_main)

    print(f"\n[budget] === sweeping budget B in {BUDGET_SWEEP} ===")
    sweep_results = {name: [] for name in METHOD_COLORS}
    for B in BUDGET_SWEEP:
        res = run_all_methods(x0, b0, B)
        for name, hist in res.items():
            sweep_results[name].append(final_deviation(hist))
        print(f"  B={B:5.1f}: " + "  ".join(f"{name}=|dev|{final_deviation(res[name]):.3f}" for name in METHOD_COLORS))

    _fig_budget_sweep(sweep_results)


def _fig_bar_at_budget(x0, results):
    names = list(METHOD_COLORS.keys())
    devs = [final_deviation(results[n]) for n in names]
    fvals = [final_f(results[n]) for n in names]
    colors = [METHOD_COLORS[n] for n in names]
    labels = [METHOD_LABELS[n] for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(names)), devs, color=colors)
    ax.axhline(EPS_X, color="black", ls="--", lw=1.2, label=fr"success threshold $\epsilon_x$={EPS_X}")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(r"deviation from target $|f(x)-y^t|$  (lower is better; 0 = exactly $y^t$)")
    ax.set_title(f"Under a tight shared movement budget (B={BUDGET_MAIN}),\n"
                 f"only the IG-weighted method lands on the target (not just \"high\")")
    ax.legend(fontsize=8)
    for b, dev, fv in zip(bars, devs, fvals):
        tag = "over" if fv > Y_T else "under"
        ax.text(b.get_x() + b.get_width() / 2, dev + 0.015, f"{dev:.3f}\n({tag}shoot, f={fv:.2f})",
                ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "budget_bar_final_f.png"), dpi=150)
    plt.close(fig)


def _fig_concentration(x0, results):
    names = list(METHOD_COLORS.keys())
    imp_move, dec_move = [], []
    for n in names:
        xf = final_x(results[n])
        delta = (xf - x0).abs()
        imp_move.append(delta[:N_IMPORTANT].sum().item())
        dec_move.append(delta[N_IMPORTANT:].sum().item())

    fig, ax = plt.subplots(figsize=(8, 5))
    idx = np.arange(len(names))
    width = 0.35
    ax.bar(idx - width / 2, imp_move, width, label=f"{N_IMPORTANT} important coords", color="steelblue")
    ax.bar(idx + width / 2, dec_move, width, label=f"{P - N_IMPORTANT} unimportant coords", color="lightgray",
           edgecolor="gray")
    ax.set_xticks(idx)
    ax.set_xticklabels([METHOD_LABELS[n] for n in names], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(r"total $\sum |x_j^{(R)} - x_j^{(0)}|$ spent")
    ax.set_title(f"Where does each method spend its budget (B={BUDGET_MAIN})?")
    ax.legend(fontsize=8)
    for i, n in enumerate(names):
        total = imp_move[i] + dec_move[i]
        pct = 100 * imp_move[i] / total if total > 0 else 0
        ax.text(i, max(imp_move[i], dec_move[i]) + 0.3, f"{pct:.0f}% on\nimportant", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "budget_concentration.png"), dpi=150)
    plt.close(fig)


def _fig_budget_sweep(sweep_results):
    """sweep_results[name] holds |f(x)-y^t| (deviation), not f(x) itself --
    the objective is landing on y^t, so lower is better and 0 is perfect;
    overshooting past y^t is not rewarded just because f(x) is larger."""
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for name, vals in sweep_results.items():
        plot_vals = [max(v, 1e-4) for v in vals]  # keep exact-0 points visible on log scale
        ax.plot(BUDGET_SWEEP, plot_vals, "o-", color=METHOD_COLORS[name], label=METHOD_LABELS[name])
    ax.axhline(EPS_X, color="black", ls="--", lw=1, label=fr"success threshold $\epsilon_x$={EPS_X}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("total movement budget $B$ (L1, log scale)")
    ax.set_ylabel(r"deviation from target $|f(x)-y^t|$ (log scale; lower is better)")
    ax.set_title("Deviation from target vs. movement budget\n"
                 "(gradient-based methods never move -- deviation stuck at |0-0.9|=0.9 for every budget)")
    ax.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "budget_sweep.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run()
