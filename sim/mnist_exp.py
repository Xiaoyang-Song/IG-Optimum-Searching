"""
Experiment 2: MNIST target-confidence "attack" -- push a real digit image x
toward a target confidence y^t for a chosen target class, using naive GD vs
the full IG-guided Algorithm 1 with an adaptive baseline starting from a pure
black or pure white reference image (the "historical baseline" analog the user
asked for). f(x) = softmax(net(x))[target_class].
"""
import os
import time

import torch
# This node is heavily shared (dozens of concurrent jobs); torch's default
# intra-op thread count (one per core) makes every tiny forward/backward pass
# pay large thread-spawn/sync overhead for essentially no compute benefit on
# a network this small. Pinning to 1 thread turns ~45ms/matmul into ~0.1ms.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from ig_update import (
    run_algorithm1, run_gd, run_pgd_sign, run_mifgsm, run_adam,
    grad_and_value, avg_path_grad_and_ig, ig_weights,
)

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

torch.manual_seed(0)
np.random.seed(0)

TARGET_CLASS = 8
Y_T = 0.9
EPS_X = 1e-2
EPS_B = 1e-2


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def load_data(n_train=8000, n_test=1000):
    tfm = transforms.ToTensor()
    train = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tfm)
    test = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tfm)

    rng = np.random.RandomState(0)
    tr_idx = rng.choice(len(train), n_train, replace=False)
    te_idx = rng.choice(len(test), n_test, replace=False)

    Xtr = torch.stack([train[i][0].view(-1) for i in tr_idx])
    Ytr = torch.tensor([train[i][1] for i in tr_idx])
    Xte = torch.stack([test[i][0].view(-1) for i in te_idx])
    Yte = torch.tensor([test[i][1] for i in te_idx])
    return Xtr, Ytr, Xte, Yte


def train_model(Xtr, Ytr, Xte, Yte, epochs=6, batch_size=128, lr=1e-3):
    net = MLP()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = Xtr.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            logits = net(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(idx)
        with torch.no_grad():
            acc = (net(Xte).argmax(1) == Yte).float().mean().item()
        print(f"[mnist] epoch {ep+1}/{epochs} loss={tot_loss/n:.4f} test_acc={acc:.4f}")
    for p in net.parameters():
        p.requires_grad_(False)
    net.eval()
    return net


def make_f(net, target_class):
    def f(x):
        logits = net(x.unsqueeze(0))
        probs = F.softmax(logits, dim=1)
        return probs[0, target_class]
    return f


def project01(x):
    return x.clamp(0.0, 1.0)


def run():
    t0 = time.time()
    Xtr, Ytr, Xte, Yte = load_data()
    net = train_model(Xtr, Ytr, Xte, Yte)
    print(f"[mnist] data+train done in {time.time()-t0:.1f}s")

    f = make_f(net, TARGET_CLASS)

    with torch.no_grad():
        probs_all = F.softmax(net(Xte), dim=1)[:, TARGET_CLASS]

    _fig_saturation_scatter(f, Xte, probs_all)
    _fig_saturation_gradmaps(f, Xte, probs_all)

    # candidate attack set: images not already of the target class, with low
    # starting confidence for it (i.e. genuinely saturated at f~0).
    cand_idx = [i for i in range(len(Xte))
                if Yte[i].item() != TARGET_CLASS and probs_all[i].item() < 0.05]
    rng = np.random.RandomState(1)
    N = min(25, len(cand_idx))
    chosen = rng.choice(cand_idx, N, replace=False)
    print(f"[mnist] running attack comparison on N={N} images (of {len(cand_idx)} eligible)")

    R = 120
    black = torch.zeros(784)
    white = torch.ones(784)

    method_names = ["gd", "a1_black", "a1_black_frozen", "a1_white", "pgd", "mifgsm", "adam"]
    results = {m: [] for m in method_names}
    example = {}
    for rank, idx in enumerate(chosen):
        x0 = Xte[idx].clone()

        _, hist_gd = run_gd(f, x0, Y_T, R=R, eta_x=6.0, eps_x=EPS_X, project_x=project01)
        results["gd"].append([abs(h["e_x"]) for h in hist_gd if "e_x" in h])

        _, _, hist_a1b = run_algorithm1(
            f, x0, black, Y_T, R=R, eta_x=3.0, eta_b=0.5, eta_kick=0.4,
            tau_x=0.02, tau_b=0.02, eps_x=EPS_X, eps_b=EPS_B, M=15, delta_b=0.15,
            top_k=4, project_x=project01, project_b=project01,
        )
        results["a1_black"].append([abs(h["x_e_x"]) for h in hist_a1b if "x_e_x" in h])

        _, _, hist_a1bf = run_algorithm1(
            f, x0, black, Y_T, R=R, eta_x=3.0, eta_b=0.0, eta_kick=0.0,
            tau_x=0.02, tau_b=1.0, eps_x=EPS_X, eps_b=EPS_B, M=15,
            project_x=project01, project_b=project01, adapt_baseline=False,
        )
        results["a1_black_frozen"].append([abs(h["x_e_x"]) for h in hist_a1bf if "x_e_x" in h])

        _, _, hist_a1w = run_algorithm1(
            f, x0, white, Y_T, R=R, eta_x=3.0, eta_b=0.5, eta_kick=0.4,
            tau_x=0.02, tau_b=0.02, eps_x=EPS_X, eps_b=EPS_B, M=15, delta_b=0.15,
            top_k=4, project_x=project01, project_b=project01,
        )
        results["a1_white"].append([abs(h["x_e_x"]) for h in hist_a1w if "x_e_x" in h])

        _, hist_pgd = run_pgd_sign(f, x0, Y_T, R=R, eta_x=0.02, eps_x=EPS_X, project_x=project01)
        results["pgd"].append([abs(h["e_x"]) for h in hist_pgd if "e_x" in h])

        _, hist_mifgsm = run_mifgsm(f, x0, Y_T, R=R, eta_x=0.02, mu=0.9, eps_x=EPS_X, project_x=project01)
        results["mifgsm"].append([abs(h["e_x"]) for h in hist_mifgsm if "e_x" in h])

        _, hist_adam = run_adam(f, x0, Y_T, R=R, eta_x=0.3, eps_x=EPS_X, project_x=project01)
        results["adam"].append([abs(h["e_x"]) for h in hist_adam if "e_x" in h])

        if rank == 0:
            example["x0"] = x0
            example["hist_gd"] = hist_gd
            example["hist_a1"] = hist_a1b
        print(f"[mnist]  image {rank+1}/{N} (idx {idx}, true={Yte[idx].item()}, f0={probs_all[idx].item():.4f}): "
              f"GD={len(hist_gd)-1} A1-black={len(hist_a1b)-1} A1-frozen={len(hist_a1bf)-1} "
              f"A1-white={len(hist_a1w)-1} PGD={len(hist_pgd)-1} MI-FGSM={len(hist_mifgsm)-1} Adam={len(hist_adam)-1}")

    _print_summary(results, R)
    _fig_convergence(results, R)

    # dedicated hardest-example failure case (lowest starting confidence)
    hard_idx = int(torch.argmin(probs_all).item())
    x0_hard = Xte[hard_idx].clone()
    _, hist_gd_hard = run_gd(f, x0_hard, Y_T, R=R, eta_x=6.0, eps_x=EPS_X, project_x=project01)
    _, _, hist_a1_hard = run_algorithm1(
        f, x0_hard, black, Y_T, R=R, eta_x=3.0, eta_b=0.5, eta_kick=0.4,
        tau_x=0.02, tau_b=0.02, eps_x=EPS_X, eps_b=EPS_B, M=15, delta_b=0.15,
        top_k=4, project_x=project01, project_b=project01,
    )
    hard_example = dict(x0=x0_hard, hist_gd=hist_gd_hard, hist_a1=hist_a1_hard)
    print(f"[mnist] hardest example idx={hard_idx} f0={probs_all[hard_idx].item():.2e}: "
          f"GD={len(hist_gd_hard)-1} iters, A1={len(hist_a1_hard)-1} iters")

    _fig_failure_single(hard_example)
    _fig_ig_heatmap(f, hard_example["x0"], black)
    _fig_filmstrips(hard_example, black, white, f)
    _fig_mechanism(hard_example)

    print(f"[mnist] total time {time.time()-t0:.1f}s")


def _success_iters(err_hist, eps):
    for r, e in enumerate(err_hist):
        if e <= eps:
            return r
    return None


def _print_summary(results, R):
    print("\n[mnist] === convergence summary (target y^t=%.2f, eps=%.3f, R=%d) ===" % (Y_T, EPS_X, R))
    for name, runs in results.items():
        iters = [_success_iters(r, EPS_X) for r in runs]
        succ = [i for i in iters if i is not None]
        rate = len(succ) / len(iters)
        med = float(np.median(succ)) if succ else float("nan")
        print(f"  {name:10s}: success_rate={rate:.2f}  median_iters_to_target={med:.1f}  n={len(iters)}")


def _fig_saturation_scatter(f, Xte, probs_all):
    idxs = np.random.RandomState(2).choice(len(Xte), 400, replace=False)
    fs, gns = [], []
    for i in idxs:
        g, y = grad_and_value(f, Xte[i])
        fs.append(y.item())
        gns.append(g.norm().item())
    fs, gns = np.array(fs), np.array(gns)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    sc = ax.scatter(fs, gns, s=10, alpha=0.5, c=fs, cmap="plasma")
    ax.set_xlabel(f"$f(x)$ = softmax prob. of class {TARGET_CLASS}")
    ax.set_ylabel(r"$\|\nabla f(x)\|_2$")
    ax.set_title("Gradient saturation on real MNIST images\n(gradient collapses near $f\\approx0$ and $f\\approx1$)")
    fig.colorbar(sc, ax=ax, label="$f(x)$")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "mnist_saturation_scatter.png"), dpi=150)
    plt.close(fig)


def _fig_saturation_gradmaps(f, Xte, probs_all):
    sat_idx = int(torch.argmin(probs_all).item())
    mid = (probs_all - 0.5).abs()
    sens_idx = int(torch.argmin(mid).item())

    fig, axes = plt.subplots(2, 2, figsize=(7, 7))
    for col, (idx, tag) in enumerate([(sat_idx, "saturated"), (sens_idx, "sensitive")]):
        x = Xte[idx]
        g, y = grad_and_value(f, x)
        axes[0, col].imshow(x.view(28, 28), cmap="gray")
        axes[0, col].set_title(f"{tag} example\n$f(x)$={y.item():.4f}")
        axes[0, col].axis("off")
        im = axes[1, col].imshow(g.view(28, 28).abs(), cmap="inferno")
        axes[1, col].set_title(f"$|\\nabla f(x)|$, $\\|\\cdot\\|_2$={g.norm().item():.2e}")
        axes[1, col].axis("off")
        fig.colorbar(im, ax=axes[1, col], fraction=0.046)
    fig.suptitle("Input-gradient magnitude: saturated vs sensitive example")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "mnist_saturation_gradmaps.png"), dpi=150)
    plt.close(fig)


def _plot_mean_std(ax, results, subset, colors, labels, band=True):
    for name in subset:
        runs = results[name]
        maxlen = max(len(r) for r in runs)
        padded = np.array([r + [r[-1]] * (maxlen - len(r)) for r in runs])
        mean = padded.mean(axis=0)
        std = padded.std(axis=0)
        xs = np.arange(maxlen)
        ax.plot(xs, mean, color=colors[name], label=labels[name])
        if band:
            ax.fill_between(xs, np.clip(mean - std, 1e-6, None), mean + std,
                             color=colors[name], alpha=0.15)


def _fig_convergence(results, R):
    colors = {"gd": "crimson", "a1_black": "steelblue", "a1_black_frozen": "darkorange",
              "a1_white": "seagreen", "pgd": "mediumseagreen", "mifgsm": "darkviolet",
              "adam": "goldenrod"}
    labels = {"gd": "naive GD (no baseline)",
              "a1_black": "Algorithm 1, adaptive baseline (black)",
              "a1_black_frozen": "Algorithm 1, frozen baseline (black)",
              "a1_white": "Algorithm 1, adaptive baseline (white)",
              "pgd": "PGD / BIM (sign)", "mifgsm": "MI-FGSM (momentum-sign)",
              "adam": "Adam"}
    N = len(results["gd"])

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    _plot_mean_std(ax, results, ["gd", "a1_black_frozen", "a1_black", "a1_white"], colors, labels)
    ax.set_yscale("log")
    ax.axhline(EPS_X, color="gray", ls="--", lw=1, label=r"$\epsilon_x$ threshold")
    ax.set_xlabel("iteration $r$")
    ax.set_ylabel(r"mean $|e_x^{(r)}|$ over sampled images (±1 std band)")
    ax.set_title(f"Does the baseline need to adapt? (N={N} MNIST images)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "mnist_convergence.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    _plot_mean_std(ax, results, ["gd", "adam", "pgd", "mifgsm", "a1_black"], colors, labels, band=False)
    ax.set_yscale("log")
    ax.axhline(EPS_X, color="gray", ls="--", lw=1, label=r"$\epsilon_x$ threshold")
    ax.set_xlabel("iteration $r$")
    ax.set_ylabel(r"mean $|e_x^{(r)}|$ over sampled images")
    ax.set_title(f"Algorithm 1 vs. popular adversarial-attack update rules (N={N} MNIST images)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "mnist_vs_baselines.png"), dpi=150)
    plt.close(fig)


def _fig_failure_single(example):
    e_gd = [abs(h["e_x"]) for h in example["hist_gd"] if "e_x" in h]
    e_a1 = [abs(h["x_e_x"]) for h in example["hist_a1"] if "x_e_x" in h]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.semilogy(e_gd, color="crimson", label="naive GD")
    ax.semilogy(e_a1, color="steelblue", label="Algorithm 1 (IG-guided)")
    ax.axhline(EPS_X, color="gray", ls="--", lw=1)
    ax.set_xlabel("iteration $r$")
    ax.set_ylabel(r"$|e_x^{(r)}|$")
    ax.set_title("Single deeply-saturated MNIST image: GD stalls, Algorithm 1 escapes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "mnist_failure_single.png"), dpi=150)
    plt.close(fig)


def _fig_ig_heatmap(f, x0, baseline):
    _, ig = avg_path_grad_and_ig(f, baseline, x0, M=20)
    w = ig_weights(ig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    axes[0].imshow(x0.view(28, 28), cmap="gray")
    axes[0].set_title("input $x^{(0)}$")
    axes[0].axis("off")
    im1 = axes[1].imshow(ig.view(28, 28).abs(), cmap="inferno")
    axes[1].set_title(r"$|IG_j|$ (attribution)")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    im2 = axes[2].imshow(w.view(28, 28), cmap="viridis")
    axes[2].set_title(r"IG coordinate weight $w_j$")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)
    fig.suptitle("IG attribution relative to black baseline")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "mnist_ig_heatmap.png"), dpi=150)
    plt.close(fig)


def _fig_filmstrips(example, black, white, f):
    hist_gd = example["hist_gd"]
    hist_a1 = example["hist_a1"]
    steps = sorted(set([0] + [int(s) for s in np.linspace(0, len(hist_a1) - 1, 6)]))

    fig, axes = plt.subplots(2, len(steps), figsize=(2 * len(steps), 4.3))
    for c, r in enumerate(steps):
        r_gd = min(r, len(hist_gd) - 1)
        axes[0, c].imshow(hist_gd[r_gd]["x"].view(28, 28), cmap="gray", vmin=0, vmax=1)
        axes[0, c].set_title(f"r={r_gd}", fontsize=8)
        axes[0, c].axis("off")
        axes[1, c].imshow(hist_a1[r]["x"].view(28, 28), cmap="gray", vmin=0, vmax=1)
        axes[1, c].set_title(f"r={r}", fontsize=8)
        axes[1, c].axis("off")
    axes[0, 0].set_ylabel("GD", fontsize=10)
    axes[1, 0].set_ylabel("Algorithm 1", fontsize=10)
    fig.suptitle("Perturbed image over iterations: naive GD (top) vs Algorithm 1 (bottom)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "mnist_filmstrip_x.png"), dpi=150)
    plt.close(fig)

    # baseline evolution from black and white starts
    def _baseline_hist(f, x0, b0):
        _, _, h = run_algorithm1(
            f, x0, b0, Y_T, R=120, eta_x=3.0, eta_b=0.5, eta_kick=0.4,
            tau_x=0.02, tau_b=0.02, eps_x=EPS_X, eps_b=EPS_B, M=15, delta_b=0.15,
            top_k=4, project_x=project01, project_b=project01,
        )
        return h

    h_black = _baseline_hist(f, example["x0"], black)
    h_white = _baseline_hist(f, example["x0"], white)
    steps_b = sorted(set([0] + [int(s) for s in np.linspace(0, len(h_black) - 1, 6)]))

    fig, axes = plt.subplots(2, len(steps_b), figsize=(2 * len(steps_b), 4.3))
    for c, r in enumerate(steps_b):
        rb = min(r, len(h_black) - 1)
        rw = min(r, len(h_white) - 1)
        axes[0, c].imshow(h_black[rb]["b"].view(28, 28), cmap="gray", vmin=0, vmax=1)
        axes[0, c].set_title(f"r={rb}", fontsize=8)
        axes[0, c].axis("off")
        axes[1, c].imshow(h_white[rw]["b"].view(28, 28), cmap="gray", vmin=0, vmax=1)
        axes[1, c].set_title(f"r={rw}", fontsize=8)
        axes[1, c].axis("off")
    fig.suptitle("Adaptive baseline evolution: from black start (top) vs white start (bottom)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "mnist_filmstrip_baseline.png"), dpi=150)
    plt.close(fig)


def _fig_mechanism(example):
    hist_a1 = example["hist_a1"]
    gloc = [h["x_gloc_norm"] for h in hist_a1 if "x_gloc_norm" in h]
    s = [h["x_s"] for h in hist_a1 if "x_s" in h]

    fig, axes = plt.subplots(2, 1, figsize=(6.5, 6), sharex=True)
    axes[0].semilogy(gloc, color="darkgreen")
    axes[0].axhline(0.02, color="gray", ls="--", lw=1, label=r"$\tau_x$")
    axes[0].set_ylabel(r"$\|g_{loc,x}^{(r)}\|_2$")
    axes[0].set_title("MNIST example: local-gradient saturation and escape gate")
    axes[0].legend()

    axes[1].plot(s, color="purple")
    axes[1].set_ylabel(r"gate $s_x^{(r)}$")
    axes[1].set_xlabel("iteration $r$")
    axes[1].set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "mnist_mechanism_gate.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run()
