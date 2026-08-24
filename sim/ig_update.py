"""
Core implementation of Algorithm 1 ("IG-Guided Current-Vehicle Update with Adaptive
Baseline Search") from IG_Guided_Target_Update_Formulation.pdf, plus plain gradient
descent for comparison.

Everything here is framework-generic: `f` is any callable mapping a 1-D torch.Tensor
to a scalar torch.Tensor, so the same code drives both the 2D synthetic toy problem
and the MNIST target-confidence experiment.
"""
from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Section 3: local gradient, average path gradient, Integrated Gradients
# ---------------------------------------------------------------------------

def grad_and_value(f, x: torch.Tensor):
    """Local gradient g_loc = df/dx and value f(x) at the current point."""
    xi = x.detach().clone().requires_grad_(True)
    y = f(xi)
    (g,) = torch.autograd.grad(y, xi)
    return g.detach(), y.detach()


def avg_path_grad_and_ig(f, b: torch.Tensor, x: torch.Tensor, M: int = 20):
    """
    Eq. (Sec 7): average path gradient g_bar_path and IG_j along the straight
    line from baseline b to current point x, using M equally spaced points
    alpha_m = m/M, m = 1..M.
    """
    b_d, x_d = b.detach(), x.detach()
    diff = x_d - b_d
    total = torch.zeros_like(x_d)
    for m in range(1, M + 1):
        alpha = m / M
        xi = (b_d + alpha * diff).clone().requires_grad_(True)
        yi = f(xi)
        (gi,) = torch.autograd.grad(yi, xi)
        total += gi.detach()
    g_path = total / M
    ig = diff * g_path
    return g_path, ig


# ---------------------------------------------------------------------------
# Section 4.1: IG-based coordinate weighting
# ---------------------------------------------------------------------------

def ig_weights(ig: torch.Tensor, eps_w: float = 1e-6) -> torch.Tensor:
    a = ig.abs() + eps_w
    p = a.numel()
    return p * a / a.sum()


# ---------------------------------------------------------------------------
# Section 4.2: saturation-aware hybrid direction (smooth gate version)
# ---------------------------------------------------------------------------

def hybrid_direction(g_loc: torch.Tensor, g_path: torch.Tensor, tau_x: float):
    gloc_norm = g_loc.norm().item()
    s = torch.exp(torch.tensor(-(gloc_norm ** 2) / (tau_x ** 2))).item()
    g_hybrid = (1 - s) * g_loc + s * g_path
    return g_hybrid, s, gloc_norm


# ---------------------------------------------------------------------------
# Current-vehicle update (Algorithm 1, lines 7-10) and plain-GD comparator (Sec 2)
# ---------------------------------------------------------------------------

def vehicle_update(f, x: torch.Tensor, b: torch.Tensor, y_t: float,
                    eta_x: float, tau_x: float, M: int = 20, project=None,
                    use_ig_weights: bool = True):
    g_path, ig = avg_path_grad_and_ig(f, b, x, M)
    w = ig_weights(ig) if use_ig_weights else torch.ones_like(ig)
    g_loc, y_x = grad_and_value(f, x)
    g_hybrid, s, gloc_norm = hybrid_direction(g_loc, g_path, tau_x)
    e_x = (y_x - y_t).item()
    x_new = x - eta_x * e_x * w * g_hybrid
    if project is not None:
        x_new = project(x_new)
    info = dict(e_x=e_x, y_x=y_x.item(), s=s, gloc_norm=gloc_norm,
                gpath_norm=g_path.norm().item(), ig=ig, w=w)
    return x_new, info


def gd_update(f, x: torch.Tensor, y_t: float, eta_x: float, project=None):
    g_loc, y_x = grad_and_value(f, x)
    e_x = (y_x - y_t).item()
    x_new = x - eta_x * e_x * g_loc
    if project is not None:
        x_new = project(x_new)
    info = dict(e_x=e_x, y_x=y_x.item(), gloc_norm=g_loc.norm().item())
    return x_new, info


# ---------------------------------------------------------------------------
# Section 5: adaptive baseline search + Section 5.1: directional path probing
# ---------------------------------------------------------------------------

def directional_path_sensitivity(f, b: torch.Tensor, d: torch.Tensor,
                                  delta_b: float, M: int = 10) -> float:
    """G_k = average over the probe path b -> b + delta_b*d of grad(f)^T d."""
    b_d = b.detach()
    total = 0.0
    for m in range(1, M + 1):
        alpha = m / M
        xi = (b_d + alpha * delta_b * d).clone().requires_grad_(True)
        yi = f(xi)
        (gi,) = torch.autograd.grad(yi, xi)
        total += torch.dot(gi.detach().flatten(), d.flatten()).item()
    return total / M


def build_baseline_candidates(x: torch.Tensor, b: torch.Tensor, ig: torch.Tensor,
                               top_k: int = 5, prev_dir: torch.Tensor | None = None):
    """
    Candidate directions per Sec 5.1: +/- e_j for IG-important coordinates,
    the previous successful baseline direction, and the normalized current
    deviation (x - b) as an optional probe.
    """
    cands = {}
    top = torch.topk(ig.abs().flatten(), k=min(top_k, ig.numel())).indices
    for j in top.tolist():
        e_j = torch.zeros_like(b).flatten()
        e_j[j] = 1.0
        e_j = e_j.view_as(b)
        cands[f"+e{j}"] = e_j
        cands[f"-e{j}"] = -e_j
    if prev_dir is not None:
        cands["prev"] = prev_dir
    dev = x - b
    if dev.norm().item() > 1e-8:
        cands["dev"] = dev / dev.norm()
    return cands


def baseline_update(f, b: torch.Tensor, y_t: float, eta_b: float, tau_b: float,
                     eta_kick: float, candidates: dict, delta_b: float = 0.1,
                     M: int = 10, project=None):
    g_loc, y_b = grad_and_value(f, b)
    e_b = (y_b - y_t).item()
    gloc_norm = g_loc.norm().item()

    if gloc_norm > tau_b:
        b_new = b - eta_b * e_b * g_loc
        if project is not None:
            b_new = project(b_new)
        return b_new, dict(mode="grad", e_b=e_b, y_b=y_b.item(),
                            gloc_norm=gloc_norm, kicked=False, kick_dir=None)

    best_name, best_val, best_d = None, None, None
    for name, d in candidates.items():
        dn = d / (d.norm() + 1e-12)
        G = directional_path_sensitivity(f, b, dn, delta_b, M)
        val = e_b * G
        if best_val is None or val < best_val:
            best_val, best_name, best_d = val, name, dn

    if best_val is not None and best_val < 0:
        b_new = b + eta_kick * best_d
        mode, kicked = "kick", True
    else:
        b_new = b.clone()
        mode, kicked = "stuck", False
    if project is not None:
        b_new = project(b_new)
    return b_new, dict(mode=mode, e_b=e_b, y_b=y_b.item(), gloc_norm=gloc_norm,
                        kicked=kicked, kick_dir=best_name if kicked else None,
                        kick_dir_tensor=best_d if kicked else None)


# ---------------------------------------------------------------------------
# Algorithm 1, full coupled loop
# ---------------------------------------------------------------------------

def run_algorithm1(f, x0: torch.Tensor, b0: torch.Tensor, y_t: float, R: int,
                    eta_x: float, eta_b: float, eta_kick: float,
                    tau_x: float, tau_b: float, eps_x: float = 1e-3,
                    eps_b: float = 1e-3, M: int = 20, delta_b: float = 0.1,
                    top_k: int = 5, project_x=None, project_b=None,
                    adapt_baseline: bool = True, use_ig_weights: bool = True):
    """
    adapt_baseline=False freezes b at b0 for the whole run (Sec 8.5's baseline
    search is disabled) so IG/path-gradient in the vehicle update are still
    computed relative to a reference, but that reference never moves -- an
    ablation against the full adaptive-baseline algorithm.
    use_ig_weights=False disables Sec 4.1's coordinate weighting (W_IG=I) while
    keeping the path-gradient escape mechanism -- isolates what the IG-derived
    per-coordinate weighting specifically contributes.
    """
    x, b = x0.clone(), b0.clone()
    history = []
    prev_dir = None
    for r in range(R):
        vx, vinfo = vehicle_update(f, x, b, y_t, eta_x, tau_x, M, project_x, use_ig_weights)
        if not adapt_baseline:
            vb, binfo = b.clone(), dict(mode="frozen", e_b=(f(b).item() - y_t),
                                         y_b=f(b).item(), gloc_norm=0.0, kicked=False,
                                         kick_dir=None)
            history.append(dict(r=r, x=x.clone(), b=b.clone(),
                                 **{f"x_{k}": v for k, v in vinfo.items()},
                                 **{f"b_{k}": v for k, v in binfo.items()}))
            x = vx
            if abs(vinfo["e_x"]) <= eps_x:
                break
            continue
        cands = build_baseline_candidates(x, b, vinfo["ig"], top_k, prev_dir)
        vb, binfo = baseline_update(f, b, y_t, eta_b, tau_b, eta_kick, cands,
                                     delta_b, max(M // 2, 5), project_b)
        if binfo.get("kicked"):
            prev_dir = binfo["kick_dir_tensor"]
        rec = dict(r=r, x=x.clone(), b=b.clone(), **{f"x_{k}": v for k, v in vinfo.items()},
                   **{f"b_{k}": v for k, v in binfo.items()})
        history.append(rec)
        x, b = vx, vb
        if abs(vinfo["e_x"]) <= eps_x and abs(binfo["e_b"]) <= eps_b:
            break
    history.append(dict(r=len(history), x=x.clone(), b=b.clone()))
    return x, b, history


def run_gd(f, x0: torch.Tensor, y_t: float, R: int, eta_x: float,
           eps_x: float = 1e-3, project_x=None):
    x = x0.clone()
    history = []
    for r in range(R):
        x_new, info = gd_update(f, x, y_t, eta_x, project_x)
        history.append(dict(r=r, x=x.clone(), **info))
        x = x_new
        if abs(info["e_x"]) <= eps_x:
            break
    history.append(dict(r=len(history), x=x.clone()))
    return x, history


# ---------------------------------------------------------------------------
# Popular existing adversarial-attack baselines, reframed for target-seeking.
# Same L2 objective L(x) = 1/2 (f(x)-y^t)^2 as Sec 2, gradient e_x * grad f(x);
# these differ only in how that gradient is turned into a step.
# ---------------------------------------------------------------------------

def pgd_sign_update(f, x: torch.Tensor, y_t: float, eta_x: float, project=None):
    """
    Iterative-FGSM / PGD (Kurakin et al. 2016; Madry et al. 2018): step by the
    SIGN of the loss gradient with a fixed magnitude, discarding how small the
    gradient actually is. This is precisely why sign-based attacks are known
    to be comparatively robust to vanishing/saturated gradients -- as long as
    the sign is still informative, a tiny nonzero gradient gives the same
    full-size step as a large one.
    """
    g_loc, y_x = grad_and_value(f, x)
    e_x = (y_x - y_t).item()
    direction = (1.0 if e_x >= 0 else -1.0) * torch.sign(g_loc)
    x_new = x - eta_x * direction
    if project is not None:
        x_new = project(x_new)
    return x_new, dict(e_x=e_x, y_x=y_x.item(), gloc_norm=g_loc.norm().item())


def mifgsm_update(f, x: torch.Tensor, y_t: float, eta_x: float,
                   momentum: torch.Tensor, mu: float = 0.9, project=None):
    """Momentum Iterative FGSM (Dong et al. 2018): accumulate an L1-normalized
    momentum of the signed gradient, then step by its sign. Momentum is meant
    to carry the update through small local plateaus."""
    g_loc, y_x = grad_and_value(f, x)
    e_x = (y_x - y_t).item()
    signed_grad = e_x * g_loc
    momentum = mu * momentum + signed_grad / (signed_grad.abs().sum() + 1e-12)
    x_new = x - eta_x * torch.sign(momentum)
    if project is not None:
        x_new = project(x_new)
    info = dict(e_x=e_x, y_x=y_x.item(), gloc_norm=g_loc.norm().item())
    return x_new, momentum, info


def adam_update(f, x: torch.Tensor, y_t: float, eta_x: float, state: dict,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8,
                 project=None):
    """Adam-optimized target-seeking (as used by e.g. Carlini & Wagner 2017's
    L2 attack): per-coordinate adaptive step size on the same L2 objective,
    with no attribution/coordinate-weighting mechanism."""
    g_loc, y_x = grad_and_value(f, x)
    e_x = (y_x - y_t).item()
    grad = e_x * g_loc
    state["t"] += 1
    state["m"] = beta1 * state["m"] + (1 - beta1) * grad
    state["v"] = beta2 * state["v"] + (1 - beta2) * grad ** 2
    m_hat = state["m"] / (1 - beta1 ** state["t"])
    v_hat = state["v"] / (1 - beta2 ** state["t"])
    x_new = x - eta_x * m_hat / (v_hat.sqrt() + eps)
    if project is not None:
        x_new = project(x_new)
    info = dict(e_x=e_x, y_x=y_x.item(), gloc_norm=g_loc.norm().item())
    return x_new, state, info


def run_pgd_sign(f, x0: torch.Tensor, y_t: float, R: int, eta_x: float,
                  eps_x: float = 1e-3, project_x=None):
    x = x0.clone()
    history = []
    for r in range(R):
        x_new, info = pgd_sign_update(f, x, y_t, eta_x, project_x)
        history.append(dict(r=r, x=x.clone(), **info))
        x = x_new
        if abs(info["e_x"]) <= eps_x:
            break
    history.append(dict(r=len(history), x=x.clone()))
    return x, history


def run_mifgsm(f, x0: torch.Tensor, y_t: float, R: int, eta_x: float,
               mu: float = 0.9, eps_x: float = 1e-3, project_x=None):
    x = x0.clone()
    momentum = torch.zeros_like(x0)
    history = []
    for r in range(R):
        x_new, momentum, info = mifgsm_update(f, x, y_t, eta_x, momentum, mu, project_x)
        history.append(dict(r=r, x=x.clone(), **info))
        x = x_new
        if abs(info["e_x"]) <= eps_x:
            break
    history.append(dict(r=len(history), x=x.clone()))
    return x, history


def run_adam(f, x0: torch.Tensor, y_t: float, R: int, eta_x: float,
             beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8,
             eps_x: float = 1e-3, project_x=None):
    x = x0.clone()
    state = dict(t=0, m=torch.zeros_like(x0), v=torch.zeros_like(x0))
    history = []
    for r in range(R):
        x_new, state, info = adam_update(f, x, y_t, eta_x, state, beta1, beta2, eps, project_x)
        history.append(dict(r=r, x=x.clone(), **info))
        x = x_new
        if abs(info["e_x"]) <= eps_x:
            break
    history.append(dict(r=len(history), x=x.clone()))
    return x, history
