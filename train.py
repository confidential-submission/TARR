"""
TARR training algorithm (Algorithm 1) and the ERM baseline.

Algorithm 1 is implemented as mini-batch SGD: per batch we apply
rr_with_prior independently to each record's sensitive feature and label,
compute the gradient over the perturbed batch, update the model, then update
both adaptive priors using the online-mean rule from lines 9–10.

The step-size for each call to rr_with_prior scales the per-record budget
(ε/t_max) to a per-batch budget (ε · B / t_max), matching the total privacy
expenditure of the per-example version across all t_max steps.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tarr.rr import rr_with_prior, update_prior


# ---------------------------------------------------------------------------
# TARR  (Algorithm 1)
# ---------------------------------------------------------------------------

def tarr_train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    *,
    sensitive_idx: int,
    eps_p: float,
    eps_f: float,
    optimizer: torch.optim.Optimizer,
    device: str,
    n_epochs: int,
    n_sensitive_cats: int = 2,
    n_label_cats: int = 2,
) -> torch.nn.Module:
    """
    Train model with TARR (Algorithm 1).

    Args:
        model            : SimpleMLP or any binary classifier with logit output
        train_loader     : yields (X, y) with y ∈ {0, 1} float
        sensitive_idx    : column index of the binary sensitive attribute in X
        eps_p            : label-DP budget ε_p
        eps_f            : individual-fairness budget ε_f
        optimizer        : already-constructed optimizer (AdamW from run scripts)
        device           : 'cpu' or 'cuda'
        n_epochs         : total training epochs
        n_sensitive_cats : m, number of sensitive-feature categories (2 for binary)
        n_label_cats     : K, number of label categories (2 for binary)

    Returns:
        trained model
    """
    model.to(device)

    n_train = len(train_loader.dataset)
    t_max = n_epochs * n_train  # total examples processed (≈ Algorithm 1's t_max)

    # Uniform priors at initialisation (Algorithm 1, lines 1–2)
    prior_f = torch.full((n_sensitive_cats,), 1.0 / n_sensitive_cats,
                         dtype=torch.float32, device=device)
    prior_l = torch.full((n_label_cats,), 1.0 / n_label_cats,
                         dtype=torch.float32, device=device)

    t = 0  # examples seen (used as denominator in prior update)

    for _ in range(n_epochs):
        model.train()
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)      # (B, 1) float in {0.0, 1.0}
            B = X.shape[0]

            # Algorithm 1, line 5: perturb sensitive feature
            s_idx_col = X[:, sensitive_idx].long()          # (B,) ∈ {0, 1}
            step_eps_f = eps_f * B / t_max
            s_perturbed, mean_oh_f = rr_with_prior(s_idx_col, step_eps_f, prior_f)

            # Algorithm 1, line 6: perturb label
            y_col = y.squeeze(-1).long()                    # (B,) ∈ {0, 1}
            step_eps_l = n_train * eps_p * B / t_max
            y_perturbed, mean_oh_l = rr_with_prior(y_col, step_eps_l, prior_l)

            # Build perturbed batch (line 7): replace sensitive column with x̌_sen
            X_hat = X.clone()
            X_hat[:, sensitive_idx] = s_perturbed.float()
            y_hat = y_perturbed.float().unsqueeze(-1)       # (B, 1)

            # Lines 7–8: gradient step
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(X_hat), y_hat)
            loss.backward()
            optimizer.step()

            # Lines 9–10: update priors (online mean, denominator = examples seen)
            t += B
            prior_f = update_prior(prior_f, mean_oh_f, t)
            prior_l = update_prior(prior_l, mean_oh_l, t)

    return model


# ---------------------------------------------------------------------------
# ERM baseline
# ---------------------------------------------------------------------------

def erm_train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer,
    device: str,
    n_epochs: int,
) -> torch.nn.Module:
    """Standard empirical-risk-minimisation baseline."""
    model.to(device)
    for _ in range(n_epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            F.binary_cross_entropy_with_logits(model(X), y).backward()
            optimizer.step()
    return model
