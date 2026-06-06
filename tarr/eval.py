"""
Fairness evaluation metrics used in the paper (Section 4.1 and Table 1).

CNS (Consistency Score, Equation (10)):
    Percentage of test inputs whose predicted class is identical when the
    sensitive feature is flipped to its other binary value {0, 1}.
    Higher CNS means more individually-fair predictions.
"""

import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def consistency_score(
    model: torch.nn.Module,
    loader: DataLoader,
    sensitive_idx: int,
    device: str = 'cpu',
) -> float:
    """
    CNS: percentage of examples whose binary prediction is unchanged
    after flipping the sensitive feature at column sensitive_idx.

    Assumes the sensitive feature is binary (values 0 and 1).

    Args:
        model         : trained classifier (logit output for binary cross-entropy)
        loader        : data loader yielding (X, y) batches
        sensitive_idx : column index of the binary sensitive feature in X
        device        : torch device string

    Returns:
        CNS as a float in [0, 100]
    """
    model.eval()
    model.to(device)
    consistent = total = 0

    for X, _ in loader:
        X = X.to(device)
        pred = model(X) > 0                     # (B, 1) bool

        X_flip = X.clone()
        X_flip[:, sensitive_idx] = 1.0 - X_flip[:, sensitive_idx]
        pred_flip = model(X_flip) > 0           # (B, 1) bool

        consistent += (pred == pred_flip).sum().item()
        total += pred.numel()

    return 100.0 * consistent / total


@torch.no_grad()
def accuracy(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str = 'cpu',
) -> float:
    """Test accuracy (%) for binary classification with logit output."""
    model.eval()
    model.to(device)
    correct = total = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = (model(X) > 0).float()
        correct += (pred == y).sum().item()
        total += y.numel()

    return 100.0 * correct / total
