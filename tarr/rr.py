import torch


def rr_with_prior(
    column_idx: torch.Tensor,
    epsilon: float,
    prior: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Randomised response with adaptive prior (TARR Algorithm 1, lines 5–6).

    Sampling rule for each record i with true category j:
        P(output = k | j) ∝ prior[k] · exp(ε · 1[k == j])

    With uniform prior this reduces to standard local-DP randomised response;
    as prior concentrates on frequently sampled values the mechanism shifts
    probability mass toward those categories.

    Args:
        column_idx : (N,) int64 tensor of category indices in [0, m)
        epsilon    : per-step privacy budget
        prior      : (m,) float32 probability vector over m categories

    Returns:
        perturbed   : (N,) int64 tensor of sampled indices
        mean_onehot : (m,) float32 mean one-hot of sampled indices,
                      used by the caller to update the prior via update_prior()
    """
    n = column_idx.shape[0]
    m = prior.shape[0]
    device = column_idx.device

    # log P(output = k | input = j) = log prior[k] + ε · 1[k == j]   (unnormalised)
    log_prior = prior.log().clamp(min=-1e9)                        # (m,)
    log_probs = log_prior.unsqueeze(0).expand(n, m).clone()        # (N, m)
    log_probs[torch.arange(n, device=device), column_idx] += epsilon

    probs = torch.softmax(log_probs, dim=-1)                       # (N, m)
    perturbed = torch.multinomial(probs, 1).squeeze(-1)            # (N,)

    mean_onehot = torch.zeros(m, dtype=prior.dtype, device=device)
    mean_onehot.scatter_add_(
        0, perturbed, torch.ones(n, dtype=prior.dtype, device=device)
    )
    mean_onehot.div_(n)
    return perturbed, mean_onehot


def update_prior(prior: torch.Tensor, mean_onehot: torch.Tensor, t: int) -> torch.Tensor:
    """
    Online-mean prior update (Algorithm 1, lines 9–10):
        prior_t = prior_{t-1} − (prior_{t-1} − mean_onehot) / t

    This is an equal-weight running average; prior_t converges to the
    empirical distribution of sampled values seen so far.

    Args:
        prior       : (m,) current prior
        mean_onehot : (m,) batch-mean one-hot returned by rr_with_prior
        t           : number of examples processed so far (1-based)
    """
    return prior - (prior - mean_onehot) / t
