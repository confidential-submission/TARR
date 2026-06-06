import torch.nn as nn


class SimpleMLP(nn.Module):
    """
    Six-layer fully connected network (paper Section 4.1).
    Architecture: Linear → ReLU repeated for each hidden width,
    then a non-activated output layer.

    Default hidden widths [64, 32, 16, 8, 4] give six linear layers total
    (five with ReLU, one output without), matching Zhang et al. [40] and
    Zheng et al. [42].
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int = 1,
        hidden: tuple = (64, 32, 16, 8, 4),
    ):
        super().__init__()
        dims = [in_features] + list(hidden)
        layers = []
        for i in range(len(dims) - 1):
            layers.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
        layers.append(nn.Linear(dims[-1], num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
