from __future__ import print_function
import torch
import torch.nn as nn

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class BarlowTwins(nn.Module):
    def __init__(self, lambd, model_dim) -> None:
        super().__init__()
        self.lambd = lambd
        self.model_dim = model_dim

        sizes = [self.model_dim, 4096]
        layers = []
        for i in range(len(sizes)-2):
            layers.append(nn.Linear(sizes[i], sizes[i+1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i+1]))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, features):
        z1 = self.projector(features[:, 0, :])
        z2 = self.projector(features[: , 1, :])

        c = self.bn(z1).T @ self.bn(z2)

        on_diag = torch.diagonal(c).add_(-1).pow(2).sum() * (1.0/256)
        off_diag = off_diagonal(c).pow_(2).sum() * (1.0 /256)

        loss = on_diag + self.lambd * off_diag
        return loss
        
