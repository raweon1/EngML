import torch
import torch.nn as nn
from torch.distributions import Normal


class SDN(nn.Module):
    def __init__(self, n_input, n_hidden):
        super(SDN, self).__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(n_input, affine=False),
            nn.Linear(n_input, n_hidden),
            nn.ELU(),
            nn.Linear(n_hidden, 2)
        )

    def forward(self, x):
        x = self.net(x)
        mu = x[:, 0]
        log_sigma = x[:, 1]
        sigma = torch.exp(torch.clip(log_sigma, -20, 20))
        dist = Normal(mu, sigma)
        return dist


class MDN(nn.Module):
    def __init__(self, n_input, n_hidden, n_components=3):
        super(MDN, self).__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(n_input, affine=False),
            nn.Linear(n_input, n_hidden),
            nn.ELU(),
            nn.Linear(n_hidden, 3 * n_components)
        )
        self.n_components = n_components

    def forward(self, x):
        x = self.net(x).reshape(-1, self.n_components, 3)
        pi = x[:, :, 0]
        mu = x[:, :, 1]
        log_sigma = x[:, :, 2]

        pi = torch.softmax(pi, -1)
        sigma = torch.exp(torch.clip(log_sigma, -20, 20))
        dist = Normal(mu, sigma)
        return pi, dist


def distribution_log_prob_loss(dist, target, pi=None):
    if pi is not None:
        loss = -torch.logsumexp(dist.log_prob(target) + torch.log(pi), dim=1)
    else:
        loss = -dist.log_prob(target)
    return loss.mean()
