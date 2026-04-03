"""Mixture Density Network for conditional density estimation.

A small MLP that outputs the parameters of a Gaussian mixture model:
  p(y | x) = sum_k  pi_k(x) * N(y; mu_k(x), sigma_k(x)^2)

where x is the context vector and y is log SFR surface density.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


class MDN(nn.Module):
    """Mixture Density Network: MLP backbone + Gaussian mixture head."""

    def __init__(self, n_inputs: int, n_hidden: int, n_layers: int, n_components: int):
        super().__init__()
        layers = [nn.Linear(n_inputs, n_hidden), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_hidden, n_hidden), nn.ReLU()]
        self.backbone = nn.Sequential(*layers)

        self.head_pi = nn.Linear(n_hidden, n_components)
        self.head_mu = nn.Linear(n_hidden, n_components)
        self.head_log_sigma = nn.Linear(n_hidden, n_components)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return mixture weights (log-softmax), means, and log-sigmas."""
        h = self.backbone(x)
        log_pi = torch.log_softmax(self.head_pi(h), dim=-1)
        mu = self.head_mu(h)
        log_sigma = self.head_log_sigma(h)
        return log_pi, mu, log_sigma


def mdn_loss(log_pi: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood of a Gaussian mixture."""
    y = y.unsqueeze(-1)
    log_normal = -0.5 * np.log(2 * np.pi) - log_sigma - 0.5 * ((y - mu) / torch.exp(log_sigma)) ** 2
    log_prob = torch.logsumexp(log_pi + log_normal, dim=-1)
    return -log_prob.mean()


def predict_mean(model: MDN, x: torch.Tensor) -> np.ndarray:
    """Mixture mean: sum_k pi_k * mu_k."""
    model.eval()
    with torch.no_grad():
        log_pi, mu, _ = model(x)
        pi = torch.exp(log_pi)
        y_pred = (pi * mu).sum(dim=-1)
    return y_pred.cpu().numpy()


def sample(model: MDN, x: torch.Tensor) -> np.ndarray:
    """Draw one sample per input from the predicted mixture distribution."""
    model.eval()
    with torch.no_grad():
        log_pi, mu, log_sigma = model(x)
        # Pick a component for each input
        k = torch.multinomial(torch.exp(log_pi), 1).squeeze(-1)
        chosen_mu = mu[torch.arange(len(k)), k]
        chosen_sigma = torch.exp(log_sigma[torch.arange(len(k)), k])
        y = chosen_mu + chosen_sigma * torch.randn_like(chosen_sigma)
    return y.cpu().numpy()


def predict_nll(model: MDN, x: torch.Tensor, y: torch.Tensor) -> float:
    """Mean NLL on a dataset."""
    model.eval()
    with torch.no_grad():
        log_pi, mu, log_sigma = model(x)
        nll = mdn_loss(log_pi, mu, log_sigma, y)
    return nll.item()


def train_mdn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_hidden: int = 128,
    n_layers: int = 3,
    n_components: int = 8,
    lr: float = 1e-3,
    n_epochs: int = 200,
    batch_size: int = 2048,
    device: str = "cpu",
) -> tuple[MDN, list[float]]:
    """Train an MDN on numpy arrays. Returns the model and loss history."""
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MDN(X_train.shape[1], n_hidden, n_layers, n_components).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    losses = []
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            log_pi, mu, log_sigma = model(xb)
            loss = mdn_loss(log_pi, mu, log_sigma, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1:4d}/{n_epochs}  NLL = {avg_loss:.4f}")

    return model, losses
