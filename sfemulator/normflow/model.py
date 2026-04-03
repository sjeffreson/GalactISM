"""Conditional normalizing flow for 1D density estimation.

A conditional neural spline flow that learns p(y | x) where y is log SFR
surface density and x is the context vector. Uses rational-quadratic spline
coupling transforms conditioned on x via an MLP.

Since y is 1D, each flow layer applies a monotone spline transform to y
whose knot parameters are predicted by a small network from x.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import softplus


# ── Rational-quadratic spline (Durkan et al. 2019) ───────────────────

def rational_quadratic_spline(
    y: torch.Tensor,
    widths: torch.Tensor,
    heights: torch.Tensor,
    derivatives: torch.Tensor,
    inverse: bool = False,
    tail_bound: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a monotone rational-quadratic spline, return (output, log_det)."""
    inside = (y > -tail_bound) & (y < tail_bound)
    out = y.clone()
    log_det = torch.zeros_like(y)

    if not inside.any():
        return out, log_det

    y_in = y[inside]

    # Bin widths/heights -> cumulative knot positions
    widths = softplus(widths[inside]) + 1e-3
    widths = widths / widths.sum(dim=-1, keepdim=True) * 2 * tail_bound
    heights = softplus(heights[inside]) + 1e-3
    heights = heights / heights.sum(dim=-1, keepdim=True) * 2 * tail_bound
    derivatives = softplus(derivatives[inside]) + 1e-3

    cum_w = torch.cumsum(widths, dim=-1) - tail_bound
    cum_h = torch.cumsum(heights, dim=-1) - tail_bound
    cum_w = torch.cat([torch.full_like(cum_w[..., :1], -tail_bound), cum_w], dim=-1)
    cum_h = torch.cat([torch.full_like(cum_h[..., :1], -tail_bound), cum_h], dim=-1)

    K = widths.shape[-1]

    if inverse:
        # Find which bin each y_in falls into (in output space)
        bin_idx = torch.searchsorted(cum_h[..., 1:].contiguous(), y_in.unsqueeze(-1)).squeeze(-1).clamp(0, K - 1)
    else:
        bin_idx = torch.searchsorted(cum_w[..., 1:].contiguous(), y_in.unsqueeze(-1)).squeeze(-1).clamp(0, K - 1)

    # Gather knot parameters for the active bin
    xk = cum_w.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    xk1 = cum_w.gather(-1, (bin_idx + 1).unsqueeze(-1)).squeeze(-1)
    yk = cum_h.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    yk1 = cum_h.gather(-1, (bin_idx + 1).unsqueeze(-1)).squeeze(-1)
    dk = derivatives.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    dk1 = derivatives.gather(-1, (bin_idx + 1).unsqueeze(-1)).squeeze(-1)

    w = xk1 - xk
    h = yk1 - yk
    s = h / w

    if inverse:
        # Invert the spline: given output, find input
        a = h * (s - dk) + (y_in - yk) * (dk1 + dk - 2 * s)
        b = h * dk - (y_in - yk) * (dk1 + dk - 2 * s)
        c = -s * (y_in - yk)
        disc = b ** 2 - 4 * a * c
        xi = (2 * c) / (-b - torch.sqrt(disc.clamp(min=1e-8)))
        out[inside] = xi * w + xk
        # log_det for inverse is negative of forward
        num = s ** 2 * (dk1 * xi ** 2 + 2 * s * xi * (1 - xi) + dk * (1 - xi) ** 2)
        den = (s + (dk1 + dk - 2 * s) * xi * (1 - xi)) ** 2
        log_det[inside] = -torch.log(num / den + 1e-8)
    else:
        xi = (y_in - xk) / w
        num = h * (s * xi ** 2 + dk * xi * (1 - xi))
        den = s + (dk1 + dk - 2 * s) * xi * (1 - xi)
        out[inside] = yk + num / den
        # log |dy/dx|
        num_d = s ** 2 * (dk1 * xi ** 2 + 2 * s * xi * (1 - xi) + dk * (1 - xi) ** 2)
        den_d = (s + (dk1 + dk - 2 * s) * xi * (1 - xi)) ** 2
        log_det[inside] = torch.log(num_d / den_d + 1e-8)

    return out, log_det


# ── Conditional spline layer ─────────────────────────────────────────

class ConditionalSplineLayer(nn.Module):
    """One spline transform whose knot parameters are conditioned on x."""

    def __init__(self, n_context: int, n_hidden: int, n_bins: int, tail_bound: float = 5.0):
        super().__init__()
        self.n_bins = n_bins
        self.tail_bound = tail_bound
        self.net = nn.Sequential(
            nn.Linear(n_context, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, 3 * n_bins + 1),
        )

    def forward(self, y: torch.Tensor, x: torch.Tensor, inverse: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform y conditioned on x. Returns (y_out, log_det)."""
        params = self.net(x)
        widths = params[..., :self.n_bins]
        heights = params[..., self.n_bins:2 * self.n_bins]
        derivatives = params[..., 2 * self.n_bins:]
        return rational_quadratic_spline(y, widths, heights, derivatives, inverse=inverse, tail_bound=self.tail_bound)


# ── Full conditional flow ────────────────────────────────────────────

class ConditionalFlow(nn.Module):
    """Stack of conditional spline layers for 1D density estimation."""

    def __init__(self, n_context: int, n_layers: int = 8, n_hidden: int = 64, n_bins: int = 16, tail_bound: float = 5.0):
        super().__init__()
        self.layers = nn.ModuleList([
            ConditionalSplineLayer(n_context, n_hidden, n_bins, tail_bound)
            for _ in range(n_layers)
        ])

    def forward(self, y: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map y -> z (base space), return (z, total_log_det)."""
        total_log_det = torch.zeros_like(y)
        for layer in self.layers:
            y, ld = layer(y, x)
            total_log_det += ld
        return y, total_log_det

    def inverse(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Map z (base space) -> y (data space)."""
        for layer in reversed(self.layers):
            z, _ = layer(z, x, inverse=True)
        return z

    def log_prob(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Log probability of y given x."""
        z, log_det = self.forward(y, x)
        log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi))
        return log_pz + log_det


def flow_nll(model: ConditionalFlow, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Mean negative log-likelihood."""
    return -model.log_prob(y, x).mean()


def predict_mean(model: ConditionalFlow, x: torch.Tensor, n_samples: int = 32, batch_size: int = 8192) -> np.ndarray:
    """Estimate the conditional mean by averaging samples, in batches to avoid OOM."""
    model.eval()
    means = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = x[start:start + batch_size]
            z = torch.randn(xb.shape[0], n_samples, device=x.device)
            x_rep = xb.unsqueeze(1).expand(-1, n_samples, -1).reshape(-1, xb.shape[1])
            y_samples = model.inverse(z.reshape(-1), x_rep)
            y_samples = y_samples.reshape(xb.shape[0], n_samples)
            means.append(y_samples.mean(dim=1))
    return torch.cat(means).cpu().numpy()


def predict_nll(model: ConditionalFlow, x: torch.Tensor, y: torch.Tensor) -> float:
    """Mean NLL on a dataset."""
    model.eval()
    with torch.no_grad():
        nll = flow_nll(model, y, x)
    return nll.item()


def sample(model: ConditionalFlow, x: torch.Tensor) -> np.ndarray:
    """Draw one sample per input from the flow."""
    model.eval()
    with torch.no_grad():
        z = torch.randn(x.shape[0], device=x.device)
        y = model.inverse(z, x)
    return y.cpu().numpy()


def train_flow(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_layers: int = 8,
    n_hidden: int = 64,
    n_bins: int = 16,
    tail_bound: float = 5.0,
    lr: float = 1e-3,
    n_epochs: int = 200,
    batch_size: int = 2048,
    device: str = "cpu",
) -> tuple[ConditionalFlow, list[float]]:
    """Train a conditional flow on numpy arrays."""
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ConditionalFlow(
        n_context=X_train.shape[1],
        n_layers=n_layers,
        n_hidden=n_hidden,
        n_bins=n_bins,
        tail_bound=tail_bound,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    losses = []
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = flow_nll(model, yb, xb)
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
