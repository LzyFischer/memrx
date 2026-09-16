"""
Listwise router: for each question, score every view and pick the argmax.

Everything is a dense (N, K) array: N questions, K views. Missing F1 is NaN
and that view is masked out.

    score(x, e_k) = MLP([h_x ; h_e ; h_x * h_e]),   h_x = relu(Wx x),  h_e = relu(We e_k)
    target_k      = softmax_k(F1'_k / tau)           (tau = 0: argmax, ties shared)
    F1'_k         = best F1 if F1_k >= best - margin, else F1_k
    loss          = cross_entropy(target, softmax(score))

e_k is the embedding of view k's text description, so views share one scoring
network instead of each having its own output head.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_target(f1: torch.Tensor, tau: float, margin: float = 0.0) -> torch.Tensor:
    """(N, K) F1 with NaN for missing -> (N, K) target distribution.

    margin: a view whose F1 is within `margin` of the question's best is
    treated as equally good (its F1 is raised to the best before the target is
    built), so 0.51 vs 0.50 does not become a label.
      tau = 0  every view within margin of the best shares the mass equally
      tau > 0  softmax(F1' / tau) on the margin-adjusted F1'
    """
    mask = torch.isfinite(f1)
    best = f1.masked_fill(~mask, -float("inf")).max(dim=1, keepdim=True).values
    near_best = mask & (f1 >= best - margin - 1e-9)
    if tau <= 0:
        hit = near_best.float()
        return hit / hit.sum(dim=1, keepdim=True)
    f1 = torch.where(near_best, best.expand_as(f1), f1)
    logits = (f1.nan_to_num(0.0) / tau).masked_fill(~mask, -float("inf"))
    return logits.softmax(dim=1)


def entropy(p: torch.Tensor) -> float:
    """Mean entropy of the rows of p: the lowest loss the router can reach."""
    return -(p * p.clamp_min(1e-12).log()).sum(dim=1).mean().item()


class Scorer(nn.Module):
    def __init__(self, d_x: int, view_embs: torch.Tensor, hidden: int):
        super().__init__()
        self.register_buffer("E", view_embs)             # (K, d_e), fixed
        self.fx = nn.Linear(d_x, hidden)
        self.fe = nn.Linear(view_embs.shape[1], hidden)
        self.head = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (N, d_x) -> scores (N, K)."""
        hx = F.relu(self.fx(x)).unsqueeze(1)               # (N, 1, H)
        he = F.relu(self.fe(self.E)).unsqueeze(0)          # (1, K, H)
        hx, he = torch.broadcast_tensors(hx, he)           # (N, K, H)
        return self.head(torch.cat([hx, he, hx * he], dim=-1)).squeeze(-1)


class Router:
    def __init__(self, view_embs, hidden=64, tau=0.1, margin=0.0, lr=3e-3, weight_decay=1e-4,
                 epochs=300, seed=0, log_every=0, device="cpu"):
        self.E = torch.as_tensor(np.asarray(view_embs), dtype=torch.float32)
        self.hidden, self.tau, self.margin = hidden, tau, margin
        self.lr, self.wd, self.epochs, self.seed = lr, weight_decay, epochs, seed
        self.log_every, self.device = log_every, device
        self.model = None
        self.mu = self.sd = None
        self.history = []                                   # training loss per epoch
        self.floor = None                                   # target entropy = lowest reachable loss

    def _x(self, X) -> torch.Tensor:
        X = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=self.device)
        return (X - self.mu) / self.sd

    def fit(self, X, f1):
        """X (N, d_x); f1 (N, K) with NaN where a view was not scored."""
        torch.manual_seed(self.seed)
        X = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=self.device)
        f1 = torch.as_tensor(np.asarray(f1), dtype=torch.float32, device=self.device)
        mask = torch.isfinite(f1)
        keep = mask.sum(1) >= 2
        X, f1, mask = X[keep], f1[keep], mask[keep]

        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        X = (X - self.mu) / self.sd
        target = make_target(f1, self.tau, self.margin)
        self.floor = entropy(target)
        if self.log_every:
            print(f"    loss floor (target entropy) {self.floor:.4f}")

        self.model = Scorer(X.shape[1], self.E, self.hidden).to(self.device)
        decay = [p for n, p in self.model.named_parameters() if n.endswith("weight")]
        no_decay = [p for n, p in self.model.named_parameters() if n.endswith("bias")]
        opt = torch.optim.Adam([{"params": decay, "weight_decay": self.wd},
                                {"params": no_decay, "weight_decay": 0.0}], lr=self.lr)

        self.model.train()
        for t in range(1, self.epochs + 1):
            logp = self.model(X).masked_fill(~mask, -float("inf")).log_softmax(dim=1)
            loss = -(target * logp.masked_fill(~mask, 0.0)).sum(dim=1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            self.history.append(loss.item())
            if self.log_every and t % self.log_every == 0:
                print(f"    epoch {t:4d}  loss {loss.item():.4f}  (above floor {loss.item() - self.floor:.4f})")
        return self

    @torch.no_grad()
    def scores(self, X) -> np.ndarray:
        self.model.eval()
        return self.model(self._x(X)).cpu().numpy()

    def predict(self, X, mask=None) -> np.ndarray:
        s = self.scores(X)
        if mask is not None:
            s = np.where(mask, s, -np.inf)
        return s.argmax(axis=1)
