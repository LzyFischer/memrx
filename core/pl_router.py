"""
MemRx component 2 — mixed-likelihood listwise supervision.

What each query gives us is a *set* of scores, one per candidate view, not a
label. The router is fitted with a Plackett-Luce objective over that set:

    p_theta(k | x) = softmax_k  f_theta(x, e_k)
    L              = KL( p_tilde || p_theta )

Softmax normalisation cancels everything shared by a group -- query
difficulty, backbone strength, judge leniency -- so what gets fitted is the
within-query ordering only, which is all routing needs.

The target p_tilde is where the second component lives. The end-to-end token
metric is censored, not merely noisy:

    s_k ~ 1[ view k's evidence is good enough for the reader to answer ]

so on most queries every view ties and the group carries no signal at all.
The gold-answer likelihood does not saturate the same way -- it measures how
confident the reader is about the right answer rather than whether it cleared
the bar -- so it can supply an ordering precisely where F1 has none. The two
are not comparable in value (one is in [0,1], the other is a negative number
whose scale moves with the query), so they are mixed at the level of the
target *distribution*, not the raw score:

    p_F1(k)  = softmax_k( s_k / tau )
    p_LL(k)  = softmax_k( l_k / tau_ll )
    alpha_q  = clip( (max_k s_k - min_k s_k) / delta, 0, 1 )
    p_tilde  = alpha_q * p_F1 + (1 - alpha_q) * p_LL

alpha is per group, not a global hyper-parameter. Where F1 separates the views
it is trusted on its own; where F1 ties, alpha goes to 0 and the group is
supervised by likelihood instead of being thrown away. Two limits worth
naming: alpha == 1 everywhere is F1-only supervision, and tau -> 0 with
alpha == 1 is an ordinary argmax classifier, so the classification baseline is
a point inside this family rather than a separate method.

Scoring function, with e_k the encoded description of view k (not a per-view
output head, so a view unseen during training can still be scored):

    h1 = relu(Wx x + bx)      h2 = relu(We e + be)
    z  = [h1 ; h2 ; h1 * h2]
    f  = w2 . relu(W1 z + b1) + b2

Plain numpy with hand-written gradients — a few thousand parameters over a
few thousand rows, so there is nothing torch would buy here.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


class RandomProjector:
    """Fixed seeded Gaussian projection, used to shrink sentence embeddings
    before they reach the router.

    With a few hundred training questions, a 384-d embedding fed in at full
    width dominates the 7 probe statistics by dimension count alone. The
    projection is seeded and fitted to nothing, so it introduces no
    train/test leakage and is identical across splits.
    """

    def __init__(self, d_in: int, d_out: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0.0, 1.0 / np.sqrt(d_out), size=(d_in, d_out)).astype(np.float32)
        self.d_in, self.d_out = d_in, d_out

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float32) @ self.W


def _relu(x):
    return np.maximum(x, 0.0)


class _Adam:
    def __init__(self, params, lr=3e-4, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
        self.p, self.lr, self.b1, self.b2, self.eps, self.wd = params, lr, b1, b2, eps, wd
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads):
        self.t += 1
        for k, g in grads.items():
            if self.wd and k.startswith("W"):
                g = g + self.wd * self.p[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            self.p[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


def _segment_softmax(s, G, n_groups):
    dt = s.dtype if np.issubdtype(s.dtype, np.floating) else np.float64
    mx = np.full(n_groups, -np.inf, dt)
    np.maximum.at(mx, G, s)
    e = np.exp(s - mx[G])
    den = np.zeros(n_groups, dt)
    np.add.at(den, G, e)
    return e / (den[G] + 1e-12)


def mixed_target(s_f1, s_ll, G, n_groups, tau=0.1, tau_ll=0.1, delta=0.1):
    """p_tilde for each row. See the module docstring.

    `s_ll` may contain NaN for cells whose likelihood was not collected; those
    groups fall back to F1 alone (alpha forced to 1) rather than being dropped.
    """
    s_f1 = np.asarray(s_f1, np.float64)
    s_ll = np.asarray(s_ll, np.float64)

    if tau <= 0:  # degenerate case: one-hot on the group argmax
        best = np.full(n_groups, -np.inf)
        np.maximum.at(best, G, s_f1)
        hit = (s_f1 >= best[G] - 1e-12).astype(np.float64)
        cnt = np.zeros(n_groups)
        np.add.at(cnt, G, hit)
        return hit / (cnt[G] + 1e-12)  # ties shared, not broken by float noise

    p_f1 = _segment_softmax(s_f1 / tau, G, n_groups)

    hi = np.full(n_groups, -np.inf)
    lo = np.full(n_groups, np.inf)
    np.maximum.at(hi, G, s_f1)
    np.minimum.at(lo, G, s_f1)
    alpha = np.clip((hi - lo) / max(delta, 1e-9), 0.0, 1.0)

    ok = np.isfinite(s_ll)
    n_ok = np.zeros(n_groups)
    n_tot = np.zeros(n_groups)
    np.add.at(n_ok, G, ok.astype(np.float64))
    np.add.at(n_tot, G, 1.0)
    alpha = np.where(n_ok >= n_tot - 1e-9, alpha, 1.0)  # incomplete LL -> F1 only

    filled = np.where(ok, s_ll, 0.0)
    p_ll = _segment_softmax(filled / tau_ll, G, n_groups)

    a = alpha[G]
    return a * p_f1 + (1.0 - a) * p_ll


class PLRouter:
    """Listwise router over the candidate menu.

    Training data is a list of groups:
        {"x": (d_x,) array, "cands": [int,...],
         "s_f1": [float,...], "s_ll": [float,...]}   (s_ll may be NaN)
    `view_embs` is (K, d_e): the encoded description of each view.
    """

    def __init__(self, view_embs, hidden=64, tau=0.1, tau_ll=0.1, delta=0.1,
                 lr=3e-3, weight_decay=1e-4, epochs=300, seed=0, verbose=False):
        self.view_embs = np.asarray(view_embs, dtype=np.float32)
        self.K, self.d_e = self.view_embs.shape
        self.hidden, self.tau, self.tau_ll, self.delta = hidden, tau, tau_ll, delta
        self.lr, self.wd, self.epochs, self.seed = lr, weight_decay, epochs, seed
        self.verbose = verbose
        self.params: Optional[Dict[str, np.ndarray]] = None
        self._mu = self._sd = None

    def _init_params(self, d_x):
        rng = np.random.default_rng(self.seed)
        H = self.hidden

        def g(a, b):
            return rng.normal(0, np.sqrt(2.0 / a), size=(a, b)).astype(np.float32)

        self.params = {
            "Wx": g(d_x, H), "bx": np.zeros(H, np.float32),
            "We": g(self.d_e, H), "be": np.zeros(H, np.float32),
            "W1": g(3 * H, H), "b1": np.zeros(H, np.float32),
            "W2": g(H, 1), "b2": np.zeros(1, np.float32),
        }

    def _flatten(self, groups):
        X, E, G, F, L = [], [], [], [], []
        for gi, grp in enumerate(groups):
            for pos, k in enumerate(grp["cands"]):
                X.append(grp["x"])
                E.append(self.view_embs[k])
                G.append(gi)
                F.append(grp["s_f1"][pos])
                L.append(grp["s_ll"][pos] if "s_ll" in grp else np.nan)
        return (np.asarray(X, np.float32), np.asarray(E, np.float32),
                np.asarray(G, np.int64), np.asarray(F, np.float64),
                np.asarray(L, np.float64))

    def _standardise(self, X, fit=False):
        if fit:
            self._mu = X.mean(0)
            self._sd = X.std(0) + 1e-6
        return (X - self._mu) / self._sd

    def _forward(self, X, E):
        p = self.params
        a1 = X @ p["Wx"] + p["bx"]; h1 = _relu(a1)
        a2 = E @ p["We"] + p["be"]; h2 = _relu(a2)
        z = np.concatenate([h1, h2, h1 * h2], axis=1)
        a3 = z @ p["W1"] + p["b1"]; h3 = _relu(a3)
        s = (h3 @ p["W2"] + p["b2"]).ravel()
        return s, (X, E, a1, h1, a2, h2, z, a3, h3)

    def _backward(self, ds, cache):
        X, E, a1, h1, a2, h2, z, a3, h3 = cache
        p, H = self.params, self.hidden
        ds = ds.reshape(-1, 1)
        gW2, gb2 = h3.T @ ds, ds.sum(0)
        da3 = (ds @ p["W2"].T) * (a3 > 0)
        gW1, gb1 = z.T @ da3, da3.sum(0)
        dz = da3 @ p["W1"].T
        dz1, dz2, dz3 = dz[:, :H], dz[:, H:2 * H], dz[:, 2 * H:]
        da1 = (dz1 + dz3 * h2) * (a1 > 0)
        da2 = (dz2 + dz3 * h1) * (a2 > 0)
        return {"Wx": X.T @ da1, "bx": da1.sum(0), "We": E.T @ da2, "be": da2.sum(0),
                "W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}

    def fit(self, groups: Sequence[dict]):
        groups = [g for g in groups if len(g["cands"]) >= 2]
        if not groups:
            raise ValueError("no usable training groups (need >= 2 candidates each)")

        X, E, G, F, L = self._flatten(groups)
        X = self._standardise(X, fit=True)
        n = len(groups)
        self._init_params(X.shape[1])
        opt = _Adam(self.params, lr=self.lr, wd=self.wd)
        ptilde = mixed_target(F, L, G, n, tau=self.tau, tau_ll=self.tau_ll, delta=self.delta)

        for ep in range(self.epochs):
            s, cache = self._forward(X, E)
            p = _segment_softmax(s, G, n)
            opt.step(self._backward((p - ptilde) / n, cache))
            if self.verbose and (ep + 1) % 50 == 0:
                ce = -float((ptilde * np.log(p + 1e-12)).sum() / n)
                print(f"    epoch {ep+1:4d}  ce={ce:.4f}")
        return self

    def score(self, x, cands):
        X = self._standardise(np.tile(np.asarray(x, np.float32), (len(cands), 1)))
        return self._forward(X, self.view_embs[list(cands)])[0]

    def predict(self, x, cands) -> int:
        return int(list(cands)[int(np.argmax(self.score(x, cands)))])
