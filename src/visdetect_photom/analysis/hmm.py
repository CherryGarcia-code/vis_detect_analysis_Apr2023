"""Bernoulli GLM-HMM for behavioral state identification.

Ported from vis_detect_analysis_Sep2025 (Neuropixels ephys project).

Implements the Generalized Linear Model Hidden Markov Model framework
described in:

    Ashwood, Roy, Stone et al. (2022). "Mice alternate between discrete
    strategies during perceptual decision-making."
    Nature Neuroscience 25, 201-212.

Each hidden state k defines a distinct behavioral strategy, parameterized
by a logistic regression mapping trial covariates to P(lick):

    P(lick_t = 1 | z_t = k, x_t) = sigmoid(w_k^T x_t)

Default covariates x_t = [1, log2(change_size), prev_choice, prev_reward,
prev_early_lick].

The model is fit via Expectation-Maximization (EM):
    E-step  : Forward-backward algorithm for state posteriors
    M-step  : Transition matrix MLE + weighted logistic regression per state

Multi-session fitting shares all parameters across sessions while allowing
independent state sequences per session (forward-backward resets at session
boundaries).

Usage
-----
    from visdetect_photom.analysis.hmm import GLMHMM, prepare_session_data, fit_best_model

    sessions_data = [prepare_session_data(s) for s in sessions]
    best_model, selection_df = fit_best_model(sessions_data, K_range=[2, 3, 4, 5])
    states = best_model.most_likely_states(sessions_data[0])
"""

from __future__ import annotations

import json
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from tqdm import tqdm

from visdetect_photom.core.constants import (
    CATCH_THRESHOLD, FA_RT_SPLIT, OUTCOME_NORMALIZATION,
)

# =====================================================================
# Constants
# =====================================================================
_EPS = 1e-300  # prevent log(0)

FEATURE_NAMES = ["bias", "stimulus", "prev_choice", "prev_reward", "prev_early_lick"]

# State labeling config (matches ephys repo convention)
# auto_label_states produces: Disengaged, Engaged, Biased (+ _1/_2 suffixes)
# rename_labels maps these to canonical names for display
HMM_LABEL_RENAME = {
    "Engaged_1":      "Disengaged",
    "Engaged_2":      "Engaged",
    "Engaged_3":      "Impulsive",
    "Biased":         "Impulsive",
    "Biased_1":       "Impulsive",
    "Biased_2":       "Impulsive_2",
    "Disengaged_1":   "Disengaged",
    "Disengaged_2":   "Disengaged_2",
}
HMM_STATE_ORDER = ["Disengaged", "Engaged", "Impulsive"]
HMM_STATE_COLORS = {
    "Disengaged":   "#bdbdbd",
    "Disengaged_2": "#969696",
    "Engaged":      "#6baed6",
    "Impulsive":    "#fb6a4a",
    "Impulsive_2":  "#e6550d",
}


# =====================================================================
# Numerics helpers
# =====================================================================

def _log_bernoulli(y: np.ndarray, logits: np.ndarray) -> np.ndarray:
    """Log probability of Bernoulli observations given logits."""
    return y * logits - np.logaddexp(0, logits)


def _nll_and_grad(w, X, y, gamma_k, l2):
    """Negative weighted log-likelihood and gradient for one state's GLM."""
    logits = X @ w
    p = expit(logits)
    nll = -np.sum(gamma_k * (y * logits - np.logaddexp(0, logits)))
    grad = -X.T @ (gamma_k * (y - p))
    if l2 > 0:
        nll += 0.5 * l2 * np.dot(w, w)
        grad += l2 * w
    return nll, grad


# =====================================================================
# Data preparation  (adapted for photometry Session/Trial dataclasses)
# =====================================================================

def prepare_session_data(
    session,
    *,
    exclude_outcomes: Sequence[str] = ("Abort", "CR"),
) -> Dict[str, Any]:
    """Extract binary choice vector *y* and covariate matrix *X* from a Session.

    Adapted from the ephys project to work with the photometry
    Session/Trial dataclasses. The task structure is identical so the
    same 5 covariates apply.

    Parameters
    ----------
    session : visdetect_photom.core.session.Session
    exclude_outcomes : outcomes to discard (default: Abort, CR).

    Returns
    -------
    dict with keys: y, X, df, session_name, feature_names, subject_id
    """
    # Build a trial DataFrame from the Session's Trial list
    rows = []
    for t in session.trials:
        outcome = t.outcome  # already normalized (Hit, Miss, FA, CR, Abort)
        is_hit = (outcome == 'Hit')
        is_miss = (outcome == 'Miss')
        is_fa = (outcome == 'FA')
        is_go = (t.change_size is not None and t.change_size > CATCH_THRESHOLD)
        is_catch = (t.change_size is not None and t.change_size <= CATCH_THRESHOLD)
        rows.append({
            'trial_index': t.trial_index,
            'outcome': outcome,
            'change_size': t.change_size if t.change_size is not None else 1.0,
            'reaction_time': t.reaction_time,
            'is_hit': is_hit,
            'is_miss': is_miss,
            'is_fa': is_fa,
            'is_go': is_go,
            'is_catch': is_catch,
        })

    df = pd.DataFrame(rows)
    session_name = session.session_id or ""

    if df.empty:
        return {"y": np.array([]), "X": np.empty((0, len(FEATURE_NAMES))),
                "df": df, "session_name": session_name,
                "subject_id": session.subject_id,
                "feature_names": list(FEATURE_NAMES)}

    # Filter excluded outcomes
    mask = ~df["outcome"].isin(list(exclude_outcomes))
    df = df[mask].reset_index(drop=True)
    if df.empty:
        return {"y": np.array([]), "X": np.empty((0, len(FEATURE_NAMES))),
                "df": df, "session_name": session_name,
                "subject_id": session.subject_id,
                "feature_names": list(FEATURE_NAMES)}

    # --- Binary choice: 1 = licked, 0 = no-lick ---
    y = (df["is_hit"] | df["is_fa"]).astype(float).values

    # --- Stimulus strength ---
    # log2(change_size); catch trials have change_size~=1 -> log2(1)=0.
    stim = np.log2(np.clip(df["change_size"].values.astype(float), 1.0, None))
    stim = np.nan_to_num(stim, nan=0.0)

    # --- History features ---
    prev_choice = np.zeros(len(df))
    prev_choice[1:] = y[:-1]

    prev_reward = np.zeros(len(df))
    hit_on_go = (df["is_hit"] & df["is_go"]).astype(float).values
    prev_reward[1:] = hit_on_go[:-1]

    # --- Impulsivity history ---
    prev_early_lick = np.zeros(len(df))
    prev_early_lick[1:] = df["is_fa"].values[:-1].astype(float)

    # --- Design matrix ---
    X = np.column_stack([
        np.ones(len(df)),   # bias / intercept
        stim,               # stimulus strength
        prev_choice,        # previous choice
        prev_reward,        # previous reward
        prev_early_lick,    # previous trial was early lick
    ])

    return {
        "y": y,
        "X": X,
        "df": df,
        "session_name": session_name,
        "subject_id": session.subject_id,
        "feature_names": list(FEATURE_NAMES),
    }


# =====================================================================
# Configuration
# =====================================================================

@dataclass
class GLMHMMConfig:
    """Hyper-parameters for GLM-HMM fitting."""
    max_iter: int = 200
    tol: float = 1e-4
    n_restarts: int = 20
    self_transition_prior: float = 0.8
    l2_penalty: float = 0.0
    glm_max_iter: int = 100
    verbose: bool = True


# =====================================================================
# Core model
# =====================================================================

class GLMHMM:
    """Bernoulli GLM-HMM for trial-by-trial behavioral state inference.

    Parameters
    ----------
    n_states : int
        Number of latent states K.
    n_features : int
        Dimensionality D of the covariate vector (including bias).
    config : GLMHMMConfig, optional
        Training hyper-parameters.
    """

    def __init__(self, n_states: int, n_features: int,
                 config: Optional[GLMHMMConfig] = None):
        self.n_states = n_states
        self.n_features = n_features
        self.config = config or GLMHMMConfig()

        # Parameters (set by _init_params or fit)
        self._weights: Optional[np.ndarray] = None   # (K, D)
        self._log_A: Optional[np.ndarray] = None     # (K, K)
        self._log_pi: Optional[np.ndarray] = None    # (K,)

        # Diagnostics
        self.train_ll_history: List[float] = []
        self.converged: bool = False
        self.feature_names: List[str] = list(FEATURE_NAMES)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @property
    def transition_matrix(self) -> np.ndarray:
        return np.exp(self._log_A)

    @property
    def initial_state_dist(self) -> np.ndarray:
        return np.exp(self._log_pi)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_params(self, seed: Optional[int] = None, smart: bool = False):
        rng = np.random.default_rng(seed)
        K, D = self.n_states, self.n_features

        # Transition matrix: strong self-transition
        p_self = self.config.self_transition_prior
        A = np.full((K, K), (1 - p_self) / max(K - 1, 1))
        np.fill_diagonal(A, p_self)
        A += rng.uniform(0, 0.02, (K, K))
        A /= A.sum(axis=1, keepdims=True)
        self._log_A = np.log(A + _EPS)

        # Initial state: uniform
        pi = np.ones(K) / K
        self._log_pi = np.log(pi + _EPS)

        # GLM weights
        if smart and K >= 2:
            bias_vals = np.linspace(-2, 2, K)
            self._weights = np.zeros((K, D))
            self._weights[:, 0] = bias_vals
            if D > 1:
                self._weights[:, 1] = rng.uniform(0.5, 1.5, K)
            self._weights += rng.normal(0, 0.1, (K, D))
        else:
            self._weights = rng.normal(0, 0.5, (K, D))

    # ------------------------------------------------------------------
    # Emission model
    # ------------------------------------------------------------------

    def _emission_log_likes(self, y: np.ndarray, X: np.ndarray) -> np.ndarray:
        T = len(y)
        K = self.n_states
        logits = X @ self._weights.T   # (T, K)
        ll = np.empty((T, K))
        for k in range(K):
            ll[:, k] = _log_bernoulli(y, logits[:, k])
        return ll

    # ------------------------------------------------------------------
    # Forward-backward
    # ------------------------------------------------------------------

    def _forward(self, log_likes: np.ndarray) -> Tuple[np.ndarray, float]:
        T, K = log_likes.shape
        log_alpha = np.empty((T, K))
        log_alpha[0] = self._log_pi + log_likes[0]
        for t in range(1, T):
            log_alpha[t] = log_likes[t] + logsumexp(
                log_alpha[t - 1, :, None] + self._log_A, axis=0
            )
        log_marginal = float(logsumexp(log_alpha[-1]))
        return log_alpha, log_marginal

    def _backward(self, log_likes: np.ndarray) -> np.ndarray:
        T, K = log_likes.shape
        log_beta = np.zeros((T, K))
        for t in range(T - 2, -1, -1):
            log_beta[t] = logsumexp(
                self._log_A + log_likes[t + 1] + log_beta[t + 1], axis=1
            )
        return log_beta

    # ------------------------------------------------------------------
    # E-step (single session)
    # ------------------------------------------------------------------

    def _e_step_session(
        self, y: np.ndarray, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        T = len(y)
        if T == 0:
            K = self.n_states
            return (np.empty((0, K)), np.zeros((K, K)), 0.0)

        log_likes = self._emission_log_likes(y, X)
        log_alpha, log_Z = self._forward(log_likes)
        log_beta = self._backward(log_likes)

        # State posteriors
        log_gamma = log_alpha + log_beta - log_Z
        gamma = np.exp(log_gamma)
        gamma = np.clip(gamma, _EPS, None)
        gamma /= gamma.sum(axis=1, keepdims=True)

        # Transition sufficient statistics
        if T > 1:
            log_xi = (
                log_alpha[:-1, :, None]
                + self._log_A[None, :, :]
                + log_likes[1:, None, :]
                + log_beta[1:, None, :]
                - log_Z
            )
            xi_sum = np.exp(log_xi).sum(axis=0)
        else:
            xi_sum = np.zeros((self.n_states, self.n_states))

        return gamma, xi_sum, log_Z

    # ------------------------------------------------------------------
    # M-step
    # ------------------------------------------------------------------

    def _fit_glm_state(self, X: np.ndarray, y: np.ndarray,
                       gamma_k: np.ndarray) -> np.ndarray:
        w0 = self._weights[0] if self._weights is not None else np.zeros(self.n_features)
        w0 = np.array(w0, dtype=float)
        result = minimize(
            _nll_and_grad,
            x0=w0,
            args=(X, y, gamma_k, self.config.l2_penalty),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": self.config.glm_max_iter, "ftol": 1e-8},
        )
        return result.x

    def _m_step(
        self,
        sessions_data: List[Dict[str, Any]],
        all_gamma: List[np.ndarray],
        total_xi: np.ndarray,
        total_init: np.ndarray,
    ):
        K = self.n_states

        # Transition matrix
        row_sums = total_xi.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, _EPS)
        A = total_xi / row_sums
        self._log_A = np.log(A + _EPS)

        # Initial state distribution
        pi = total_init / max(total_init.sum(), _EPS)
        self._log_pi = np.log(pi + _EPS)

        # GLM weights per state
        y_all = np.concatenate([s["y"] for s in sessions_data])
        X_all = np.concatenate([s["X"] for s in sessions_data])
        gamma_all = np.concatenate(all_gamma)

        for k in range(K):
            self._weights[k] = self._fit_glm_state(X_all, y_all, gamma_all[:, k])

    # ------------------------------------------------------------------
    # EM fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        sessions_data: List[Dict[str, Any]],
        seed: Optional[int] = None,
        smart_init: bool = False,
    ) -> float:
        self._init_params(seed=seed, smart=smart_init)
        K = self.n_states
        prev_ll = -np.inf
        self.train_ll_history = []

        for iteration in range(self.config.max_iter):
            # E-step
            all_gamma: List[np.ndarray] = []
            total_xi = np.zeros((K, K))
            total_init = np.zeros(K)
            total_ll = 0.0

            for s in sessions_data:
                if len(s["y"]) == 0:
                    continue
                gamma, xi_sum, ll = self._e_step_session(s["y"], s["X"])
                all_gamma.append(gamma)
                total_xi += xi_sum
                total_init += gamma[0]
                total_ll += ll

            self.train_ll_history.append(total_ll)

            # Convergence check
            rel_change = abs(total_ll - prev_ll) / max(abs(prev_ll), 1.0)
            if iteration > 0 and rel_change < self.config.tol:
                self.converged = True
                if self.config.verbose:
                    print(f"  EM converged at iteration {iteration + 1}  "
                          f"(LL={total_ll:.2f}, delta={rel_change:.2e})")
                break
            prev_ll = total_ll

            # M-step
            self._m_step(sessions_data, all_gamma, total_xi, total_init)

        if not self.converged and self.config.verbose:
            print(f"  EM did not converge after {self.config.max_iter} iterations "
                  f"(LL={total_ll:.2f})")

        return total_ll

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def most_likely_states(self, session_data: Dict[str, Any]) -> np.ndarray:
        """Viterbi decoding — most likely state sequence."""
        y, X = session_data["y"], session_data["X"]
        T = len(y)
        if T == 0:
            return np.array([], dtype=int)

        K = self.n_states
        log_likes = self._emission_log_likes(y, X)

        log_delta = np.empty((T, K))
        psi = np.zeros((T, K), dtype=int)

        log_delta[0] = self._log_pi + log_likes[0]
        for t in range(1, T):
            candidates = log_delta[t - 1, :, None] + self._log_A
            psi[t] = candidates.argmax(axis=0)
            log_delta[t] = log_likes[t] + candidates.max(axis=0)

        z = np.empty(T, dtype=int)
        z[-1] = int(log_delta[-1].argmax())
        for t in range(T - 2, -1, -1):
            z[t] = psi[t + 1, z[t + 1]]
        return z

    def state_posteriors(self, session_data: Dict[str, Any]) -> np.ndarray:
        y, X = session_data["y"], session_data["X"]
        if len(y) == 0:
            return np.empty((0, self.n_states))
        gamma, _, _ = self._e_step_session(y, X)
        return gamma

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def log_likelihood(self, sessions_data: List[Dict[str, Any]]) -> float:
        ll = 0.0
        for s in sessions_data:
            if len(s["y"]) == 0:
                continue
            _, _, ll_s = self._e_step_session(s["y"], s["X"])
            ll += ll_s
        return ll

    def n_params(self) -> int:
        K, D = self.n_states, self.n_features
        return K * D + K * (K - 1) + (K - 1)

    def _total_trials(self, sessions_data: List[Dict[str, Any]]) -> int:
        return sum(len(s["y"]) for s in sessions_data)

    def bic(self, sessions_data: List[Dict[str, Any]]) -> float:
        ll = self.log_likelihood(sessions_data)
        n = self._total_trials(sessions_data)
        return -2 * ll + self.n_params() * np.log(max(n, 1))

    def aic(self, sessions_data: List[Dict[str, Any]]) -> float:
        ll = self.log_likelihood(sessions_data)
        return -2 * ll + 2 * self.n_params()

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------

    def state_psychometrics(
        self,
        stim_values: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        if stim_values is None:
            stim_values = np.array([0, 0.32, 0.43, 0.58, 1.0, 2.0])
        rows = []
        for k in range(self.n_states):
            for sv in stim_values:
                x = np.zeros(self.n_features)
                x[0] = 1.0
                x[1] = sv
                p = float(expit(self._weights[k] @ x))
                rows.append({"state": k, "stimulus": sv, "p_lick": p})
        return pd.DataFrame(rows)

    def sort_states_by_bias(self):
        """Re-order states so State 0 = lowest bias (most disengaged)."""
        order = np.argsort(self._weights[:, 0])
        self._weights = self._weights[order]
        self._log_A = self._log_A[order][:, order]
        self._log_pi = self._log_pi[order]

    def summary(self) -> str:
        K, D = self.n_states, self.n_features
        lines = [
            f"GLM-HMM  K={K}  D={D}  params={self.n_params()}  "
            f"converged={self.converged}",
            "",
            "GLM weights (rows=states, cols=features):",
            f"  features: {self.feature_names}",
        ]
        for k in range(K):
            w_str = "  ".join(f"{v:+.3f}" for v in self._weights[k])
            lines.append(f"  State {k}: [{w_str}]")
        lines.append("")
        lines.append("Transition matrix A:")
        A = self.transition_matrix
        for k in range(K):
            row = "  ".join(f"{v:.3f}" for v in A[k])
            lines.append(f"  [{row}]")
        lines.append("")
        lines.append(f"Initial state dist: "
                      f"[{' '.join(f'{v:.3f}' for v in self.initial_state_dist)}]")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path) -> "GLMHMM":
        with open(path, "rb") as f:
            return pickle.load(f)


# =====================================================================
# Model selection
# =====================================================================

@dataclass
class KFitTask:
    K: int
    sessions_data: List[Dict[str, Any]]
    n_features: int
    config: GLMHMMConfig
    n_restarts: int
    base_seed: int = 0


def _fit_single_K(task: KFitTask) -> Tuple[int, Optional["GLMHMM"], float, int]:
    K = task.K
    best_ll_K = -np.inf
    best_model_K = None
    n_failures = 0

    for r in range(task.n_restarts):
        model = GLMHMM(K, task.n_features, config=task.config)
        smart = (r == 0)
        seed = task.base_seed + r * 137 + K * 7
        try:
            ll = model.fit(task.sessions_data, seed=seed, smart_init=smart)
            if ll > best_ll_K:
                best_ll_K = ll
                best_model_K = model
        except Exception:
            n_failures += 1
            continue

    if best_model_K is not None:
        best_model_K.sort_states_by_bias()

    return K, best_model_K, best_ll_K, n_failures


def fit_best_model(
    sessions_data: List[Dict[str, Any]],
    K_range: Sequence[int] = (2, 3, 4, 5),
    config: Optional[GLMHMMConfig] = None,
    verbose: bool = True,
    n_workers: int = 1,
    seed: int = 0,
) -> Tuple["GLMHMM", pd.DataFrame, Dict[int, "GLMHMM"]]:
    """Fit GLM-HMMs for each K, selecting the best by BIC."""
    cfg = config or GLMHMMConfig()
    cfg_copy = GLMHMMConfig(**{k: getattr(cfg, k) for k in cfg.__dataclass_fields__})
    cfg_copy.verbose = False

    n_features = sessions_data[0]["X"].shape[1] if len(sessions_data) > 0 else len(FEATURE_NAMES)

    tasks = [
        KFitTask(
            K=K, sessions_data=sessions_data, n_features=n_features,
            config=cfg_copy, n_restarts=cfg.n_restarts, base_seed=seed,
        )
        for K in K_range
    ]

    records = []
    all_models: Dict[int, GLMHMM] = {}

    if n_workers > 1:
        if verbose:
            print(f"\nFitting {len(K_range)} K values in parallel with {n_workers} workers...")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_fit_single_K, task) for task in tasks]
            for future in tqdm(futures, desc="Fitting K values", disable=not verbose):
                try:
                    K, best_model_K, best_ll_K, n_failures = future.result()
                    if best_model_K is not None:
                        bic_val = best_model_K.bic(sessions_data)
                        aic_val = best_model_K.aic(sessions_data)
                        all_models[K] = best_model_K
                        records.append({"K": K, "best_ll": best_ll_K, "bic": bic_val,
                                        "aic": aic_val, "n_params": best_model_K.n_params()})
                        if verbose:
                            msg = f"K={K}  LL={best_ll_K:.2f}  BIC={bic_val:.2f}  AIC={aic_val:.2f}"
                            if n_failures > 0:
                                msg += f"  ({n_failures}/{cfg.n_restarts} restarts failed)"
                            print(f"  {msg}")
                except Exception as exc:
                    if verbose:
                        print(f"  K={K}: FAILED - {exc}")
    else:
        for task in tasks:
            K = task.K
            if verbose:
                print(f"\n{'='*50}")
                print(f"Fitting K={K} states  ({cfg.n_restarts} restarts)")
                print(f"{'='*50}")

            K, best_model_K, best_ll_K, n_failures = _fit_single_K(task)

            if best_model_K is not None:
                bic_val = best_model_K.bic(sessions_data)
                aic_val = best_model_K.aic(sessions_data)
                all_models[K] = best_model_K
                records.append({"K": K, "best_ll": best_ll_K, "bic": bic_val,
                                "aic": aic_val, "n_params": best_model_K.n_params()})
                if verbose:
                    msg = f">>> K={K}  best LL={best_ll_K:.2f}  BIC={bic_val:.2f}  AIC={aic_val:.2f}"
                    if n_failures > 0:
                        msg += f"  ({n_failures}/{cfg.n_restarts} restarts failed)"
                    print(f"  {msg}")

    selection_df = pd.DataFrame(records)
    if selection_df.empty:
        raise RuntimeError("All model fits failed.")

    best_K = int(selection_df.loc[selection_df["bic"].idxmin(), "K"])
    best_model = all_models[best_K]
    if verbose:
        print(f"\n*** Best model: K={best_K} (by BIC) ***\n")
        print(best_model.summary())

    return best_model, selection_df, all_models


# =====================================================================
# Auto-labelling
# =====================================================================

def auto_label_states(model: GLMHMM) -> List[str]:
    """Assign human-readable labels based on psychometric profile.

    After states are sorted by ascending bias:
      - P(lick | catch) > 0.65  → "Biased"
      - P(lick | max_stim) < 0.40  → "Disengaged"
      - Otherwise  → "Engaged"
    """
    K = model.n_states
    D = model.n_features
    labels = []
    for k in range(K):
        x_catch = np.zeros(D); x_catch[0] = 1.0
        x_high  = np.zeros(D); x_high[0] = 1.0; x_high[1] = 2.0
        p_catch = float(expit(model.weights[k] @ x_catch))
        p_high = float(expit(model.weights[k] @ x_high))

        if p_catch > 0.65:
            labels.append("Biased")
        elif p_high < 0.40:
            labels.append("Disengaged")
        else:
            labels.append("Engaged")

    # De-duplicate
    seen = {}
    for i, lab in enumerate(labels):
        if lab in seen:
            seen[lab] += 1
            labels[i] = f"{lab}_{seen[lab]}"
        else:
            seen[lab] = 1
    for lab_base, count in seen.items():
        if count > 1:
            first_idx = next(j for j, l in enumerate(labels) if l == lab_base)
            labels[first_idx] = f"{lab_base}_1"

    return labels


def rename_labels(labels: List[str]) -> List[str]:
    """Apply the canonical renaming (Biased→Impulsive, etc.)."""
    return [HMM_LABEL_RENAME.get(lab, lab) for lab in labels]


# =====================================================================
# Session decoding
# =====================================================================

def decode_session(
    model: GLMHMM,
    session,
    state_labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Decode a session: return trial DataFrame with state assignments.

    Columns added: hmm_state, hmm_state_label, p_state_0...K-1.
    """
    data = prepare_session_data(session)
    if len(data["y"]) == 0:
        return data["df"]

    states = model.most_likely_states(data)
    posteriors = model.state_posteriors(data)

    df = data["df"].copy()
    df["hmm_state"] = states
    if state_labels is not None:
        df["hmm_state_label"] = [state_labels[s] for s in states]
    for k in range(model.n_states):
        df[f"p_state_{k}"] = posteriors[:, k]

    return df
