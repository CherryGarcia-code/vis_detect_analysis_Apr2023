"""Downstream analysis utilities for GLM-HMM behavioral states.

Ported from vis_detect_analysis_Sep2025. Contains only the behavioral
functions; neural (ephys-specific) functions are replaced by photometry
PETH-based analyses in separate scripts.

Provides:
  - Cross-validation (leave-one-session-out)
  - Per-state behavioral metrics (d', criterion, hit/FA rates)
  - Learning trajectory (state fractions + d' per session)
  - Online (causal) single-trial state prediction
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm


# =====================================================================
# 1.  Cross-Validation  (Leave-One-Session-Out)
# =====================================================================

def loso_cross_validation(
    sessions_data: List[Dict[str, Any]],
    K: int,
    *,
    config=None,
    n_restarts: int = 10,
    max_iter: int = 200,
    seed: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Leave-One-Session-Out cross-validation for a GLM-HMM at a given K.

    Returns DataFrame: fold, held_out_session, n_trials_test,
    train_ll, test_ll, test_ll_per_trial, test_accuracy.
    """
    from visdetect_photom.analysis.hmm import GLMHMM, GLMHMMConfig

    cfg = config or GLMHMMConfig(
        max_iter=max_iter, n_restarts=n_restarts, verbose=False
    )
    cfg_fold = GLMHMMConfig(**{
        k: getattr(cfg, k) for k in cfg.__dataclass_fields__
    })
    cfg_fold.verbose = False

    n_features = sessions_data[0]["X"].shape[1]
    n_sessions = len(sessions_data)
    records = []

    for fold_idx in range(n_sessions):
        held_out = sessions_data[fold_idx]
        train = [s for i, s in enumerate(sessions_data) if i != fold_idx]
        sname = held_out.get("session_name", f"session_{fold_idx}")

        if verbose:
            print(f"  Fold {fold_idx + 1}/{n_sessions}  "
                  f"(held-out: {sname}, {len(held_out['y'])} trials)")

        best_ll = -np.inf
        best_model = None
        for r in range(cfg.n_restarts):
            model = GLMHMM(K, n_features, config=cfg_fold)
            try:
                ll = model.fit(train, seed=seed + r * 137 + fold_idx * 7)
            except Exception:
                continue
            if ll > best_ll:
                best_ll = ll
                best_model = model

        if best_model is None:
            warnings.warn(f"All restarts failed for fold {fold_idx}")
            continue

        # Evaluate on held-out session
        test_ll = best_model.log_likelihood([held_out])
        n_test = len(held_out["y"])

        # Prediction accuracy
        states = best_model.most_likely_states(held_out)
        X_test = held_out["X"]
        y_test = held_out["y"]
        p_lick = np.array([
            expit(best_model.weights[states[t]] @ X_test[t])
            for t in range(n_test)
        ])
        pred_choice = (p_lick >= 0.5).astype(float)
        accuracy = np.mean(pred_choice == y_test)

        records.append({
            "fold": fold_idx,
            "held_out_session": sname,
            "n_trials_test": n_test,
            "train_ll": best_ll,
            "test_ll": test_ll,
            "test_ll_per_trial": test_ll / max(n_test, 1),
            "test_accuracy": accuracy,
        })

    return pd.DataFrame(records)


# =====================================================================
# 2.  Per-State Behavioral Metrics
# =====================================================================

def compute_state_behavioral_metrics(
    assignments_df: pd.DataFrame,
    state_labels: List[str],
    n_states: int,
) -> pd.DataFrame:
    """Compute SDT and behavioral metrics per HMM state.

    Returns DataFrame: state, label, n_trials, fraction, hit_rate_go,
    catch_lick_rate, early_lick_rate, dprime, criterion.
    """
    rows = []
    N = len(assignments_df)
    for k in range(n_states):
        sub = assignments_df[assignments_df["hmm_state"] == k]
        n = len(sub)
        if n == 0:
            rows.append({"state": k, "label": state_labels[k], "n_trials": 0,
                          "fraction": 0.0})
            continue

        # SDT: go and catch trials with outcome Hit or Miss only
        go_all = sub[sub["is_go"] == True]
        catch_all = sub[sub["is_catch"] == True]

        out_go = go_all["outcome"] if "outcome" in go_all.columns else pd.Series(dtype=object)
        out_catch = catch_all["outcome"] if "outcome" in catch_all.columns else pd.Series(dtype=object)
        go = go_all[out_go.isin(["Hit", "Miss"])]
        catch = catch_all[out_catch.isin(["Hit", "Miss"])]

        hit_rate = go["is_hit"].mean() if len(go) > 0 else 0.0
        early_lick_rate = sub["is_fa"].mean()
        catch_lick = catch["is_hit"].mean() if len(catch) > 0 else 0.0

        if len(go) > 0 and len(catch) > 0:
            hr_c = np.clip(hit_rate, 0.01, 0.99)
            far_c = np.clip(catch_lick, 0.01, 0.99)
            dprime = float(norm.ppf(hr_c) - norm.ppf(far_c))
            criterion = float(-0.5 * (norm.ppf(hr_c) + norm.ppf(far_c)))
        else:
            dprime = np.nan
            criterion = np.nan

        rows.append({
            "state": k,
            "label": state_labels[k],
            "n_trials": n,
            "fraction": n / N,
            "hit_rate_go": hit_rate,
            "catch_lick_rate": catch_lick,
            "early_lick_rate": early_lick_rate,
            "dprime": dprime,
            "criterion": criterion,
        })
    return pd.DataFrame(rows)


def compute_per_session_state_metrics(
    assignments_df: pd.DataFrame,
    state_labels: List[str],
    n_states: int,
) -> pd.DataFrame:
    """Compute behavioral metrics per session x state."""
    all_rows = []
    for sname, sdf in assignments_df.groupby("session_name", sort=False):
        per_state = compute_state_behavioral_metrics(sdf, state_labels, n_states)
        per_state.insert(0, "session_name", sname)
        all_rows.append(per_state)
    return pd.concat(all_rows, ignore_index=True)


# =====================================================================
# 3.  Across-Learning State Dynamics
# =====================================================================

def compute_learning_trajectory(
    assignments_df: pd.DataFrame,
    state_labels: List[str],
    n_states: int,
    session_order: Optional[List[str]] = None,
) -> pd.DataFrame:
    """State fractions and d' per session, ordered chronologically.

    Returns DataFrame: session_name, session_idx, subject_id,
    overall_dprime, frac_<label>, dprime_<label> for each state.
    """
    metrics = compute_per_session_state_metrics(
        assignments_df, state_labels, n_states
    )
    sessions = (
        session_order
        if session_order
        else list(assignments_df["session_name"].unique())
    )

    rows = []
    for idx, sname in enumerate(sessions):
        sdf = assignments_df[assignments_df["session_name"] == sname]
        if len(sdf) == 0:
            continue
        row: Dict[str, Any] = {
            "session_name": sname,
            "session_idx": idx,
            "subject_id": sdf["subject_id"].iloc[0] if "subject_id" in sdf.columns else "",
        }

        # Overall d'
        go = sdf[(sdf["is_go"] == True) & (sdf["outcome"].isin(["Hit", "Miss"]))]
        catch = sdf[(sdf["is_catch"] == True) & (sdf["outcome"].isin(["Hit", "Miss"]))]
        if len(go) > 0 and len(catch) > 0:
            hr = np.clip(go["is_hit"].mean(), 0.01, 0.99)
            far = np.clip(catch["is_hit"].mean(), 0.01, 0.99)
            row["overall_dprime"] = float(norm.ppf(hr) - norm.ppf(far))
        else:
            row["overall_dprime"] = np.nan

        # Per-state fractions and d'
        s_metrics = metrics[metrics["session_name"] == sname]
        for k in range(n_states):
            lbl = state_labels[k]
            s_row = s_metrics[s_metrics["state"] == k]
            row[f"frac_{lbl}"] = float(s_row["fraction"].values[0]) if len(s_row) > 0 else 0.0
            row[f"dprime_{lbl}"] = float(s_row["dprime"].values[0]) if len(s_row) > 0 else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


# =====================================================================
# 4.  Online (Causal) Prediction
# =====================================================================

def forward_only_state_posteriors(model, session_data: Dict[str, Any]) -> np.ndarray:
    """Causal (forward-only) state posteriors P(z_t | y_{1:t}).

    Unlike full forward-backward, these posteriors only use past data,
    suitable for real-time / online state prediction.

    Returns (T, K) array.
    """
    from scipy.special import logsumexp as _logsumexp

    y, X = session_data["y"], session_data["X"]
    T = len(y)
    if T == 0:
        return np.empty((0, model.n_states))

    log_likes = model._emission_log_likes(y, X)
    K = model.n_states

    # Forward pass (same as model._forward)
    log_alpha = np.empty((T, K))
    log_alpha[0] = model._log_pi + log_likes[0]

    for t in range(1, T):
        log_alpha[t] = log_likes[t] + _logsumexp(
            log_alpha[t - 1, :, None] + model._log_A, axis=0
        )

    # Normalize each timestep independently (causal posterior)
    posteriors = np.empty((T, K))
    for t in range(T):
        log_norm = _logsumexp(log_alpha[t])
        posteriors[t] = np.exp(log_alpha[t] - log_norm)

    return posteriors


def predict_trial_by_trial(
    model,
    session_data: Dict[str, Any],
    causal: bool = True,
) -> pd.DataFrame:
    """Per-trial predictions: P(lick), state, posteriors.

    Parameters
    ----------
    model : GLMHMM
    session_data : prepared session dict
    causal : if True, use forward-only posteriors; else full forward-backward.

    Returns DataFrame with: trial_idx, y, p_lick, predicted_state,
    p_state_0..K-1.
    """
    y = session_data["y"]
    X = session_data["X"]
    T = len(y)
    if T == 0:
        return pd.DataFrame()

    if causal:
        posteriors = forward_only_state_posteriors(model, session_data)
    else:
        posteriors = model.state_posteriors(session_data)

    predicted_state = posteriors.argmax(axis=1)

    # P(lick) = sum_k P(z=k) * sigmoid(w_k @ x)
    p_lick = np.zeros(T)
    for t in range(T):
        for k in range(model.n_states):
            p_lick[t] += posteriors[t, k] * expit(model.weights[k] @ X[t])

    rows = {
        "trial_idx": np.arange(T),
        "y": y,
        "p_lick": p_lick,
        "predicted_state": predicted_state,
    }
    for k in range(model.n_states):
        rows[f"p_state_{k}"] = posteriors[:, k]

    return pd.DataFrame(rows)


# =====================================================================
# 5.  I/O Utilities
# =====================================================================

def load_hmm_results(results_dir, K=None):
    """Load saved HMM results: model, assignments, labels.

    Parameters
    ----------
    results_dir : str or Path
    K : int, optional. If None, loads the best model.

    Returns (model, assignments_df, state_labels).
    """
    from pathlib import Path
    from visdetect_photom.analysis.hmm import GLMHMM

    d = Path(results_dir)
    suffix = f"_K{K}" if K else ""

    model_path = d / f"model{suffix}.pkl"
    assignments_path = d / f"state_assignments{suffix}.csv"
    labels_path = d / f"state_labels{suffix}.json"

    model = GLMHMM.load(model_path)

    import json
    assignments_df = pd.read_csv(assignments_path)
    with open(labels_path) as f:
        state_labels = json.load(f)

    return model, assignments_df, state_labels
