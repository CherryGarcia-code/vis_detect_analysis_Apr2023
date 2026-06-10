"""Swappable trial-level behavioral-state labeling.

`StateProvider` is the seam: any object with `get_trial_states(session)` returning
one label per trial works. The HMM is just one backend (`HMMStateProvider`); the
default (`PooledStateProvider`) does no filtering. Keep this SEPARATE from the
session-level learning-stage logic in `core/staging.py`.
"""
from typing import Protocol, Iterable, Set
import numpy as np


class StateProvider(Protocol):
    def get_trial_states(self, session) -> np.ndarray:
        """Return an array of per-trial state labels, len == len(session.trials)."""
        ...


class PooledStateProvider:
    """Default: no state distinction; every trial is 'All'."""

    def get_trial_states(self, session) -> np.ndarray:
        return np.array(["All"] * len(session.trials), dtype=object)


class HMMStateProvider:
    """Trial states from a fitted GLM-HMM. Lazy: artifacts load on first use."""

    def __init__(self, results_dir, K=None):
        self.results_dir = results_dir
        self.K = K
        self._model = None
        self._labels = None

    def _ensure_loaded(self):
        if self._model is None:
            from visdetect_photom.analysis.hmm_downstream import load_hmm_results
            self._model, _, self._labels = load_hmm_results(self.results_dir, self.K)

    def get_trial_states(self, session) -> np.ndarray:
        self._ensure_loaded()
        from visdetect_photom.analysis.hmm import decode_session
        df = decode_session(self._model, session, self._labels)
        labels = np.array(["NA"] * len(session.trials), dtype=object)
        if "hmm_state_label" in df.columns and "trial_index" in df.columns:
            for _, row in df.iterrows():
                ti = int(row["trial_index"])
                if 0 <= ti < len(labels):
                    labels[ti] = row["hmm_state_label"]
        return labels


def filter_trials_by_state(session, provider: StateProvider,
                           keep_states: Iterable[str]) -> Set[int]:
    """Return the set of trial indices whose state is in keep_states."""
    states = provider.get_trial_states(session)
    keep = set(keep_states)
    return {i for i, s in enumerate(states) if s in keep}
