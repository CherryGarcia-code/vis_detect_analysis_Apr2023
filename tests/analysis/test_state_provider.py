import numpy as np
from types import SimpleNamespace
from visdetect_photom.analysis.state_provider import (
    PooledStateProvider, HMMStateProvider, filter_trials_by_state,
)


def _session(n):
    return SimpleNamespace(trials=[SimpleNamespace(trial_index=i) for i in range(n)],
                           subject_id="013")


def test_pooled_returns_all_label():
    s = _session(5)
    states = PooledStateProvider().get_trial_states(s)
    assert list(states) == ["All"] * 5


def test_filter_keeps_matching_indices():
    s = _session(4)

    class Fake:
        def get_trial_states(self, session):
            return np.array(["Engaged", "Disengaged", "Engaged", "NA"], dtype=object)

    keep = filter_trials_by_state(s, Fake(), {"Engaged"})
    assert keep == {0, 2}


def test_pooled_filter_keeps_everything():
    s = _session(3)
    keep = filter_trials_by_state(s, PooledStateProvider(), {"All"})
    assert keep == {0, 1, 2}


def test_hmm_provider_is_lazy_constructible():
    # Constructing must NOT import/load HMM artifacts.
    p = HMMStateProvider(results_dir="does/not/exist")
    assert p.results_dir == "does/not/exist"
