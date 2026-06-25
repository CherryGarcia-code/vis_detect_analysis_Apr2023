# Intersectional MOs-recipient Cohort (BG_027–030) — C1+C2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the audited C1 (waiting-period brake) and C2 (response geometry) analyses on the input-defined GCaMP6f cohort BG_027–030 in a 2×2 (cell-type × region/input), with session as the statistical unit and a rank-based secondary comparison to the matched bulk-8m cells — without ever pooling the two cohorts.

**Architecture:** Approach B — leave the bulk C1/C2 scripts untouched; add a `core/cohort.py` registry + session-unit aggregator + cross-cohort matcher, a session-level delta/AUROC variant in `suppression.py`, and new thin scripts under `scripts/analysis/intersectional/` that import the audited primitives. A Phase-0 Opus audit (Workflow) of the reused machinery gates the cohort work. The never-pool rule is enforced centrally in `excluded_mice`.

**Tech Stack:** Python 3 (invoked as `py`), numpy<2.0, pandas<2.3, scipy<2.0, matplotlib<3.9, seaborn<0.14; pytest. Package uses a `src/` layout (not pip-installed).

## Global Constraints

- **Invoke Python as `py`** (Windows). Tests run via `py -m pytest`.
- **Never pool cohorts.** BG_027–030 = GCaMP6f, MOs-recipient, intersectional; never combined with bulk GCaMP8m (BG_008–020). Enforced in `excluded_mice` (Task 3).
- **`fa` ≠ SDT-FA.** Behavioral FA = anticipatory baseline lick; SDT-FA = catch-trial lick (`outcome=='Hit'` & `change_size <= CATCH_THRESHOLD`, `CATCH_THRESHOLD = 1.01`). Both tracks run separately.
- **Alignment:** never align FA/abort to a change that did not occur; grating onset = `absolute_change_time − change_time`.
- **D1 vs D2 = different animals** → group/cell-level **sign contrast**, never within-animal anticorrelation.
- **Normalization:** window scalars use the **session-z-scored** dF/F already on the trace, **no per-trial re-baseline**.
- **Stats:** non-parametric, two-sided, effect sizes alongside every p-value; all RNG **seed = 42**; bootstrap n=1000, permutation n=10000.
- **n = 1 mouse / cell** → results are **within-animal** (session-unit CIs generalize to this mouse's sessions, not a population of mice). No cross-mouse inferential test inside the cohort.
- **Cross-cohort comparison uses only rank/standardized quantities** (AUROC, push–pull sign, rank-biserial, latencies); **never raw dF/F magnitude**.
- **Reused constants:** `MIN_TRIALS_PER_GROUP` (suppression min-N), `HAZARD_SEED=42`, `HAZARD_RESAMPLES=20`, `MIN_PHOTOM_CSV_BYTES`, `PROF_MIN_SESSIONS`.
- **All spawned subagents / workflow agents run on Opus 4.8** (`model: 'opus'`), per the user's standing rule — no cheaper tier, ever.
- **Parallel-chat safety:** the shared working tree may be on another chat's branch; verify branch before any git op; do not modify the bulk C1/C2 scripts (`08_d1_d2_geometry.py`, `11_fa_suppression.py`).

---

## Prerequisite: implementation base branch (do this before Task 0)

The C1 *code* (`src/visdetect_photom/analysis/suppression.py`, `scripts/analysis/photometry/11_fa_suppression.py`) currently exists **only on local `main` (b2736d2)** — bundled with the parallel G1 chat's unpushed commits. `origin/main` (0843b35), the base of this design branch (`analysis/intersectional-mos-design`), has the C1 *docs* but not the code. C2 (`geometry.py`) and the shared primitives **are** on origin/main.

Phase 0 audits `suppression.py`, and Phases 2–3 import it, so the implementation worktree **must contain the C1 code.** Choose ONE (coordinate with the user / the G1 chat):

- **Preferred:** the C1 code is pushed to `origin/main`; then rebase this design branch onto the updated `origin/main` (`git rebase origin/main` inside the worktree) so it carries spec + plan + C1 code.
- **Fallback:** create the implementation worktree based on local `main` (b2736d2) and bring the spec+plan over: `git worktree add -b analysis/intersectional-mos ../_wt_intersectional_mos_impl main`, then `git cherry-pick <design-branch commits>` (the spec `8cc8095` and this plan's commit). This pulls the G1 commits into the base — acceptable but entangles histories; prefer the rebase path.

Confirm `py -c "import sys; sys.path.insert(0,'src'); import visdetect_photom.analysis.suppression"` succeeds in the worktree before Task 0.

---

## File structure

| File | Responsibility | Task |
|---|---|---|
| `src/visdetect_photom/core/cohort.py` | **NEW.** Cohort registry, `non_bulk_subjects`, `load_cohort_sessions`, `summarize_sessions_by_cell`, `match_cohort_cells`, rank-based guard | 1,4,5 |
| `src/visdetect_photom/core/constants.py` | **EDIT (additive).** Add 027–030 to `SUBJECT_GENOTYPE` + `SUBJECT_ROI_REGION` | 2 |
| `src/visdetect_photom/core/staging.py` | **EDIT (additive).** `excluded_mice` unions non-bulk cohort | 3 |
| `src/visdetect_photom/analysis/suppression.py` | **EDIT (additive).** `compute_session_delta_and_auroc` (session-unit) | 6 |
| `scripts/data_management/stage_intersectional_cohort.py` | **NEW.** Idempotent ceph→local staging + name-normalization | 7 |
| `scripts/analysis/intersectional/c1_cohort_suppression.py` | **NEW.** C1 scoped to cohort: 2×2, session-unit + PETHs | 8 |
| `scripts/analysis/intersectional/c2_cohort_geometry.py` | **NEW.** C2 scoped to cohort: 2×2, session-unit + PETHs | 9 |
| `scripts/analysis/intersectional/cohort_qc_report.py` | **NEW.** 6f QC pass-rate + metric-distribution report | 10 |
| `scripts/analysis/intersectional/cohort_cross_compare.py` | **NEW.** Rank-based bulk-vs-cohort comparison | 11 |
| `scripts/analysis/intersectional/cohort_companion_peths.py` | **NEW.** Per-cell companion PETH panels (alignments never mixed) | 12 |
| `tests/core/test_cohort.py`, `tests/core/test_excluded_cohort.py`, `tests/analysis/test_suppression_session_unit.py`, `tests/data_management/test_stage_intersectional.py`, `tests/scripts/test_cohort_smoke.py` | **NEW.** Tests | 1–11 |
| package modules | **EDIT only if Phase-0 audit finds bugs** (each fix + regression test) | 0 |

---

## Task 0: Phase-0 Opus correctness audit of the reused machinery (GATE)

**Files:**
- Create: `docs/superpowers/audits/2026-06-23-intersectional-mos-machinery-audit.md`
- Modify (only if bugs found): `src/visdetect_photom/analysis/suppression.py`, `geometry.py`, `group_statistics.py`, `core/qc.py`, `analysis/statistics.py`, `core/session.py`
- Test (per fix): `tests/analysis/` or `tests/core/` regression test

**Interfaces:**
- Produces: an audit report + a green test suite on the audited modules. No new public API.

This task is exploratory (findings are not knowable in advance), so it is a **procedure**, not a fixed code edit. The deliverable is concrete: a written report + any bug fixes shipped TDD-style + the full suite green.

- [ ] **Step 1: Run the audit workflow.** Author and run a Workflow (all agents `model: 'opus'`) that fans out parallel reviewers over (module × dimension), adversarially verifies each finding (≥2 skeptics, default-to-refuted), and synthesizes. Use this script shape:

```js
export const meta = {
  name: 'intersectional-machinery-audit',
  description: 'Audit reused C1/C2 photometry primitives against the discipline rules',
  phases: [{ title: 'Review' }, { title: 'Verify' }, { title: 'Synthesize' }],
}
const TARGETS = [
  {mod: 'analysis/suppression.py',  dims: ['fa-vs-sdt grouping','scheme1 window inclusion + motor buffer','scheme3 hazard match + determinism (seed=42,R=20)','session-z no per-trial rebaseline','AUROC orientation (withhold=positive)','min-N guards']},
  {mod: 'analysis/geometry.py',     dims: ['alignment (FA never aligned to change)','extract_peth unpacking/time-axis','push-pull sign correctness','per-mouse aggregation (no trial-count N)']},
  {mod: 'analysis/group_statistics.py', dims: ['auroc_score = U/(npos*nneg)','pushpull_sign_contrast CIs/seed','bootstrap_ci/permutation_test seeds','extract_peak abs-max sign preservation','latency extractors']},
  {mod: 'core/qc.py',               dims: ['merge_hemispheres logic (both/one/neither)','region pairing subject-aware','region_sources correctness']},
  {mod: 'analysis/statistics.py',   dims: ['calculate_sdt_metrics masking (go/catch by change_size)','extract_peth baseline/window']},
  {mod: 'core/session.py',          dims: ['abs_rt computation','grating-onset back-out = abs_change - change_time']},
]
const found = await pipeline(
  TARGETS.flatMap(t => t.dims.map(d => ({mod: t.mod, dim: d}))),
  item => agent(`Read src/visdetect_photom/${item.mod}. Audit ONLY the dimension "${item.dim}" against the project discipline rules in CLAUDE.md and docs/superpowers/specs/2026-06-23-intersectional-mos-cohort-design.md §3. Report concrete defects with file:line and a failing-case description, or state "clean".`,
    {label: `review:${item.mod}:${item.dim}`, phase: 'Review', model: 'opus', schema: {type:'object', properties:{defects:{type:'array', items:{type:'object', properties:{file:{type:'string'}, line:{type:'integer'}, claim:{type:'string'}, failing_case:{type:'string'}}, required:['file','claim','failing_case']}}}, required:['defects']}}),
  review => parallel((review.defects||[]).map(d => () =>
    agent(`Adversarially verify this claimed defect; default refuted=true if uncertain. Claim: ${d.claim}. Failing case: ${d.failing_case}. File ${d.file}:${d.line||'?'}.`,
      {label: `verify:${d.file}`, phase: 'Verify', model: 'opus', schema: {type:'object', properties:{refuted:{type:'boolean'}, reason:{type:'string'}}, required:['refuted','reason']}})
      .then(v => ({...d, refuted: v?.refuted, reason: v?.reason}))))
)
const confirmed = found.flat().filter(Boolean).filter(d => d.refuted === false)
return { confirmed }
```

- [ ] **Step 2: Write the audit report.** Save the confirmed defects (and "clean" dimensions) to `docs/superpowers/audits/2026-06-23-intersectional-mos-machinery-audit.md`. Commit the report.

```bash
git add docs/superpowers/audits/2026-06-23-intersectional-mos-machinery-audit.md
git commit -m "audit(intersectional): Phase-0 correctness review of reused C1/C2 machinery"
```

- [ ] **Step 3: For each confirmed defect, write a failing regression test.** Place in the matching `tests/` file; assert the correct behavior. Run `py -m pytest <test> -v` → expect FAIL.

- [ ] **Step 4: Fix the defect minimally; run the test → PASS.** Then run the full suite `py -m pytest -q` → all green (the bulk C1/C2 tests must still pass).

- [ ] **Step 5: Commit each fix.**

```bash
git add tests/ src/visdetect_photom/
git commit -m "fix(audit): <one-line defect summary> + regression test"
```

- [ ] **Step 6: Gate.** If zero confirmed defects, record "no defects found" in the report and proceed. The cohort tasks build only on the now-audited code.

---

## Task 1: Cohort registry — `core/cohort.py`

**Files:**
- Create: `src/visdetect_photom/core/cohort.py`
- Test: `tests/core/test_cohort.py`

**Interfaces:**
- Produces:
  - `COHORTS: dict` — keys `'bulk_8m'`, `'intersectional_mos'`; each value `{subjects: list[str], indicator: str}` (+ `inputs: dict` for intersectional).
  - `cohort_of(subject_id) -> str | None`
  - `subjects_in(cohort_name) -> list[str]`
  - `indicator_of(subject_id) -> str | None`
  - `cortical_input(subject_id) -> str | None` (e.g. `'aMOs'`, `'pMOs'`, else None)
  - `non_bulk_subjects() -> list[str]` (all subjects in cohorts other than `'bulk_8m'`)
  - `load_cohort_sessions(cohort_name, root_dir, max_sessions=None) -> list[Session]`

- [ ] **Step 1: Write the failing test.**

```python
# tests/core/test_cohort.py
from visdetect_photom.core import cohort

def test_registry_membership_and_lookups():
    assert set(cohort.subjects_in("intersectional_mos")) == {"BG_027","BG_028","BG_029","BG_030"}
    assert cohort.cohort_of("BG_027") == "intersectional_mos"
    assert cohort.cohort_of("BG_013") == "bulk_8m"
    assert cohort.cohort_of("BG_999") is None
    assert cohort.indicator_of("BG_029") == "GCaMP6f"
    assert cohort.indicator_of("BG_013") == "GCaMP8m"
    assert cohort.cortical_input("BG_027") == "aMOs"   # VMS
    assert cohort.cortical_input("BG_028") == "pMOs"   # DMS
    assert cohort.cortical_input("BG_013") is None
    assert set(cohort.non_bulk_subjects()) == {"BG_027","BG_028","BG_029","BG_030"}
```

- [ ] **Step 2: Run → FAIL** (`py -m pytest tests/core/test_cohort.py -v`; "No module named cohort").

- [ ] **Step 3: Implement.**

```python
# src/visdetect_photom/core/cohort.py
"""Cohort registry: which subjects belong to which recording cohort, their
indicator, and (for the intersectional cohort) the cortical input region.

The bulk GCaMP8m cohort and the intersectional GCaMP6f (MOs-recipient) cohort
must NEVER be pooled. `non_bulk_subjects()` feeds the central never-pool guard
in core/staging.excluded_mice.
"""
from pathlib import Path

COHORTS = {
    "bulk_8m": {
        "indicator": "GCaMP8m",
        "subjects": ["BG_008","BG_009","BG_010","BG_011","BG_013","BG_014",
                     "BG_015","BG_016","BG_017","BG_018","BG_019","BG_020"],
    },
    "intersectional_mos": {
        "indicator": "GCaMP6f",
        "subjects": ["BG_027","BG_028","BG_029","BG_030"],
        # VMS cells receive aMOs; DMS cells receive pMOs.
        "inputs": {"BG_027": "aMOs", "BG_028": "pMOs",
                   "BG_029": "pMOs", "BG_030": "aMOs"},
    },
}

def _norm(subject_id) -> str:
    s = str(subject_id)
    if not s.startswith("BG_") and s.isdigit():
        return f"BG_{s.zfill(3)}"
    return s

def subjects_in(cohort_name):
    return list(COHORTS[cohort_name]["subjects"])

def cohort_of(subject_id):
    s = _norm(subject_id)
    for name, spec in COHORTS.items():
        if s in spec["subjects"]:
            return name
    return None

def indicator_of(subject_id):
    name = cohort_of(subject_id)
    return COHORTS[name]["indicator"] if name else None

def cortical_input(subject_id):
    s = _norm(subject_id)
    return COHORTS["intersectional_mos"].get("inputs", {}).get(s)

def non_bulk_subjects():
    out = []
    for name, spec in COHORTS.items():
        if name != "bulk_8m":
            out.extend(spec["subjects"])
    return out

def load_cohort_sessions(cohort_name, root_dir, max_sessions=None):
    """Load only the sessions belonging to `cohort_name` under root_dir."""
    from visdetect_photom.core import io
    from visdetect_photom.core.session import load_session_from_files
    from visdetect_photom.core.constants import MIN_PHOTOM_CSV_BYTES
    wanted = set(subjects_in(cohort_name))
    files = io.find_all_sessions(str(root_dir), recursive=True,
                                 min_photom_bytes=MIN_PHOTOM_CSV_BYTES)
    sessions, n = [], 0
    for sf in files:
        if max_sessions and n >= max_sessions:
            break
        try:
            sess = load_session_from_files(sf)
        except Exception:
            continue
        if _norm(sess.subject_id) not in wanted:
            continue
        sessions.append(sess)
        n += 1
    return sessions
```

- [ ] **Step 4: Run → PASS** (`py -m pytest tests/core/test_cohort.py -v`).

- [ ] **Step 5: Commit.**

```bash
git add src/visdetect_photom/core/cohort.py tests/core/test_cohort.py
git commit -m "feat(cohort): add cohort registry (bulk_8m / intersectional_mos)"
```

---

## Task 2: Wire 027–030 into genotype + region maps — `constants.py`

**Files:**
- Modify: `src/visdetect_photom/core/constants.py` (`SUBJECT_GENOTYPE` ~lines 97–110; `SUBJECT_ROI_REGION` ~lines 66–71)
- Test: `tests/core/test_cohort.py` (add cases)

**Interfaces:**
- Consumes: `get_genotype`, `get_roi_region`, `get_region_pairs_for_subject` (existing).
- Produces: genotype D1/D2 + region VMS/DMS resolution for 027–030.

- [ ] **Step 1: Add failing tests.**

```python
# append to tests/core/test_cohort.py
from visdetect_photom.analysis.group_utils import get_genotype
from visdetect_photom.core.constants import get_roi_region
from visdetect_photom.core.qc import get_region_pairs_for_subject

def test_cohort_genotype_and_region_maps():
    assert get_genotype("BG_027") == "D1" and get_genotype("BG_028") == "D1"
    assert get_genotype("BG_029") == "D2" and get_genotype("BG_030") == "D2"
    assert get_roi_region("G0", "BG_027") == "VMS_L"   # 027/030 = VMS
    assert get_roi_region("G2", "BG_030") == "VMS_R"
    assert get_roi_region("G0", "BG_028") == "DMS_L"   # 028/029 = DMS
    assert get_region_pairs_for_subject("BG_027") == {"VMS": ("G0", "G2")}
    assert get_region_pairs_for_subject("BG_029") == {"DMS": ("G0", "G2")}
```

- [ ] **Step 2: Run → FAIL** (`get_genotype('BG_027')` returns "Unknown").

- [ ] **Step 3: Edit `SUBJECT_GENOTYPE`** — add after `'BG_020': 'D1',`:

```python
    # Intersectional MOs-recipient cohort (GCaMP6f) — SEPARATE cohort, never
    # pooled with bulk (see core/cohort.py + excluded_mice). Cell type only.
    'BG_027': 'D1',
    'BG_028': 'D1',
    'BG_029': 'D2',
    'BG_030': 'D2',
```

- [ ] **Step 4: Edit `SUBJECT_ROI_REGION`** — add after the `'BG_011'` line:

```python
    # Intersectional cohort: 027/030 record VMS; 028/029 record DMS (G0/G2).
    'BG_027': {'G0': 'VMS_L', 'G2': 'VMS_R'},
    'BG_030': {'G0': 'VMS_L', 'G2': 'VMS_R'},
    'BG_028': {'G0': 'DMS_L', 'G2': 'DMS_R'},
    'BG_029': {'G0': 'DMS_L', 'G2': 'DMS_R'},
```

- [ ] **Step 5: Run → PASS** (`py -m pytest tests/core/test_cohort.py -v`).

- [ ] **Step 6: Commit.**

```bash
git add src/visdetect_photom/core/constants.py tests/core/test_cohort.py
git commit -m "feat(cohort): wire BG_027-030 genotype + region maps (separate GCaMP6f cohort)"
```

---

## Task 3: Enforce never-pool centrally — `excluded_mice`

**Files:**
- Modify: `src/visdetect_photom/core/staging.py` (`excluded_mice`, ~lines 40–48)
- Test: `tests/core/test_excluded_cohort.py`

**Interfaces:**
- Consumes: `cohort.non_bulk_subjects()`.
- Produces: `excluded_mice(manifest)` returns its staging-Excluded set **unioned with** the non-bulk cohort, even when `manifest is None`.

- [ ] **Step 1: Write the failing test.**

```python
# tests/core/test_excluded_cohort.py
from visdetect_photom.core.staging import excluded_mice

def test_excluded_mice_unions_non_bulk_cohort_even_without_manifest():
    excl = excluded_mice(None)
    for s in ("BG_027", "BG_028", "BG_029", "BG_030"):
        assert s in excl, "intersectional cohort must be excluded from the bulk default"
    assert "BG_013" not in excl  # a bulk mouse is not excluded by default
```

- [ ] **Step 2: Run → FAIL** (`excluded_mice(None)` returns `set()`).

- [ ] **Step 3: Edit `excluded_mice`.**

```python
def excluded_mice(manifest) -> set:
    """Subjects (BG_0XX) to skip in the BULK default analysis.

    Includes (a) subjects whose every staged session is 'Excluded', AND
    (b) every subject in a non-bulk cohort (e.g. the intersectional
    MOs-recipient GCaMP6f cohort), which must NEVER be pooled with the bulk
    GCaMP8m mice. (b) applies even when no manifest is present.
    """
    from visdetect_photom.core.cohort import non_bulk_subjects
    excl = set(non_bulk_subjects())
    if manifest is None or "stage" not in manifest.columns:
        return excl
    for subj, grp in manifest.groupby("subject_id"):
        if (grp["stage"] == "Excluded").all():
            excl.add(_norm_subject(subj))
    return excl
```

- [ ] **Step 4: Run → PASS**; then full suite `py -m pytest -q` (bulk scripts now also skip the cohort — confirm no regressions).

- [ ] **Step 5: Commit.**

```bash
git add src/visdetect_photom/core/staging.py tests/core/test_excluded_cohort.py
git commit -m "fix(cohort): never-pool guard — excluded_mice unions non-bulk cohort"
```

---

## Task 4: Session-unit aggregator — `summarize_sessions_by_cell`

**Files:**
- Modify: `src/visdetect_photom/core/cohort.py`
- Test: `tests/core/test_cohort.py` (add cases)

**Interfaces:**
- Consumes: `group_statistics.bootstrap_ci`.
- Produces: `summarize_sessions_by_cell(per_session_df, value_cols=("delta","auroc"), cell_keys=("subject_id","genotype","region")) -> pd.DataFrame` — one row per cell with `n_sessions` and, per value col, `<col>_mean`, `<col>_ci_lo`, `<col>_ci_hi` (session-bootstrap, seed 42).

- [ ] **Step 1: Add the failing test.**

```python
# append to tests/core/test_cohort.py
import numpy as np, pandas as pd
from visdetect_photom.core.cohort import summarize_sessions_by_cell

def test_summarize_sessions_by_cell_bootstraps_over_sessions():
    rows = [{"subject_id":"BG_029","genotype":"D2","region":"DMS",
             "session_id":f"s{i}","auroc":0.6+0.01*i,"delta":0.1*i} for i in range(20)]
    out = summarize_sessions_by_cell(pd.DataFrame(rows))
    r = out.iloc[0]
    assert r["n_sessions"] == 20
    assert r["auroc_mean"] == pytest_approx(np.mean([0.6+0.01*i for i in range(20)]))
    assert r["auroc_ci_lo"] < r["auroc_mean"] < r["auroc_ci_hi"]

def pytest_approx(x):  # tiny local helper to avoid extra import noise
    import pytest
    return pytest.approx(x, rel=1e-6)
```

- [ ] **Step 2: Run → FAIL** (function missing).

- [ ] **Step 3: Implement (append to `cohort.py`).**

```python
def summarize_sessions_by_cell(per_session_df, value_cols=("delta", "auroc"),
                               cell_keys=("subject_id", "genotype", "region")):
    """Per cell: bootstrap CI of each value col over that cell's per-session values.

    This is the within-animal session-unit summary for the n=1-mouse/cell cohort.
    """
    import numpy as np
    import pandas as pd
    from visdetect_photom.analysis.group_statistics import bootstrap_ci
    if per_session_df is None or per_session_df.empty:
        return pd.DataFrame()
    cell_keys = list(cell_keys)
    out = []
    for key, g in per_session_df.groupby(cell_keys):
        row = dict(zip(cell_keys, key if isinstance(key, tuple) else (key,)))
        row["n_sessions"] = int(len(g))
        for col in value_cols:
            vals = g[col].to_numpy(dtype=float)
            ci = bootstrap_ci(vals)
            row[f"{col}_mean"] = ci["observed"]
            row[f"{col}_ci_lo"] = ci["ci_lo"]
            row[f"{col}_ci_hi"] = ci["ci_hi"]
        out.append(row)
    return pd.DataFrame(out)
```

- [ ] **Step 4: Run → PASS.**

- [ ] **Step 5: Commit.**

```bash
git add src/visdetect_photom/core/cohort.py tests/core/test_cohort.py
git commit -m "feat(cohort): session-unit aggregator (bootstrap over sessions per cell)"
```

---

## Task 5: Cross-cohort matcher + rank-based guard — `cohort.py`

**Files:**
- Modify: `src/visdetect_photom/core/cohort.py`
- Test: `tests/core/test_cohort.py` (add cases)

**Interfaces:**
- Produces:
  - `RANK_BASED_METRICS = {"auroc","sign","rank_biserial_r","peak_latency","onset_latency"}`
  - `assert_rank_based(metric_name)` — raises `ValueError` if `metric_name` not in `RANK_BASED_METRICS` (guards cross-indicator magnitude comparison).
  - `match_cohort_cells(genotype, region) -> dict` with keys `intersectional` (the one cohort subject) and `bulk` (list of matched bulk subjects), by genotype × region.

- [ ] **Step 1: Add the failing test.**

```python
# append to tests/core/test_cohort.py
import pytest
from visdetect_photom.core.cohort import match_cohort_cells, assert_rank_based

def test_match_cohort_cells_pairs_by_genotype_region():
    m = match_cohort_cells("D2", "DMS")
    assert m["intersectional"] == ["BG_029"]
    assert set(m["bulk"]) == {"BG_016", "BG_018", "BG_019"}
    m2 = match_cohort_cells("D1", "VMS")
    assert m2["intersectional"] == ["BG_027"]
    assert set(m2["bulk"]) == {"BG_008", "BG_009"}

def test_assert_rank_based_refuses_magnitude():
    assert_rank_based("auroc")            # ok
    with pytest.raises(ValueError):
        assert_rank_based("signed_auc")   # a magnitude metric -> refused across indicators
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement (append to `cohort.py`).**

```python
RANK_BASED_METRICS = {"auroc", "sign", "rank_biserial_r", "peak_latency", "onset_latency"}

def assert_rank_based(metric_name):
    """Guard: cross-cohort (6f vs 8m) comparison may use only indicator-invariant
    quantities. Raises ValueError for magnitude metrics (e.g. signed_auc)."""
    if metric_name not in RANK_BASED_METRICS:
        raise ValueError(
            f"{metric_name!r} is a magnitude metric; cross-indicator comparison "
            f"is restricted to {sorted(RANK_BASED_METRICS)}")

def match_cohort_cells(genotype, region):
    """Pair the one intersectional cell with the matched bulk cell(s)
    (same genotype x region). Region compared on base name (DMS/VMS/VLS)."""
    from visdetect_photom.analysis.group_utils import get_genotype
    from visdetect_photom.core.constants import get_roi_region

    def _base_region_of(subject):
        # subjects here are G0/G2 cells; base region from G0 mapping
        r = get_roi_region("G0", subject)
        return r.rsplit("_", 1)[0] if r else None

    def _members(cohort_name):
        return [s for s in subjects_in(cohort_name)
                if get_genotype(s) == genotype and _base_region_of(s) == region]

    return {"intersectional": _members("intersectional_mos"),
            "bulk": _members("bulk_8m")}
```

- [ ] **Step 4: Run → PASS.**

- [ ] **Step 5: Commit.**

```bash
git add src/visdetect_photom/core/cohort.py tests/core/test_cohort.py
git commit -m "feat(cohort): cross-cohort matcher + rank-based-only guard"
```

---

## Task 6: Session-level delta/AUROC — `compute_session_delta_and_auroc`

**Files:**
- Modify: `src/visdetect_photom/analysis/suppression.py`
- Test: `tests/analysis/test_suppression_session_unit.py`

**Interfaces:**
- Consumes: `auroc_score`, `MIN_TRIALS_PER_GROUP` (already in suppression.py).
- Produces: `compute_session_delta_and_auroc(per_trial_df, min_n=MIN_TRIALS_PER_GROUP) -> pd.DataFrame` — like `compute_delta_and_auroc` but groups by `(subject_id, genotype, region, session_id)`, one row per session with `delta`, `auroc`, `n_lick`, `n_withhold`.

- [ ] **Step 1: Write the failing test.**

```python
# tests/analysis/test_suppression_session_unit.py
import numpy as np, pandas as pd
from visdetect_photom.analysis.suppression import compute_session_delta_and_auroc

def _rows(session_id, lick, withhold):
    r = [{"subject_id":"BG_029","genotype":"D2","region":"DMS","track":"behavioral_fa",
          "scheme":"scheme1","group":"lick","trial_index":i,"scalar":v,"session_id":session_id}
         for i,v in enumerate(lick)]
    r += [{"subject_id":"BG_029","genotype":"D2","region":"DMS","track":"behavioral_fa",
           "scheme":"scheme1","group":"withhold","trial_index":100+i,"scalar":v,"session_id":session_id}
          for i,v in enumerate(withhold)]
    return r

def test_session_unit_emits_one_row_per_session():
    rows = _rows("sA", np.zeros(10), np.ones(10)) + _rows("sB", np.ones(10), np.zeros(10))
    out = compute_session_delta_and_auroc(pd.DataFrame(rows))
    assert set(out["session_id"]) == {"sA", "sB"}
    a = out[out["session_id"] == "sA"].iloc[0]
    assert a["delta"] > 0 and a["auroc"] > 0.5      # withhold>lick in sA
    b = out[out["session_id"] == "sB"].iloc[0]
    assert b["delta"] < 0 and b["auroc"] < 0.5      # withhold<lick in sB
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement (append to `suppression.py`, mirroring `compute_delta_and_auroc` with `session_id` in the groupby).**

```python
def compute_session_delta_and_auroc(per_trial_df, min_n=MIN_TRIALS_PER_GROUP):
    """Per (subject_id, genotype, region, session_id): delta = mean(withhold) -
    mean(lick) and AUROC of the scalar discriminating withhold (positive) from
    lick. Session is the unit (for the n=1-mouse/cell intersectional cohort).
    Sessions with < min_n finite scalars in either group are dropped.

    Caller must pass a DataFrame filtered to a single (track, scheme).
    """
    if per_trial_df.empty:
        return pd.DataFrame()
    df = per_trial_df[per_trial_df["group"].isin(["lick", "withhold"])].copy()
    df = df[np.isfinite(df["scalar"].astype(float))]
    out = []
    for (subj, geno, region, sid), g in df.groupby(
            ["subject_id", "genotype", "region", "session_id"]):
        lick = g[g["group"] == "lick"]["scalar"].to_numpy(dtype=float)
        wh = g[g["group"] == "withhold"]["scalar"].to_numpy(dtype=float)
        if lick.size < min_n or wh.size < min_n:
            continue
        scores = np.concatenate([wh, lick])
        labels = np.concatenate([np.ones(wh.size), np.zeros(lick.size)])
        out.append({"subject_id": subj, "genotype": geno, "region": region,
                    "session_id": sid, "n_lick": int(lick.size),
                    "n_withhold": int(wh.size),
                    "delta": float(np.mean(wh) - np.mean(lick)),
                    "auroc": auroc_score(scores, labels)})
    return pd.DataFrame(out)
```

- [ ] **Step 4: Run → PASS.**

- [ ] **Step 5: Commit.**

```bash
git add src/visdetect_photom/analysis/suppression.py tests/analysis/test_suppression_session_unit.py
git commit -m "feat(suppression): session-unit delta/AUROC for n=1-mouse/cell cohort"
```

---

## Task 7: Reproducible staging script — `stage_intersectional_cohort.py`

**Files:**
- Create: `scripts/data_management/stage_intersectional_cohort.py`
- Test: `tests/data_management/test_stage_intersectional.py`

**Interfaces:**
- Produces:
  - `normalize_filename(name, subject) -> str` — maps a stale `BG_027__photom_*` / `BG_027__photom_IO_*` (and single→double underscore) to the correct subject prefix; leaves correctly-named files unchanged.
  - `stage(subject, ceph_dir, dest_dir, *, dry_run=True, min_bytes=0) -> dict` — idempotent copy of top-level csv+json with name normalization; returns a summary dict (`copied`, `renamed`, `skipped`).

Pure-function `normalize_filename` is unit-tested; `stage` is tested for idempotency on temp dirs. The ceph copy itself is run manually by the user (it touches the X: mount).

- [ ] **Step 1: Write the failing test.**

```python
# tests/data_management/test_stage_intersectional.py
import importlib.util, os
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "stage_intersectional_cohort",
    Path(__file__).resolve().parents[2] / "scripts" / "data_management" / "stage_intersectional_cohort.py")
stage_mod = importlib.util.module_from_spec(_SPEC); _SPEC.loader.exec_module(stage_mod)

def test_normalize_filename_fixes_stale_subject_and_underscore():
    # stale BG_027 photom inside a BG_030 folder -> corrected to BG_030
    assert stage_mod.normalize_filename("BG_027__photom_2024-12-04T12_01_00.csv", "BG_030") \
        == "BG_030__photom_2024-12-04T12_01_00.csv"
    assert stage_mod.normalize_filename("BG_027__photom_IO_2024-12-04T12_01_00.csv", "BG_030") \
        == "BG_030__photom_IO_2024-12-04T12_01_00.csv"
    # single underscore -> double underscore, correct subject untouched otherwise
    assert stage_mod.normalize_filename("BG_030_trials.json", "BG_030") == "BG_030__trials.json"
    # already-correct name is unchanged
    assert stage_mod.normalize_filename("BG_030__session_settings.json", "BG_030") \
        == "BG_030__session_settings.json"

def test_stage_is_idempotent(tmp_path):
    src = tmp_path / "ceph" / "BG_030"; src.mkdir(parents=True)
    (src / "BG_027__photom_2024-12-04T12_01_00.csv").write_text("x" * 100)
    (src / "BG_030__trials.json").write_text("{}")
    dest = tmp_path / "dest" / "BG_030"
    r1 = stage_mod.stage("BG_030", src, dest, dry_run=False)
    r2 = stage_mod.stage("BG_030", src, dest, dry_run=False)
    assert (dest / "BG_030__photom_2024-12-04T12_01_00.csv").exists()
    assert r2["copied"] == 0  # second run copies nothing new
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement.**

```python
# scripts/data_management/stage_intersectional_cohort.py
"""Idempotent staging of the intersectional GCaMP6f cohort (BG_027-030) from
ceph to local photom_data/intrsct_GCaMP6f/. Copies ONLY top-level csv+json,
normalizes the documented stale-subject mislabel (BG_027-named photom inside
BG_029/030 folders) and single->double underscores. NEVER writes to ceph.

Manual run (X: mount):
    py scripts/data_management/stage_intersectional_cohort.py --subject BG_030 \
        --ceph "X:/public/projects/BeJG_20230130_VisDetect/wIntersectGCaMP6F/BG_030" \
        --dest photom_data/intrsct_GCaMP6f/BG_030 --apply
"""
import argparse
import re
import shutil
from pathlib import Path

_STALE_PREFIX = "BG_027"   # the documented stale acquisition subject id

def normalize_filename(name, subject):
    """Return the corrected local filename for a ceph file belonging to `subject`."""
    # Fix stale subject prefix on photom / photom_IO csvs.
    if name.startswith(_STALE_PREFIX) and subject != _STALE_PREFIX:
        name = subject + name[len(_STALE_PREFIX):]
    # Normalize single underscore after subject to double (BG_030_x -> BG_030__x),
    # but do not create triple underscores.
    name = re.sub(rf"^{re.escape(subject)}_(?!_)", f"{subject}__", name)
    return name

def _iter_top_level(ceph_dir):
    for p in sorted(Path(ceph_dir).iterdir()):
        if p.is_file() and p.suffix.lower() in (".csv", ".json"):
            yield p

def stage(subject, ceph_dir, dest_dir, *, dry_run=True, min_bytes=0):
    dest = Path(dest_dir); 
    summary = {"copied": 0, "renamed": 0, "skipped": 0}
    if not dry_run:
        dest.mkdir(parents=True, exist_ok=True)
    for src in _iter_top_level(ceph_dir):
        if src.stat().st_size < min_bytes:
            summary["skipped"] += 1
            continue
        new_name = normalize_filename(src.name, subject)
        if new_name != src.name:
            summary["renamed"] += 1
        target = dest / new_name
        if target.exists() and target.stat().st_size == src.stat().st_size:
            summary["skipped"] += 1
            continue
        if not dry_run:
            shutil.copy2(src, target)
        summary["copied"] += 1
    return summary

def main():
    ap = argparse.ArgumentParser(description="Stage intersectional cohort from ceph")
    ap.add_argument("--subject", required=True)
    ap.add_argument("--ceph", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--min-bytes", type=int, default=0)
    ap.add_argument("--apply", action="store_true", help="actually copy (default: dry-run)")
    args = ap.parse_args()
    res = stage(args.subject, args.ceph, args.dest,
                dry_run=not args.apply, min_bytes=args.min_bytes)
    print(f"{args.subject}: {res} ({'APPLIED' if args.apply else 'DRY-RUN'})")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run → PASS** (`py -m pytest tests/data_management/test_stage_intersectional.py -v`).

- [ ] **Step 5: Commit.**

```bash
git add scripts/data_management/stage_intersectional_cohort.py tests/data_management/test_stage_intersectional.py
git commit -m "feat(cohort): reproducible idempotent ceph staging with mislabel fix"
```

---

## Task 8: C1 cohort script — `c1_cohort_suppression.py`

**Files:**
- Create: `scripts/analysis/intersectional/c1_cohort_suppression.py`
- Test: `tests/scripts/test_cohort_smoke.py` (C1 case)

**Interfaces:**
- Consumes: `cohort.load_cohort_sessions`, `cohort.summarize_sessions_by_cell`; `suppression.build_suppression_dataset`, `compute_delta_and_auroc`, `compute_session_delta_and_auroc`, `assign_proficiency_bins`; `PooledStateProvider`.
- Produces CSVs in `FIGURES/intersectional_mos/`: `cohort_c1_session_scalars.csv` (per session), `cohort_c1_cell_summary.csv` (per cell: pooled AUROC/delta + session-bootstrap CIs), `cohort_c1_qualifying_n.csv`; PNGs: `cohort_c1_brake_2x2.png`, companion PETHs.

- [ ] **Step 1: Write the failing smoke test.**

```python
# tests/scripts/test_cohort_smoke.py
import os, subprocess, sys, pytest
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(REPO, "photom_data", "intrsct_GCaMP6f")
C1 = os.path.join(REPO, "scripts", "analysis", "intersectional", "c1_cohort_suppression.py")

@pytest.mark.skipif(not os.path.isdir(DATA), reason="cohort data not present")
def test_c1_cohort_runs(tmp_path):
    out = tmp_path / "out"
    proc = subprocess.run([sys.executable, C1, "--max_sessions", "4",
                           "--root_dir", DATA, "--output_dir", str(out)],
                          cwd=REPO, capture_output=True, text=True, timeout=1200)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert (out / "cohort_c1_session_scalars.csv").exists()
    assert (out / "cohort_c1_cell_summary.csv").exists()
```

- [ ] **Step 2: Run → FAIL** (script missing). (If cohort data absent locally, the test skips — then validate manually per Step 4.)

- [ ] **Step 3: Implement.**

```python
# scripts/analysis/intersectional/c1_cohort_suppression.py
"""C1 for the intersectional MOs-recipient cohort (BG_027-030), in a 2x2.

Session is the statistical unit (n=1 mouse/cell): per-cell pooled AUROC/delta
(compute_delta_and_auroc) PLUS per-session distribution with session-bootstrap
CIs (summarize_sessions_by_cell). Trial-pooled companion PETHs are illustrative.
NEVER pooled with bulk-8m. D1 vs D2 reported as a cell-level sign contrast only.
"""
import argparse, logging, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns

_repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo_root / "src"))

from visdetect_photom.core import cohort
from visdetect_photom.core.constants import GENOTYPE_COLORS
from visdetect_photom.analysis.state_provider import PooledStateProvider
from visdetect_photom.analysis.suppression import (
    build_suppression_dataset, compute_delta_and_auroc,
    compute_session_delta_and_auroc, assign_proficiency_bins,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
TRACKS = ["behavioral_fa", "sdt_fa"]
SCHEMES = ["scheme1", "scheme3"]

def _plot_brake_2x2(cell_summary, out_dir):
    """2x2 grid: rows = genotype (D1/D2), cols = region (DMS/VMS); bar = AUROC
    (behavioral_fa/scheme3) with session-bootstrap CI; chance line at 0.5."""
    sub = cell_summary[(cell_summary["track"] == "behavioral_fa")
                       & (cell_summary["scheme"] == "scheme3")]
    genos, regions = ["D1", "D2"], ["DMS", "VMS"]
    fig, axes = plt.subplots(2, 2, figsize=(8, 7), squeeze=False)
    fig.suptitle("C1 brake (intersectional MOs-recipient) — withhold-vs-FA AUROC\n"
                 "(session-unit; n=1 mouse/cell, within-animal)", fontsize=11)
    for ri, geno in enumerate(genos):
        for ci, region in enumerate(regions):
            ax = axes[ri][ci]
            row = sub[(sub["genotype"] == geno) & (sub["region"] == region)]
            if len(row):
                r = row.iloc[0]
                lo = r["auroc_mean"] - r["auroc_ci_lo"]; hi = r["auroc_ci_hi"] - r["auroc_mean"]
                ax.bar([0], [r["auroc_mean"]], color=GENOTYPE_COLORS[geno],
                       yerr=[[max(lo,0)], [max(hi,0)]], capsize=4)
                ax.set_title(f"{geno} · {region} ({r['subject_id']}, {int(r['n_sessions'])} sess)", fontsize=8)
            else:
                ax.set_title(f"{geno} · {region} (no data)", fontsize=8)
            ax.axhline(0.5, color="k", ls="--", lw=0.8)
            ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_ylabel("AUROC", fontsize=8)
            sns.despine(ax=ax)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "cohort_c1_brake_2x2.png"
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    logging.info(f"Saved {p}")

def main():
    ap = argparse.ArgumentParser(description="C1 — intersectional MOs-recipient cohort")
    rr = Path(__file__).resolve().parents[3]
    ap.add_argument("--root_dir", default=str(rr / "photom_data" / "intrsct_GCaMP6f"))
    ap.add_argument("--output_dir", default=str(rr / "FIGURES" / "intersectional_mos"))
    ap.add_argument("--no-qc", action="store_true", default=False)
    ap.add_argument("--max_sessions", type=int, default=None)
    args = ap.parse_args()
    use_qc = not args.no_qc
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    sessions = cohort.load_cohort_sessions("intersectional_mos", args.root_dir,
                                           max_sessions=args.max_sessions)
    if not sessions:
        logging.error("No cohort sessions loaded."); sys.exit(1)
    logging.info(f"Loaded {len(sessions)} cohort sessions.")
    prof_bins = assign_proficiency_bins(sessions, manifest=None)  # cohort not staged -> date split

    sp, keep = PooledStateProvider(), ["All"]
    all_sess_scalars, cell_rows, qual_rows = [], [], []
    for track in TRACKS:
        for scheme in SCHEMES:
            df = build_suppression_dataset(sessions, track=track, scheme=scheme,
                                           use_qc=use_qc, state_provider=sp, keep_states=keep)
            if df.empty:
                continue
            df["prof_bin"] = df["session_id"].map(prof_bins)
            df["track"], df["scheme"] = track, scheme

            per_session = compute_session_delta_and_auroc(df)
            if not per_session.empty:
                per_session["track"], per_session["scheme"] = track, scheme
                all_sess_scalars.append(per_session)
                summ = cohort.summarize_sessions_by_cell(per_session)
                # pooled per-cell point estimate (all trials), one row/cell at n=1/cell
                pooled = compute_delta_and_auroc(df)[
                    ["subject_id", "genotype", "region", "delta", "auroc"]].rename(
                    columns={"delta": "delta_pooled", "auroc": "auroc_pooled"})
                merged = summ.merge(pooled, on=["subject_id", "genotype", "region"], how="left")
                merged["track"], merged["scheme"] = track, scheme
                cell_rows.append(merged)

            g = df.copy(); g["finite"] = np.isfinite(g["scalar"].astype(float))
            qn = (g.groupby(["track","scheme","region","genotype","group"])["finite"]
                    .agg(n_total="size", n_finite="sum").reset_index())
            qual_rows.append(qn)

    if not all_sess_scalars:
        logging.error("No waiting-period scalars extracted."); sys.exit(1)
    pd.concat(all_sess_scalars, ignore_index=True).to_csv(
        out_dir / "cohort_c1_session_scalars.csv", index=False)
    cell_summary = pd.concat(cell_rows, ignore_index=True)
    cell_summary.to_csv(out_dir / "cohort_c1_cell_summary.csv", index=False)
    pd.concat(qual_rows, ignore_index=True).to_csv(
        out_dir / "cohort_c1_qualifying_n.csv", index=False)
    _plot_brake_2x2(cell_summary, out_dir)
    logging.info("Done.")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run → PASS** (or, if data absent, run manually: `py scripts/analysis/intersectional/c1_cohort_suppression.py --max_sessions 4` and confirm the two CSVs + the 2×2 PNG appear under `FIGURES/intersectional_mos/`).

- [ ] **Step 5: Commit.**

```bash
git add scripts/analysis/intersectional/c1_cohort_suppression.py tests/scripts/test_cohort_smoke.py
git commit -m "feat(cohort): C1 brake in 2x2 (session-unit + bulk-safe) for BG_027-030"
```

---

## Task 9: C2 cohort script — `c2_cohort_geometry.py`

**Files:**
- Create: `scripts/analysis/intersectional/c2_cohort_geometry.py`
- Test: `tests/scripts/test_cohort_smoke.py` (C2 case)

**Interfaces:**
- Consumes: `cohort.load_cohort_sessions`, `cohort.summarize_sessions_by_cell`; `geometry.compute_geometry_metrics_for_session` (per-session rows; the script injects `session_id`); `PooledStateProvider`.
- Produces: `cohort_c2_session_metrics.csv` (per session, with `session_id`), `cohort_c2_cell_summary.csv` (per cell × epoch: session-bootstrap CI of `signed_auc`, `peak_latency`, `onset_latency`), `cohort_c2_geometry_2x2.png`.

- [ ] **Step 1: Add the failing smoke case.**

```python
# append to tests/scripts/test_cohort_smoke.py
C2 = os.path.join(REPO, "scripts", "analysis", "intersectional", "c2_cohort_geometry.py")

@pytest.mark.skipif(not os.path.isdir(DATA), reason="cohort data not present")
def test_c2_cohort_runs(tmp_path):
    out = tmp_path / "out"
    proc = subprocess.run([sys.executable, C2, "--max_sessions", "4",
                           "--root_dir", DATA, "--output_dir", str(out)],
                          cwd=REPO, capture_output=True, text=True, timeout=1200)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert (out / "cohort_c2_cell_summary.csv").exists()
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement.** (`compute_geometry_metrics_for_session` returns `(rows, traces, time_axis)`; rows lack `session_id`, so inject it. Use pooled epochs only — `change_size` NaN — for the cell summary.)

```python
# scripts/analysis/intersectional/c2_cohort_geometry.py
"""C2 response geometry for the intersectional MOs-recipient cohort (BG_027-030).

Session is the statistical unit: per-session geometry metrics (signed_auc +
latencies), summarized per cell with session-bootstrap CIs, in a 2x2. NEVER
pooled with bulk-8m. D1 vs D2 = cell-level sign contrast only.
"""
import argparse, logging, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns

_repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo_root / "src"))

from visdetect_photom.core import cohort
from visdetect_photom.core.constants import GENOTYPE_COLORS
from visdetect_photom.analysis.state_provider import PooledStateProvider
from visdetect_photom.analysis.geometry import compute_geometry_metrics_for_session

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
SUMMARY_EPOCHS = ["change_hit", "change_miss", "hit_lick", "fa_lick"]
SUMMARY_VALUES = ("signed_auc", "peak_latency", "onset_latency")

def _plot_geometry_2x2(cell_summary, out_dir, value="signed_auc"):
    genos, regions = ["D1", "D2"], ["DMS", "VMS"]
    epochs = SUMMARY_EPOCHS
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), squeeze=False)
    fig.suptitle(f"C2 geometry (intersectional MOs-recipient) — {value} by epoch\n"
                 "(session-unit; n=1 mouse/cell, within-animal)", fontsize=11)
    x = np.arange(len(epochs))
    for ri, geno in enumerate(genos):
        for ci, region in enumerate(regions):
            ax = axes[ri][ci]
            sub = cell_summary[(cell_summary["genotype"] == geno)
                               & (cell_summary["region"] == region)].set_index("epoch")
            means = [sub.loc[e, f"{value}_mean"] if e in sub.index else np.nan for e in epochs]
            ax.bar(x, means, color=GENOTYPE_COLORS[geno])
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xticks(x); ax.set_xticklabels(epochs, rotation=45, ha="right", fontsize=7)
            ax.set_title(f"{geno} · {region}", fontsize=9); ax.set_ylabel(value, fontsize=8)
            sns.despine(ax=ax)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "cohort_c2_geometry_2x2.png"
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    logging.info(f"Saved {p}")

def main():
    ap = argparse.ArgumentParser(description="C2 — intersectional MOs-recipient cohort")
    rr = Path(__file__).resolve().parents[3]
    ap.add_argument("--root_dir", default=str(rr / "photom_data" / "intrsct_GCaMP6f"))
    ap.add_argument("--output_dir", default=str(rr / "FIGURES" / "intersectional_mos"))
    ap.add_argument("--no-qc", action="store_true", default=False)
    ap.add_argument("--max_sessions", type=int, default=None)
    args = ap.parse_args()
    use_qc = not args.no_qc
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    sessions = cohort.load_cohort_sessions("intersectional_mos", args.root_dir,
                                           max_sessions=args.max_sessions)
    if not sessions:
        logging.error("No cohort sessions loaded."); sys.exit(1)
    logging.info(f"Loaded {len(sessions)} cohort sessions.")

    sp, keep = PooledStateProvider(), ["All"]
    rows = []
    for sess in sessions:
        srows, _, _ = compute_geometry_metrics_for_session(
            sess, use_qc=use_qc, state_provider=sp, keep_states=keep)
        for r in srows:
            r["session_id"] = sess.session_id
            rows.append(r)
    if not rows:
        logging.error("No geometry metrics extracted."); sys.exit(1)
    per_session = pd.DataFrame(rows)
    per_session.to_csv(out_dir / "cohort_c2_session_metrics.csv", index=False)

    # pooled epochs only (change_size NaN); summarize per cell x epoch over sessions
    pooled = per_session[per_session["change_size"].isna()
                         & per_session["epoch"].isin(SUMMARY_EPOCHS)]
    cell_rows = []
    for epoch, g in pooled.groupby("epoch"):
        summ = cohort.summarize_sessions_by_cell(g, value_cols=SUMMARY_VALUES)
        summ["epoch"] = epoch
        cell_rows.append(summ)
    cell_summary = pd.concat(cell_rows, ignore_index=True) if cell_rows else pd.DataFrame()
    cell_summary.to_csv(out_dir / "cohort_c2_cell_summary.csv", index=False)
    if not cell_summary.empty:
        _plot_geometry_2x2(cell_summary, out_dir)
    logging.info("Done.")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run → PASS** (or manual run; confirm `cohort_c2_cell_summary.csv` + the 2×2 PNG).

- [ ] **Step 5: Commit.**

```bash
git add scripts/analysis/intersectional/c2_cohort_geometry.py tests/scripts/test_cohort_smoke.py
git commit -m "feat(cohort): C2 geometry in 2x2 (session-unit) for BG_027-030"
```

---

## Task 10: QC calibration report — `cohort_qc_report.py`

**Files:**
- Create: `scripts/analysis/intersectional/cohort_qc_report.py`
- Test: `tests/scripts/test_cohort_smoke.py` (QC case)

**Interfaces:**
- Consumes: `cohort.load_cohort_sessions`; `qc.compute_session_roi_qc`.
- Produces: `cohort_qc_report.csv` — per (subject, session, roi): the QC metrics + a `passed` flag, so 6f pass-rates and metric distributions per cell are visible. (Same 8m-tuned thresholds; this only REPORTS — no threshold change.)

- [ ] **Step 1: Add the failing smoke case.**

```python
# append to tests/scripts/test_cohort_smoke.py
QC = os.path.join(REPO, "scripts", "analysis", "intersectional", "cohort_qc_report.py")

@pytest.mark.skipif(not os.path.isdir(DATA), reason="cohort data not present")
def test_cohort_qc_report_runs(tmp_path):
    out = tmp_path / "out"
    proc = subprocess.run([sys.executable, QC, "--max_sessions", "4",
                           "--root_dir", DATA, "--output_dir", str(out)],
                          cwd=REPO, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert (out / "cohort_qc_report.csv").exists()
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement.** (`compute_session_roi_qc(session)` returns `{roi: {metric: value, ...}}`; flatten to rows. Include a `passed` key if present in the returned dict; otherwise omit.)

```python
# scripts/analysis/intersectional/cohort_qc_report.py
"""6f QC calibration report for the intersectional cohort (BG_027-030).

Applies the SAME 8m-tuned QC (compute_session_roi_qc) and only REPORTS the
per-ROI metrics + pass flag, so 6f pass-rates and metric distributions per cell
are visible. An indicator-aware threshold is introduced ONLY if these data
demand it, and then documented in constants — never silently.
"""
import argparse, logging, sys
from pathlib import Path
import pandas as pd

_repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo_root / "src"))

from visdetect_photom.core import cohort
from visdetect_photom.core.qc import compute_session_roi_qc

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def main():
    ap = argparse.ArgumentParser(description="Intersectional cohort QC report")
    rr = Path(__file__).resolve().parents[3]
    ap.add_argument("--root_dir", default=str(rr / "photom_data" / "intrsct_GCaMP6f"))
    ap.add_argument("--output_dir", default=str(rr / "FIGURES" / "intersectional_mos"))
    ap.add_argument("--max_sessions", type=int, default=None)
    args = ap.parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    sessions = cohort.load_cohort_sessions("intersectional_mos", args.root_dir,
                                           max_sessions=args.max_sessions)
    if not sessions:
        logging.error("No cohort sessions loaded."); sys.exit(1)
    rows = []
    for sess in sessions:
        qc = compute_session_roi_qc(sess)
        for roi, metrics in qc.items():
            row = {"subject_id": f"BG_{str(sess.subject_id).zfill(3)}"
                   if not str(sess.subject_id).startswith("BG_") else str(sess.subject_id),
                   "session_id": sess.session_id, "roi": roi}
            row.update({k: v for k, v in metrics.items()})
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "cohort_qc_report.csv", index=False)
    logging.info(f"Wrote {out_dir / 'cohort_qc_report.csv'} ({len(df)} roi-rows).")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run → PASS** (or manual).

- [ ] **Step 5: Commit.**

```bash
git add scripts/analysis/intersectional/cohort_qc_report.py tests/scripts/test_cohort_smoke.py
git commit -m "feat(cohort): 6f QC calibration report (same thresholds, report-only)"
```

---

## Task 11: Cross-cohort comparison — `cohort_cross_compare.py`

**Files:**
- Create: `scripts/analysis/intersectional/cohort_cross_compare.py`
- Test: `tests/scripts/test_cross_compare.py`

**Interfaces:**
- Consumes: `cohort.match_cohort_cells`, `cohort.assert_rank_based`; the cohort `cohort_c1_cell_summary.csv` and the bulk `FIGURES/C1_fa_suppression/c1_auroc_stats.csv` (rank-based fields only).
- Produces: `cohort_cross_compare.csv` — per (genotype, region): cohort AUROC vs bulk AUROC (rank-based), with `delta_auroc = cohort - bulk` and the matched subject ids; `cohort_cross_compare.png`.

The function `build_cross_compare(cohort_c1_csv, bulk_auroc_csv) -> pd.DataFrame` is unit-tested with synthetic CSVs (no real data needed); the CLI wraps it.

- [ ] **Step 1: Write the failing test.**

```python
# tests/scripts/test_cross_compare.py
import importlib.util
from pathlib import Path
import pandas as pd

_P = Path(__file__).resolve().parents[2] / "scripts" / "analysis" / "intersectional" / "cohort_cross_compare.py"
_S = importlib.util.spec_from_file_location("cohort_cross_compare", _P)
mod = importlib.util.module_from_spec(_S); _S.loader.exec_module(mod)

def test_build_cross_compare_matches_and_is_rank_based(tmp_path):
    coh = tmp_path / "coh.csv"; bulk = tmp_path / "bulk.csv"
    pd.DataFrame([{"subject_id":"BG_029","genotype":"D2","region":"DMS",
                   "track":"behavioral_fa","scheme":"scheme3","auroc_mean":0.62}]).to_csv(coh, index=False)
    pd.DataFrame([{"genotype":"D2","region":"DMS","auroc_mean":0.55}]).to_csv(bulk, index=False)
    out = mod.build_cross_compare(str(coh), str(bulk))
    r = out[(out["genotype"]=="D2") & (out["region"]=="DMS")].iloc[0]
    assert r["auroc_cohort"] == 0.62 and r["auroc_bulk"] == 0.55
    assert abs(r["delta_auroc"] - 0.07) < 1e-9
    assert r["intersectional_subject"] == "BG_029"
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement.**

```python
# scripts/analysis/intersectional/cohort_cross_compare.py
"""Rank-based bulk-vs-intersectional comparison (secondary, caveated).

Compares ONLY indicator-invariant quantities (AUROC here; extend to sign /
latency similarly). Magnitude (dF/F) is never compared across indicators —
enforced by cohort.assert_rank_based.
"""
import argparse, logging, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns

_repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo_root / "src"))
from visdetect_photom.core import cohort
from visdetect_photom.core.constants import GENOTYPE_COLORS

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def build_cross_compare(cohort_c1_csv, bulk_auroc_csv,
                        track="behavioral_fa", scheme="scheme3"):
    cohort.assert_rank_based("auroc")  # guard: rank-based only
    coh = pd.read_csv(cohort_c1_csv)
    if "track" in coh.columns:
        coh = coh[(coh["track"] == track) & (coh["scheme"] == scheme)]
    bulk = pd.read_csv(bulk_auroc_csv)
    out = []
    for geno in ("D1", "D2"):
        for region in ("DMS", "VMS"):
            m = cohort.match_cohort_cells(geno, region)
            if not m["intersectional"]:
                continue
            cs = m["intersectional"][0]
            crow = coh[(coh["genotype"] == geno) & (coh["region"] == region)]
            brow = bulk[(bulk["genotype"] == geno) & (bulk["region"] == region)]
            a_c = float(crow.iloc[0]["auroc_mean"]) if len(crow) else np.nan
            a_b = float(brow.iloc[0]["auroc_mean"]) if len(brow) else np.nan
            out.append({"genotype": geno, "region": region,
                        "intersectional_subject": cs, "bulk_subjects": ",".join(m["bulk"]),
                        "auroc_cohort": a_c, "auroc_bulk": a_b,
                        "delta_auroc": a_c - a_b})
    return pd.DataFrame(out)

def main():
    ap = argparse.ArgumentParser(description="Rank-based bulk-vs-intersectional comparison")
    rr = Path(__file__).resolve().parents[3]
    ap.add_argument("--cohort_c1", default=str(rr / "FIGURES" / "intersectional_mos" / "cohort_c1_cell_summary.csv"))
    ap.add_argument("--bulk_auroc", default=str(rr / "FIGURES" / "C1_fa_suppression" / "c1_auroc_stats.csv"))
    ap.add_argument("--output_dir", default=str(rr / "FIGURES" / "intersectional_mos"))
    args = ap.parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = build_cross_compare(args.cohort_c1, args.bulk_auroc)
    df.to_csv(out_dir / "cohort_cross_compare.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [f"{r.genotype}·{r.region}" for r in df.itertuples()]
    x = np.arange(len(df))
    ax.bar(x - 0.2, df["auroc_bulk"], 0.4, label="bulk-8m", color="#999999")
    ax.bar(x + 0.2, df["auroc_cohort"], 0.4, label="MOs-recipient 6f", color="#d62728")
    ax.axhline(0.5, color="k", ls="--", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1)
    ax.set_ylabel("brake AUROC (rank-based; magnitudes not compared)")
    ax.set_title("Cross-cohort brake AUROC (caveated: 6f vs 8m, n=1/cell)")
    ax.legend(); sns.despine(ax=ax)
    fig.savefig(out_dir / "cohort_cross_compare.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.info("Done.")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run → PASS.**

- [ ] **Step 5: Commit.**

```bash
git add scripts/analysis/intersectional/cohort_cross_compare.py tests/scripts/test_cross_compare.py
git commit -m "feat(cohort): rank-based bulk-vs-intersectional cross comparison"
```

---

## Task 12: Companion PETH panels per cell — `cohort_companion_peths.py`

**Files:**
- Create: `scripts/analysis/intersectional/cohort_companion_peths.py`
- Test: `tests/scripts/test_cohort_smoke.py` (PETH case)

**Interfaces:**
- Consumes: `cohort.load_cohort_sessions`; `geometry.compute_geometry_metrics_for_session` (its 2nd/3rd returns: `traces = {(region, epoch): mean_trace}` and `time_axis`).
- Produces: `cohort_companion_peths.png` — one row per cell (subject), with a **change-aligned** panel (change_hit vs change_miss) and a **separate lick-aligned** panel (hit_lick vs fa_lick). Alignments are never mixed on one panel (standing rule).

- [ ] **Step 1: Add the failing smoke case.**

```python
# append to tests/scripts/test_cohort_smoke.py
PETH = os.path.join(REPO, "scripts", "analysis", "intersectional", "cohort_companion_peths.py")

@pytest.mark.skipif(not os.path.isdir(DATA), reason="cohort data not present")
def test_cohort_companion_peths_runs(tmp_path):
    out = tmp_path / "out"
    proc = subprocess.run([sys.executable, PETH, "--max_sessions", "4",
                           "--root_dir", DATA, "--output_dir", str(out)],
                          cwd=REPO, capture_output=True, text=True, timeout=1200)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert (out / "cohort_companion_peths.png").exists()
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement.**

```python
# scripts/analysis/intersectional/cohort_companion_peths.py
"""Companion PETH panels per cell for the intersectional cohort (BG_027-030).

Per cell (subject): a change-aligned panel (change_hit vs change_miss) and a
SEPARATE lick-aligned panel (hit_lick vs fa_lick). Trial-pooled (per-session
mean traces averaged across sessions) — illustrative, not the statistic.
Alignments are never mixed on one panel.
"""
import argparse, logging, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt, seaborn as sns

_repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo_root / "src"))
from visdetect_photom.core import cohort
from visdetect_photom.analysis.state_provider import PooledStateProvider
from visdetect_photom.analysis.geometry import compute_geometry_metrics_for_session

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
CHANGE_EPOCHS = [("change_hit", "Hit"), ("change_miss", "Miss")]
LICK_EPOCHS = [("hit_lick", "Hit lick"), ("fa_lick", "FA lick")]

def _norm(s):
    s = str(s)
    return s if s.startswith("BG_") else f"BG_{s.zfill(3)}"

def main():
    ap = argparse.ArgumentParser(description="Intersectional cohort companion PETHs")
    rr = Path(__file__).resolve().parents[3]
    ap.add_argument("--root_dir", default=str(rr / "photom_data" / "intrsct_GCaMP6f"))
    ap.add_argument("--output_dir", default=str(rr / "FIGURES" / "intersectional_mos"))
    ap.add_argument("--no-qc", action="store_true", default=False)
    ap.add_argument("--max_sessions", type=int, default=None)
    args = ap.parse_args()
    use_qc = not args.no_qc
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    sessions = cohort.load_cohort_sessions("intersectional_mos", args.root_dir,
                                           max_sessions=args.max_sessions)
    if not sessions:
        logging.error("No cohort sessions loaded."); sys.exit(1)

    sp, keep = PooledStateProvider(), ["All"]
    # accumulate per (subject, epoch) -> list of per-session mean traces
    acc = defaultdict(list)
    time_axis = None
    for sess in sessions:
        _, traces, t = compute_geometry_metrics_for_session(
            sess, use_qc=use_qc, state_provider=sp, keep_states=keep)
        if t is not None and time_axis is None:
            time_axis = t
        subj = _norm(sess.subject_id)
        for (region, epoch), tr in traces.items():
            acc[(subj, epoch)].append(tr)
    if time_axis is None:
        logging.error("No traces extracted."); sys.exit(1)

    subjects = sorted({k[0] for k in acc})
    fig, axes = plt.subplots(len(subjects), 2, figsize=(11, 3 * max(len(subjects), 1)),
                             squeeze=False)
    fig.suptitle("Intersectional cohort — companion PETHs (trial-pooled; illustrative)", fontsize=12)
    for ri, subj in enumerate(subjects):
        for ci, (epochs, xl, title) in enumerate(
                [(CHANGE_EPOCHS, "Time from change (s)", "change-aligned"),
                 (LICK_EPOCHS, "Time from lick (s)", "lick-aligned")]):
            ax = axes[ri][ci]
            for epoch, label in epochs:
                trs = acc.get((subj, epoch))
                if not trs:
                    continue
                m = np.nanmean(np.array(trs), axis=0)
                ax.plot(time_axis, m, lw=1.4, label=label)
            ax.axvline(0, color="k", ls="--", lw=0.8)
            ax.axhline(0, color="grey", lw=0.5, alpha=0.4)
            ax.set_title(f"{subj} — {title}", fontsize=9)
            ax.set_xlabel(xl, fontsize=8); ax.set_ylabel("Δ z-dF/F", fontsize=8)
            ax.legend(fontsize=7); sns.despine(ax=ax)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = out_dir / "cohort_companion_peths.png"
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    logging.info(f"Saved {p}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run → PASS** (or manual).

- [ ] **Step 5: Commit.**

```bash
git add scripts/analysis/intersectional/cohort_companion_peths.py tests/scripts/test_cohort_smoke.py
git commit -m "feat(cohort): per-cell companion PETH panels (alignments never mixed)"
```

---

## Final verification

- [ ] **Run the whole suite:** `py -m pytest -q` → all green (bulk suites included; the never-pool guard must not regress them).
- [ ] **End-to-end on real data** (once the cohort base branch has the C1 code + data is local): run, in order, `c1_cohort_suppression.py`, `c2_cohort_geometry.py`, `cohort_qc_report.py`, `cohort_companion_peths.py`, then `cohort_cross_compare.py`. Confirm `FIGURES/intersectional_mos/` holds the session-scalar + cell-summary CSVs, the QC report, the cross-compare CSV, and the 2×2 + companion-PETH + cross-compare PNGs.
- [ ] **Sanity:** confirm a bulk run (`11_fa_suppression.py`) logs BG_027–030 among excluded mice (never pooled), and the cohort cell summaries name n=1 mouse/cell with session-bootstrap CIs.

---

## Self-review notes (author)

- **Spec coverage:** Phase 0 audit → Task 0; constants/registry/staging (spec §4) → Tasks 1,2,7; never-pool (spec §2) → Task 3; session-unit stats (spec §5) → Tasks 4,6,8,9; companion PETHs per cell (spec §5/§6) → Task 12; QC same-pipeline + 6f calibration (spec §5 QC) → Task 10; rank-based cross-cohort (spec §6) → Tasks 5,11.
- **n=1/cell** honored: cohort scripts use session-unit summaries; no cross-mouse inferential test inside a cohort.
- **Parallel-chat safety:** no edits to `08_*`/`11_*`; never-pool enforced in `excluded_mice` (shared, additive).
