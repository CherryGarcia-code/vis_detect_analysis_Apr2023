# Cortico-Basal Ganglia Visual Change Detection: Project Overview

## Preface
This project investigates neural dynamics in the medial striatum of mice performing a visual change-detection task. The experiment combines photometry recordings, and behavioral monitoring to study how striatal circuits support learning and action selection.

## Experimental Setup
- **Subjects:** Drd1-Cre or Drd2-Cre mice expressing GCaMP8m in DMS or VMS or VLS
-
- **Task:** Mice perform a visual change-detection task:
  - **Trial Structure:**  
    - Inter-trial interval (gray screen)
    - Baseline (drifting grating, 1 Hz, noisy)
    - Speed change (to 1.25, 1.35, 1.5, 2, or 4 Hz) after ≥6s
    - Mouse must lick within 2.15s of change for reward
    - Outcomes: Hit, Miss, False Alarm (FA), Abort
- **Recording:**  
  - photometry (chronic)
  - Video (face, pupil, body)
  - Behavioral sensors (wheel, lick spout)
- 

## Data Acquisition & Storage
- **Raw Data:**  
  - Neural: `.bin` files, spike times, clusters, templates
  - Behavioral: trial logs, outcomes, reaction times
  - Video: synchronized camera streams
  - Session metadata: settings, parameters

- **Example Data Structure:**  
  - `NPX_probes`: neural data (channels, spike times, clusters, waveforms)
  - `NI_events`: event timings (stimulus, licks, valves, etc.)
  - `behav_data`: trial-by-trial info (stimulus, outcome, reaction time)
  - `SessionSettings` & `ComputerSettings`: experiment/session metadata

## Research Questions

### Technical
- Are striatal cells properly optotagged? Can we classify DI/D2 SPNs?
- Can waveform properties be used to define cell types if optotagging is inefficient?
- Can units be reliably tracked across days/training?

### Biological
- How do single neurons respond to different trial outcomes?
- Can neurons be grouped by response patterns?
- What does population activity look like across trials and learning?
- How do mice learn to balance impulsive actions and stimulus sensitivity?
- How is expert performance represented in neural activity?

---

## Getting Started

- See the `PROMPT.md` for coding and analysis instructions.
- Example session schema: `RAW_SESSION_SCHEMA_BG_031_260325.json`
