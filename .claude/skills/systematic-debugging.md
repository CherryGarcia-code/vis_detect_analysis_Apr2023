---
name: systematic-debugging
description: Use when encountering any bug, test failure, or unexpected behavior, before proposing fixes
---

# Systematic Debugging

## Overview

Random fixes waste time and create new bugs. Quick patches mask underlying issues.

**Core principle:** ALWAYS find root cause before attempting fixes. Symptom fixes are failure.

## The Iron Law

```
NO FIXES WITHOUT ROOT CAUSE INVESTIGATION FIRST
```

If you haven't completed Phase 1, you cannot propose fixes.

## When to Use

Use for ANY technical issue: bugs, unexpected behavior, performance problems, script failures.

**Use ESPECIALLY when:**
- Under time pressure
- "Just one quick fix" seems obvious
- You've already tried multiple fixes
- You don't fully understand the issue

## The Four Phases

### Phase 1: Root Cause Investigation

**BEFORE attempting ANY fix:**

1. **Read Error Messages Carefully** — Don't skip. Read stack traces completely. Note line numbers, file paths.
2. **Reproduce Consistently** — Can you trigger it reliably? If not, gather more data.
3. **Check Recent Changes** — Git diff, new dependencies, config changes.
4. **Trace Data Flow** — Where does the bad value originate? Fix at source, not symptom.

### Phase 2: Pattern Analysis

1. **Find Working Examples** — Similar working code in the codebase
2. **Compare Against References** — Read reference implementation COMPLETELY
3. **Identify Differences** — List every difference, however small

### Phase 3: Hypothesis and Testing

1. **Form Single Hypothesis** — "I think X is the root cause because Y"
2. **Test Minimally** — SMALLEST possible change, one variable at a time
3. **Verify** — If it worked → Phase 4. If not → NEW hypothesis (don't pile fixes)

### Phase 4: Implementation

1. **Single Fix** — ONE change, no "while I'm here" improvements
2. **Verify Fix** — Run the script, check output
3. **If 3+ Fixes Failed** — STOP. Question the approach. Discuss with user.

## Red Flags - STOP and Follow Process

If you catch yourself thinking:
- "Quick fix for now, investigate later"
- "Just try changing X and see"
- "I don't fully understand but this might work"
- "One more fix attempt" (when already tried 2+)

**ALL mean: STOP. Return to Phase 1.**

## Quick Reference

| Phase | Key Activities | Success Criteria |
|-------|---------------|------------------|
| **1. Root Cause** | Read errors, reproduce, trace data | Understand WHAT and WHY |
| **2. Pattern** | Find working examples, compare | Identify differences |
| **3. Hypothesis** | Form theory, test minimally | Confirmed or new hypothesis |
| **4. Implementation** | Fix, verify | Bug resolved, output correct |
