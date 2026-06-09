---
name: verification-before-completion
description: Use when about to claim work is complete — requires running verification and confirming output before making success claims
---

# Verification Before Completion

## Overview

Claiming work is complete without verification is dishonesty, not efficiency.

**Core principle:** Evidence before claims, always.

## The Iron Law

```
NO COMPLETION CLAIMS WITHOUT FRESH VERIFICATION EVIDENCE
```

If you haven't run the verification command in this message, you cannot claim it passes.

## The Gate Function

```
BEFORE claiming any status:

1. IDENTIFY: What command/check proves this claim?
2. RUN: Execute the FULL command (fresh, complete)
3. READ: Full output, check exit code, inspect results
4. VERIFY: Does output confirm the claim?
   - If NO: State actual status with evidence
   - If YES: State claim WITH evidence
5. ONLY THEN: Make the claim
```

## Common Failures

| Claim | Requires | Not Sufficient |
|-------|----------|----------------|
| Script runs | Script output: exit 0 | "should work now" |
| Figure correct | Inspect output figure | Script ran without errors |
| Stats valid | Check stats values | "computed correctly" |
| Bug fixed | Reproduce: resolved | Code changed, assumed fixed |
| dF/F correct | Check signal values, plot | "preprocessing complete" |
| PETH extracted | Check matrix shape, time axis | "no errors" |

## Red Flags - STOP

- Using "should", "probably", "seems to"
- Expressing satisfaction before verification ("Great!", "Done!")
- About to commit without verification
- Relying on partial verification

## Key Patterns

**Analysis scripts:**
```
OK: [Run script] [See: figure saved, data exported] "Script complete, figure at FIGURES/..."
BAD: "Should produce the correct figure now"
```

**Bug fixes:**
```
OK: [Run failing script] [See: no error, correct output] "Bug fixed, verified"
BAD: "Fixed the typo, should work now"
```

## The Bottom Line

Run the command. Read the output. Inspect the result. THEN claim the result. Non-negotiable.
