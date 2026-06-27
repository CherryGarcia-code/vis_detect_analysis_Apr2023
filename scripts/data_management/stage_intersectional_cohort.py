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
    dest = Path(dest_dir)
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
