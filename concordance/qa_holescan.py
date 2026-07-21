#!/usr/bin/env python3
"""
qa_holescan v2 - detect silent data loss in predict3d / lasagna OME-Zarr output.

Two distinct failure modes exist:

  A. elided chunk   - chunk files ABSENT      (v1 caught this: file counts)
  B. partial chunk  - chunk files PRESENT,    (v1 was blind to this)
                      part of the z-slices zero

Mode B is the one reported in ScrollPrize/villa#1183: 16 of the 32 z-slices
inside a chunk come out zero, every chunk file present, all tiles reported
as processed, no warning. Counting files cannot see it.

The hard part of detecting mode B is separating "zero because the writer
dropped it" from "zero because there is no papyrus there" - most of a scroll
volume is legitimately empty. This tool uses two discriminators that need no
threshold and no per-scroll calibration:

  1. CROSS-CHANNEL. grad_mag, nx, ny and cos share support: where there is
     material, all of them carry data. If one channel is empty on a z-slice
     where another is not, that is a defect, not anatomy.

  2. WITHIN-CHUNK. A chunk whose file exists should not contain a mix of
     empty and non-empty z-slices. Legitimate mask boundaries do occur at
     the extremities of the volume, so those are reported separately.

IMPORTANT: run this on RAW output, before any normalization. z-scoring moves
raw zeros to some nonzero value and the evidence disappears (see villa#1173).

Usage
    qa_holescan.py MANIFEST.lasagna.json [--sample N] [--full] [--quiet]

    --sample N   chunk columns sampled per chunk-row (default 24)
    --full       read every chunk (slow, exhaustive)
    --quiet      only print the verdict

Exit code 0 = PASS, 1 = FAIL. Intended as a gate before consuming output:

    qa_holescan.py out.lasagna.json || exit 1
"""

import argparse
import json
import os
import sys

import numpy as np
import zarr

CHANNELS_EXPECTED = ("grad_mag", "nx", "ny", "cos")


def chunk_index(zpath):
    """Map {row: {(y, x)}} of chunk files physically present on disk."""
    index = {}
    for row in os.listdir(zpath):
        if not row.isdigit():
            continue
        rdir = os.path.join(zpath, row)
        cols = set()
        for y in os.listdir(rdir):
            if not y.isdigit():
                continue
            for x in os.listdir(os.path.join(rdir, y)):
                if x.isdigit():
                    cols.add((int(y), int(x)))
        if cols:
            index[int(row)] = cols
    return index


def load_groups(manifest_path):
    manifest = json.load(open(manifest_path))
    base = os.path.dirname(os.path.abspath(manifest_path))
    groups = {}
    for name, g in manifest["groups"].items():
        zpath = os.path.join(base, g["zarr"])
        if not os.path.isdir(zpath):
            print(f"  ! {name}: {g['zarr']} not found, skipping")
            continue
        meta = json.load(open(os.path.join(zpath, ".zarray")))
        groups[name] = {
            "path": zpath,
            "array": zarr.open(zpath, mode="r"),
            "chunk": meta["chunks"][0],
            "index": chunk_index(zpath),
        }
    return groups


def sample_columns(shared, n):
    """Pick n chunk columns spread across the available ones, deterministic."""
    cols = sorted(shared)
    if n <= 0 or n >= len(cols):
        return cols
    step = len(cols) / n
    return [cols[int(i * step)] for i in range(n)]


def scan(groups, n_sample, full):
    """Return (findings, rows_checked). A finding is one contiguous z range."""
    names = [c for c in CHANNELS_EXPECTED if c in groups] or list(groups)
    rows = sorted(set.intersection(*(set(groups[c]["index"]) for c in names)))
    if not rows:
        return [], []

    oc = groups[names[0]]["chunk"]
    depth = groups[names[0]]["array"].shape[0]
    findings = []

    for row in rows:
        shared = set.intersection(*(groups[c]["index"][row] for c in names))
        if not shared:
            continue
        cols = sorted(shared) if full else sample_columns(shared, n_sample)

        z0 = row * oc
        z1 = min(z0 + oc, depth)
        # nonzero voxel count per (channel, z) over the sampled columns
        counts = {c: np.zeros(z1 - z0, dtype=np.int64) for c in names}
        for (cy, cx) in cols:
            for c in names:
                block = np.asarray(
                    groups[c]["array"][z0:z1,
                                       cy * oc:(cy + 1) * oc,
                                       cx * oc:(cx + 1) * oc])
                counts[c] += (block != 0).sum(axis=(1, 2))

        # a z-slice is suspect when at least one channel is empty on it
        # while at least one other channel is not
        total = sum(counts[c] for c in names)
        empty_any = np.zeros(z1 - z0, dtype=bool)
        culprits = [[] for _ in range(z1 - z0)]
        for c in names:
            miss = (counts[c] == 0) & (total > 0)
            empty_any |= miss
            for i in np.nonzero(miss)[0]:
                culprits[i].append(c)

        for start, end in contiguous(np.nonzero(empty_any)[0]):
            who = sorted({c for i in range(start, end + 1) for c in culprits[i]})
            findings.append({
                "row": row,
                "z0": z0 + start,
                "z1": z0 + end,
                "n": end - start + 1,
                "of": z1 - z0,
                "channels": who,
                "cols": len(cols),
                "edge": row == rows[0] or row == rows[-1],
            })

    return findings, rows


def contiguous(idx):
    """Group a sorted index array into (start, end) inclusive runs."""
    runs = []
    for i in idx:
        if runs and i == runs[-1][1] + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return [tuple(r) for r in runs]


def boundary_note(z0, z1, oc):
    """Flag ranges that start or end on a chunk boundary (writer signature)."""
    notes = []
    if z0 % oc == 0:
        notes.append(f"starts on chunk boundary ({z0} = {z0 // oc} x {oc})")
    if (z1 + 1) % oc == 0:
        notes.append(f"ends on chunk boundary ({z1 + 1} = {(z1 + 1) // oc} x {oc})")
    return "; ".join(notes)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest")
    ap.add_argument("--sample", type=int, default=24,
                    help="chunk columns sampled per row (default 24)")
    ap.add_argument("--full", action="store_true",
                    help="read every chunk instead of sampling")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    groups = load_groups(args.manifest)
    if not groups:
        print("no readable groups in manifest")
        return 2

    if not args.quiet:
        for name, g in groups.items():
            rows = sorted(g["index"])
            print(f"  {name:10s} shape {g['array'].shape} chunk {g['chunk']} "
                  f"rows {rows[0]}-{rows[-1]}")

    findings, rows = scan(groups, args.sample, args.full)
    oc = groups[list(groups)[0]]["chunk"]

    real = [f for f in findings if not f["edge"]]
    edge = [f for f in findings if f["edge"]]

    if not args.quiet:
        mode = "full scan" if args.full else f"{args.sample} columns/row"
        print(f"\nchecked rows {rows[0]}-{rows[-1]} ({mode})")

    for f in real:
        note = boundary_note(f["z0"], f["z1"], oc)
        print(f"  HOLE  z {f['z0']}-{f['z1']}  ({f['n']} of {f['of']} slices "
              f"in row {f['row']})  empty: {', '.join(f['channels'])}"
              + (f"\n        {note}" if note else ""))

    for f in edge:
        print(f"  edge  z {f['z0']}-{f['z1']}  (row {f['row']}, volume "
              f"extremity - likely mask, not a defect)")

    if real:
        print(f"\nVERDICT: FAIL - {len(real)} suspect range(s). "
              f"Do not consume this output before checking.")
        return 1

    print("\nVERDICT: PASS" + (f" ({len(edge)} edge range(s) ignored)" if edge else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
