#!/usr/bin/env python3
"""
qa_holescan v3 - detect silent data loss in predict3d / lasagna OME-Zarr output.

Failure modes:

  A. elided chunk   - chunk files ABSENT      (v1: file counts)
  B. partial chunk  - chunk files PRESENT,    (v2: cross-channel support)
                      part of the z-slices zero

Mode B is the one reported in ScrollPrize/villa#1183: 16 of the 32 z-slices
inside a chunk come out zero, every chunk file present, no warning.

v3 adds three things, after running v2 on masked full volumes produced
false positives (reported by waldkauz, unrolling-vc3d, 28 Jul 2026):

  1. MASK FILTER (waldkauz). A mask is z-invariant; a writer hole is
     z-localized. Zeros inside a suspect range that were already zero just
     outside it are standing footprint (mask or anatomy); only zeros that
     appear inside the range count as the hole. Handles volumes where the
     mask is applied to some channels but not others, which breaks the
     bare cross-channel invariant. Border connectivity of the new-zero
     area is reported as secondary evidence.

  2. CONTEXT FILTER (waldkauz). A suspect range must have nonzero data in
     the same channel both before and after it in z. A range that runs to
     the volume end is mask or anatomy, not a writer hole.

  3. PHYSICAL-ZERO CLASS. grad_mag can legitimately reach 0 where winding
     spacing is too large for the patch to represent (waldkauz). Signature
     from villa#1183 separates the classes: writer defects are rectangular,
     chunk-row aligned, sharp-edged, and empty across ALL columns at once.
     Physical zeros are spatially partial and do not respect chunk
     boundaries. Column-partial zeros are reported as LOCAL, informational,
     and do not fail the gate.

Discriminators still need no threshold and no per-scroll calibration.

IMPORTANT: run this on RAW output, before any normalization (villa#1173).

Usage
    qa_holescan.py MANIFEST.lasagna.json [--sample N] [--full] [--quiet]

    --sample N   chunk columns sampled per chunk-row (default 24)
    --full       read every chunk (slow, exhaustive)
    --quiet      only print the verdict

Exit code 0 = PASS, 1 = FAIL. Intended as a gate before consuming output:

    qa_holescan.py out.lasagna.json || exit 1

Only DEFECT findings fail the gate. MASKED and LOCAL findings are printed
for information.
"""

import argparse
import json
import os
import sys

import numpy as np
import zarr

CHANNELS_EXPECTED = ("grad_mag", "nx", "ny", "cos")
FLOOD_DS = 8          # downsample factor for the border flood fill
MASK_BORDER_FRAC = 0.99  # zero area border-connected above this => MASKED


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


def border_connected_fraction(zslice):
    """Fraction of the zero area of a 2D slice connected to its border.

    Pure numpy flood fill on a downsampled copy. Good enough to classify
    mask (border-connected) against interior holes; not meant to be exact.
    """
    m = (zslice[::FLOOD_DS, ::FLOOD_DS] == 0)
    if not m.any():
        return 1.0
    conn = np.zeros_like(m)
    conn[0, :] = m[0, :]
    conn[-1, :] = m[-1, :]
    conn[:, 0] = m[:, 0]
    conn[:, -1] = m[:, -1]
    for _ in range(max(m.shape)):
        grown = conn.copy()
        grown[1:, :] |= conn[:-1, :]
        grown[:-1, :] |= conn[1:, :]
        grown[:, 1:] |= conn[:, :-1]
        grown[:, :-1] |= conn[:, 1:]
        grown &= m
        if (grown == conn).all():
            break
        conn = grown
    return conn.sum() / m.sum()


def channel_has_data(group, z_from, z_to, cols, oc):
    """True if the channel has any nonzero voxel in [z_from, z_to) over cols."""
    depth = group["array"].shape[0]
    z_from = max(0, z_from)
    z_to = min(depth, z_to)
    if z_from >= z_to:
        return None  # range does not exist: volume end
    for (cy, cx) in cols:
        block = np.asarray(group["array"][z_from:z_to,
                                          cy * oc:(cy + 1) * oc,
                                          cx * oc:(cx + 1) * oc])
        if (block != 0).any():
            return True
    return False


def scan(groups, n_sample, full):
    """Return (findings, checked, unchecked).

    Channels are partitioned by grid signature (shape, chunk): the
    cross-channel invariant only holds between channels on the same grid.
    cos often lives on its own grid (--cos-scaledown), so it is scanned
    with its peers when it has any, and reported as unchecked when alone.

    Finding classes:
      DEFECT - channel empty across ALL sampled columns on z-slices where
               another channel has data, interior zeros, data on both sides
      MASKED - zeros match the standing footprint, or range at a volume end
      LOCAL  - column-partial cross-channel zeros (physical-zero candidates)
    """
    sigs = {}
    for name, g in groups.items():
        key = (tuple(g["array"].shape), g["chunk"])
        sigs.setdefault(key, []).append(name)

    findings, checked, unchecked = [], [], []

    for key, part in sigs.items():
        names = [c for c in CHANNELS_EXPECTED if c in part] or sorted(part)
        if len(names) < 2:
            unchecked.append(names)
            continue
        rows = sorted(set.union(*(set(groups[c]["index"]) for c in names)))
        if not rows:
            unchecked.append(names)
            continue

        oc = groups[names[0]]["chunk"]
        depth = groups[names[0]]["array"].shape[0]
        checked.append((names, rows))

        for row in rows:
            shared = set().union(*(groups[c]["index"].get(row, set())
                                   for c in names))
            if not shared:
                continue
            cols = sorted(shared) if full else sample_columns(shared, n_sample)

            z0 = row * oc
            z1 = min(z0 + oc, depth)
            nz = z1 - z0
            ncols = len(cols)

            # nonzero voxel count per (channel, z, column)
            counts = {c: np.zeros((nz, ncols), dtype=np.int64) for c in names}
            for j, (cy, cx) in enumerate(cols):
                for c in names:
                    block = np.asarray(
                        groups[c]["array"][z0:z1,
                                           cy * oc:(cy + 1) * oc,
                                           cx * oc:(cx + 1) * oc])
                    counts[c][:, j] = (block != 0).sum(axis=(1, 2))

            # per-slice, per-channel: empty over ALL columns (writer signature)
            # vs empty on SOME columns where another channel has data there
            row_total = sum(counts[c] for c in names)          # (nz, ncols)
            all_empty = np.zeros(nz, dtype=bool)
            culprits = [[] for _ in range(nz)]
            local_cells = {c: 0 for c in names}

            for c in names:
                others = row_total - counts[c]
                full_miss = (counts[c].sum(axis=1) == 0) & (others.sum(axis=1) > 0)
                all_empty |= full_miss
                for i in np.nonzero(full_miss)[0]:
                    culprits[i].append(c)
                # column-partial: this channel zero in a cell where others aren't,
                # on slices that are NOT full-miss (those are counted above)
                part = (counts[c] == 0) & (others > 0)
                part[full_miss, :] = False
                local_cells[c] += int(part.sum())

            for start, end in contiguous(np.nonzero(all_empty)[0]):
                who = sorted({c for i in range(start, end + 1) for c in culprits[i]})
                f = {
                    "row": row,
                    "z0": z0 + start,
                    "z1": z0 + end,
                    "n": end - start + 1,
                    "of": nz,
                    "channels": who,
                    "cols": ncols,
                    "oc": oc,
                    "class": "DEFECT",
                    "why": "",
                }

                # context filter (waldkauz): data before AND after in the same
                # channel, looking across row boundaries; volume end => MASKED
                g = groups[who[0]]
                before = channel_has_data(g, f["z0"] - 3, f["z0"], cols, oc)
                after = channel_has_data(g, f["z1"] + 1, f["z1"] + 4, cols, oc)
                if before is None or after is None:
                    f["class"], f["why"] = "MASKED", "range touches volume end"
                elif not before or not after:
                    f["class"], f["why"] = "MASKED", "no data on one side"
                else:
                    # mask filter: a mask is z-invariant, a writer hole is
                    # z-localized. Zeros inside the range that were already
                    # zero just outside it are standing footprint (mask or
                    # anatomy); zeros that appear only inside the range are
                    # the hole.
                    mid = (f["z0"] + f["z1"]) // 2
                    ref = np.abs(np.asarray(g["array"][max(0, f["z0"] - 3):f["z0"]])).max(axis=0)
                    sl = np.asarray(g["array"][mid])
                    extra = (sl == 0) & (ref != 0)
                    denom = max(1, int((ref != 0).sum()))
                    frac_extra = extra.sum() / denom
                    if frac_extra < 0.01:
                        f["class"] = "MASKED"
                        f["why"] = "zeros match the standing footprint outside the range"
                    else:
                        bc = border_connected_fraction(np.where(extra, 0, 1))
                        f["why"] = (f"{frac_extra:.0%} of populated footprint newly "
                                    f"zero inside the range")
                        if bc < 1.0:
                            f["why"] += f"; new-zero area {bc:.0%} border-connected"

                findings.append(f)
                # mode A annotation: culprit chunk files absent on disk
                for c in who:
                    have = groups[c]["index"].get(row, set())
                    absent = sum(1 for col in cols if col not in have)
                    if absent:
                        f["why"] = (f["why"] + "; " if f["why"] else "") + \
                            f"{absent}/{len(cols)} chunk files absent ({c})"

            n_local = sum(local_cells.values())
            if n_local:
                findings.append({
                    "row": row, "z0": z0, "z1": z1 - 1, "n": 0, "of": nz,
                    "channels": [c for c in names if local_cells[c]],
                    "cols": ncols, "oc": oc, "class": "LOCAL",
                    "why": f"{n_local} column-partial zero cells "
                           f"(physical-zero candidates, not chunk-shaped)",
                })

    return findings, checked, unchecked


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

    findings, checked, unchecked = scan(groups, args.sample, args.full)

    defects = [f for f in findings if f["class"] == "DEFECT"]
    masked = [f for f in findings if f["class"] == "MASKED"]
    local = [f for f in findings if f["class"] == "LOCAL"]

    if not args.quiet:
        mode = "full scan" if args.full else f"{args.sample} columns/row"
        for names, rows in checked:
            print(f"\nchecked {'+'.join(names)} rows {rows[0]}-{rows[-1]} ({mode})")
        for names in unchecked:
            print(f"\nnot cross-checkable: {'+'.join(names)} "
                  f"(single channel on its grid)")
        if not checked:
            print("\nno grid with two or more channels to cross-check")

    for f in defects:
        note = boundary_note(f["z0"], f["z1"], f["oc"])
        if f["why"]:
            note = f"{note}; {f['why']}" if note else f["why"]
        print(f"  HOLE   z {f['z0']}-{f['z1']}  ({f['n']} of {f['of']} slices "
              f"in row {f['row']}, all {f['cols']} columns)  "
              f"empty: {', '.join(f['channels'])}"
              + (f"\n         {note}" if note else ""))

    if not args.quiet:
        for f in masked:
            print(f"  masked z {f['z0']}-{f['z1']}  (row {f['row']}: {f['why']})")
        for f in local:
            print(f"  local  row {f['row']}: {f['why']} "
                  f"channels: {', '.join(f['channels'])}")

    if defects:
        print(f"\nVERDICT: FAIL - {len(defects)} defect range(s). "
              f"Do not consume this output before checking.")
        return 1

    extras = []
    if masked:
        extras.append(f"{len(masked)} masked")
    if local:
        extras.append(f"{len(local)} local")
    print("\nVERDICT: PASS" + (f" ({', '.join(extras)} range(s) noted)" if extras else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
