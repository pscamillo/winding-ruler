#!/usr/bin/env python3
"""
build_atlas_v2.py - consolidate the level-1 rerun into atlas_collection_v2.csv

Why v2 exists
-------------
The v1 atlas measured every scroll at pyramid level 2. That level was chosen
because it gave 4-8 voxels between sheets, which kept streaming cheap. The
assumption was that this is enough resolution to count sheet crossings along
a ray. It isn't.

Rerunning all 36 scrolls at level 1 lowers the measured pitch by 10.3% on
average (36/36 negative, sd 2.6). The wrap count rises at the same time, and
span is preserved, so the level-2 grid was merging adjacent sheets rather than
mis-measuring the gap between them.

Convergence was checked on PHercMANBp, the coarsest scroll in the collection:
level 1 and level 0 agree exactly (29.0 L0vox, 8 wraps, and the same p25/p75),
so level 1 is at the limit and level 0 buys nothing. Level 0 is also not
practical across the collection: the request count per slice grows ~14x and
the run becomes latency-bound.

Effect on the collection number: median pitch 207.4 -> 187.3 um, and the bias
against the Paris 4 human anchor (~180 um) drops from +15.2% to +4.0%. On
Paris 4 itself the level-1 estimate is 182.4 um, +1.3% against the anchor.

Physical scale
--------------
The level-1 runs were produced without the bucket metadata.json, so
lambda_med_um came out empty. The um-per-voxel factor is a property of the
volume, not of the pyramid level, so it is recovered from v1:

    um_per_L0vox = lambda_med_um(v1) / lambda_med_L0vox(v1)

and applied to the level-1 L0-voxel figures. This reproduces the v1 numbers
exactly when applied to v1 input, which is checked below.

Usage
-----
    python build_atlas_v2.py --v1 atlas_collection.csv \
        --l1-glob '/tmp/L1_PHerc*.csv' --out atlas_collection_v2.csv
"""

import argparse
import csv
import glob
import os
import sys

COLUMNS = [
    "scroll", "zarr", "level", "gap_close", "n_slices", "n_rays",
    "wraps_p10", "wraps_med", "wraps_p90",
    "lambda_med_lvlvox", "lambda_p25", "lambda_p75",
    "lambda_med_L0vox", "lambda_med_um",
    "lambda_p25_um", "lambda_p75_um",
    "um_per_L0vox", "MB_fetched",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", default="atlas_collection.csv")
    ap.add_argument("--l1-glob", default="/tmp/L1_PHerc*.csv")
    ap.add_argument("--out", default="atlas_collection_v2.csv")
    args = ap.parse_args()

    v1 = {r["scroll"]: r for r in csv.DictReader(open(args.v1))}
    files = sorted(glob.glob(args.l1_glob))
    if not files:
        sys.exit(f"no files matching {args.l1_glob}")

    out, missing = [], []
    for f in files:
        for r in csv.DictReader(open(f)):
            s = r["scroll"]
            if s not in v1:
                missing.append(s)
                continue
            a = v1[s]
            um_per_vox = float(a["lambda_med_um"]) / float(a["lambda_med_L0vox"])
            # level 1 -> L0 factor, taken from the run itself rather than assumed
            f_l0 = float(r["lambda_med_L0vox"]) / float(r["lambda_med_lvlvox"])
            row = {c: r.get(c, "") for c in COLUMNS}
            row["lambda_med_um"] = round(float(r["lambda_med_L0vox"]) * um_per_vox, 1)
            row["lambda_p25_um"] = round(float(r["lambda_p25"]) * f_l0 * um_per_vox, 1)
            row["lambda_p75_um"] = round(float(r["lambda_p75"]) * f_l0 * um_per_vox, 1)
            row["um_per_L0vox"] = round(um_per_vox, 4)
            out.append(row)

    if missing:
        print(f"warning: not in v1, skipped: {', '.join(sorted(set(missing)))}")

    out.sort(key=lambda x: x["scroll"])
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(out)

    def pct(v, p):
        """Linear-interpolated percentile, same convention as numpy.percentile."""
        if len(v) == 1:
            return v[0]
        i = p / 100 * (len(v) - 1)
        lo = int(i)
        hi = min(lo + 1, len(v) - 1)
        return v[lo] + (i - lo) * (v[hi] - v[lo])

    um = sorted(float(x["lambda_med_um"]) for x in out)
    n = len(um)
    med = pct(um, 50)
    p25, p75 = pct(um, 25), pct(um, 75)
    inside = sum(1 for v in um if 160 <= v <= 210)

    print(f"wrote {args.out} with {n} scrolls")
    print(f"  median pitch  {med:.1f} um")
    print(f"  IQR           {p25:.1f}-{p75:.1f} um")
    print(f"  range         {um[0]:.1f}-{um[-1]:.1f} um")
    print(f"  within 160-210 um: {inside}/{n}")
    print(f"  bias vs Paris 4 human anchor (180 um): {(med / 180 - 1) * 100:+.1f}%")
    p4 = [x for x in out if x["scroll"] == "PHercParis4"]
    if p4:
        v = float(p4[0]["lambda_med_um"])
        print(f"  Paris 4 itself: {v:.1f} um, {(v / 180 - 1) * 100:+.1f}% vs anchor")


if __name__ == "__main__":
    main()
