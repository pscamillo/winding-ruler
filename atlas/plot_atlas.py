#!/usr/bin/env python3
"""
plot_atlas.py - render the collection-wide winding pitch figure from a CSV.

The v1 figure was produced ad hoc and never versioned, which is why it could
not be regenerated when the level-1 rerun changed the numbers. This script
takes the atlas CSV as input so the figure always has provenance.

Usage
-----
    python atlas/plot_atlas.py --csv results/atlas_collection_v2.csv \
        --out results/winding_atlas_collection_v2.png

    # to reproduce the v1 figure:
    python atlas/plot_atlas.py --csv results/atlas_collection.csv \
        --out /tmp/v1.png --anchor-note "method bias +15.2%"
"""

import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def pct(v, p):
    """Linear-interpolated percentile (numpy convention)."""
    v = sorted(v)
    if len(v) == 1:
        return v[0]
    i = p / 100 * (len(v) - 1)
    lo = int(i)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (i - lo) * (v[hi] - v[lo])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/atlas_collection_v2.csv")
    ap.add_argument("--out", default="results/winding_atlas_collection_v2.png")
    ap.add_argument("--anchor", type=float, default=180.0,
                    help="human-annotation anchor in um")
    ap.add_argument("--anchor-note", default=None,
                    help="override the bias text under the anchor label")
    ap.add_argument("--level", default=None,
                    help="pyramid level, for the title (read from CSV if absent)")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.csv))
            if r.get("lambda_med_um")]
    data = [(r["scroll"], float(r["lambda_med_um"]), float(r["wraps_med"]))
            for r in rows]
    data.sort(key=lambda x: x[1])
    names = [d[0].replace("PHerc", "") for d in data]
    lam = np.array([d[1] for d in data])
    wr = np.array([d[2] for d in data])
    level = args.level or rows[0].get("level", "?")

    med = pct(lam, 50)
    p25, p75 = pct(lam, 25), pct(lam, 75)
    bias = (med / args.anchor - 1) * 100
    note = args.anchor_note or f"method bias {bias:+.1f}%"
    r_size = float(np.corrcoef(wr, lam)[0, 1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8),
                                   gridspec_kw={"width_ratios": [1.35, 1]})
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(wr.min(), wr.max())
    colors = cmap(norm(wr))

    y = np.arange(len(lam))
    ax1.barh(y, lam, color=colors, edgecolor="k", linewidth=0.3, height=0.72)
    ax1.axvspan(p25, p75, alpha=0.10, color="steelblue", zorder=0)
    ax1.axvline(med, color="steelblue", lw=1.4, ls="--",
                label=f"collection median {med:.1f} \u00b5m "
                      f"(IQR {p25:.1f}\u2013{p75:.1f})")
    ax1.axvline(args.anchor, color="crimson", lw=1.6,
                label=f"Paris4 human-annotation anchor \u2248{args.anchor:.0f} \u00b5m\n({note})")
    ax1.set_yticks(y)
    ax1.set_yticklabels(names, fontsize=7.2)
    ax1.invert_yaxis()
    ax1.set_xlim(0, max(lam) * 1.05)
    ax1.set_xlabel("median winding pitch \u03bb (\u00b5m, physical)")
    ax1.set_title("Winding pitch across the Herculaneum collection\n"
                  f"{len(lam)} scrolls \u00b7 m7 surface predictions \u00b7 "
                  f"centroid-axis radial rays \u00b7 pyramid level {level}",
                  fontsize=10)
    ax1.legend(fontsize=7.4, loc="lower right")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax1, pad=0.01, fraction=0.03)
    cb.set_label("median wraps per ray", fontsize=8)

    ax2.scatter(wr, lam, c=colors, s=52, edgecolor="k", linewidth=0.4)
    ax2.axhspan(p25, p75, alpha=0.10, color="steelblue")
    ax2.axhline(med, color="steelblue", lw=1.2, ls="--")
    ax2.axhline(args.anchor, color="crimson", lw=1.2)
    for n, l, w in data:
        if n in ("PHercParis4", "PHerc0268", "PHerc1218", "PHercMANBp"):
            ax2.annotate(n.replace("PHerc", ""), (w, l), fontsize=7,
                         xytext=(4, 4), textcoords="offset points")
    ax2.set_xlabel("median wraps per ray (scroll size)")
    ax2.set_ylabel("median winding pitch \u03bb (\u00b5m)")
    big = wr >= 20
    r_big = float(np.corrcoef(wr[big], lam[big])[0, 1]) if big.sum() > 2 else r_size
    ax2.set_title(f"Pitch is independent of scroll size\n"
                  f"r = {r_size:.2f} overall, {r_big:.2f} excluding fragments "
                  f"under 20 wraps/ray", fontsize=10)
    ax2.set_ylim(min(lam) * 0.9, max(lam) * 1.06)

    fig.suptitle("Herculaneum Winding Atlas \u2014 collection-wide measurement "
                 "of scroll winding geometry", fontsize=12, y=0.99)
    fig.text(0.5, 0.005,
             "streamed from vesuvius-challenge open-data \u00b7 calibrated against "
             "Paris4 human relative-winding annotations \u00b7 raw m7 predictions "
             "detect ~70\u201385% of sheets (floor estimate)",
             ha="center", fontsize=7, color="0.35")
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.savefig(args.out, dpi=160)

    print(f"wrote {args.out}")
    print(f"  n={len(lam)}  median {med:.1f}  IQR {p25:.1f}-{p75:.1f}  "
          f"range {min(lam):.1f}-{max(lam):.1f}")
    print(f"  bias vs anchor {bias:+.1f}%   pitch-vs-size r = {r_size:.2f}")


if __name__ == "__main__":
    main()
