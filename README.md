# winding-ruler

Measurement tools and results for **winding evidence** in the Vesuvius
Challenge spiral-fitting pipeline: how much human winding annotations move
the fit, what the published lasagna signal can and cannot do, and the first
collection-wide measurement of Herculaneum scroll winding geometry.

**Full write-up:** [`docs/SUBMISSION_winding_evidence.md`](docs/SUBMISSION_winding_evidence.md)

## Headline results

- Human winding annotations are **statistically redundant where verified
  patches are dense** and worth **+3–5 pp** where patches thin out
  (PHerc Paris 4, two windows × two seeds, train-free evaluation).
- A calibrated estimator recovers human relative-winding labels at **93%**
  for adjacent pairs — yet three progressively better constraint generators
  all degrade the fit. Root causes identified per generator.

  *Revised 22 Jul 2026.* The write-up attributed the residual to signal
  materialization resolution. That was pre-registered as testable and has
  since been falsified: regenerating the lasagna field at 2x finer scaledown
  does not improve winding accuracy (93.0% coarse vs 88.3% fine at dw=1 over
  20 seeds; unit-step concordance 92.3% on both grids). The bottleneck is
  elsewhere.

- **Winding pitch across the collection: median 187.3 µm (IQR 181.5–193.4),
  35/36 scrolls within 160–210 µm**, independent of scroll size and scan
  campaign. Human-anchored physical pitch ≈175–180 µm.

  *Revised 22 Jul 2026.* The original figures (207 µm, IQR 206–212, 34/35
  within 190–242) were measured at pyramid level 2, which merges adjacent
  sheets: rerunning all 36 scrolls at level 1 lowers the pitch by 10.3% on
  average (36/36, sd 2.6) and drops the bias against the human anchor from
  +15.2% to +4.1%. On Paris 4 itself the level-1 estimate is 182.4 µm, +1.3%
  against the anchor. Current data is `results/atlas_collection_v2.csv`;
  `atlas_collection.csv` is kept as the v1 record. See
  `atlas/build_atlas_v2.py` for the derivation.

![collection atlas](results/winding_atlas_collection_v2.png)

**Gate overlays** — arms (a) vs (c) on the same W2 slice, near-identical to
the eye; the measured +3–5 pp lives in sub-visual local deviations
(pixel divergence 0.6–0.7%/slice, concentrated in the patch-sparse folds):

![gate overlay](results/gate_overlay_s10946.png)

![gate overlay zoom](results/gate_overlay_s10946_zoom.png)

## Layout

    concordance/   pairwise Δw estimators v1–v1.5 + distance-confound test
    generators/    constraint generators v1, v2, v2.1, v3 (all evaluated, all post-mortemed)
    evaluation/    ensemble E1, difficulty D1, train-free evaluator recipe, run tags & seeds
    atlas/         winding_atlas v1.2 — streaming collection-wide pitch measurement
    results/       atlas_collection_v2.csv (current), atlas_collection.csv (v1),
                   figures, per-run satisfaction logs
    docs/          submission write-up + method notes

## Reproducing

Everything runs against public data: the spiral-input PHercParis4 dataset
(HF snapshot 2026-07-15, villa @ 37c37de) and the vesuvius-challenge
open-data S3 bucket (streamed; no bulk download needed for the atlas).
Single consumer GPU (RTX 5070, 12 GB) for the fit experiments; the atlas is
CPU + network only. Fixed seeds throughout; pre-registered criteria noted
inline in each script header.

Coordinate note: the dataset README states (z,y,x); point-collection JSONs
are (x,y,z) in practice; track dbm arrays are (z,y,x).

## Credit

Ray/pitch formulation follows Diego-dcv's technical note §2; per-scroll
profiles complement iyando's stitched-instance profiles (PHerc 1218/0332);
lasagna pipeline provenance confirmed by waldkauz; evaluator-shape guidance
from sean (bruniss).

## Tooling note

I work in Portuguese; English drafting and pair-programming were LLM-assisted, 
as in my previous projects. Measurements, experimental decisions, code, and all 
reported numbers are my own and reproducible from this repo.

## License

MIT — see [LICENSE](LICENSE).
