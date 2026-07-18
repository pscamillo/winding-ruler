# Winding Evidence, Measured: from annotation value to collection-wide geometry

**Paulo Sergio Camillo (pscamillo) — July 2026 progress submission**
Code & data: https://github.com/pscamillo/winding-ruler · All experiments on a single RTX 5070 (12 GB) + streamed open-data.

---

## TL;DR

The winding-constraints page says: *"We don't know the exact minimum amount of
winding evidence necessary for a given scroll ahead of time."* This submission
answers that quantitatively for PHerc Paris 4, maps what the published lasagna
signal can and cannot do, and delivers the first collection-wide measurement of
scroll winding geometry (35 scrolls). Headline numbers:

- Human winding annotations are **statistically redundant where verified
  patches are dense** (fit lands in the same place without them, 2 seeds each
  way) and worth a real **+3–5 pp** where patches thin out. "How much
  annotation" is less about density and more about **where**.
- A calibrated estimator recovers human relative-winding labels at **93%** for
  adjacent pairs (validated on virgin splits) — but three progressively better
  constraint generators built on it all **degrade** the fit (52% vs 73.7%
  no-constraint baseline). Root causes identified for each; the bottleneck is
  the materialized signal resolution, not the estimator.
- **Winding pitch across the Herculaneum collection: median 207 µm, IQR
  206–212, 34/35 scrolls within 190–242 µm** — independent of scroll size
  (16–113 wraps, r = −0.21) and of scan campaign (2.4–9.4 µm, 78–116 keV).
  Anchored to the Paris 4 human annotations, the physical pitch is ≈175–180 µm.

## 1. Motivation

My original goal was practical rather than academic: to make automatically
generated winding constraints work. That immediately raised a more fundamental
question — what information is already present in the published signal, and
where do human annotations actually move the fit? This submission is the
result of following that question wherever the measurements led.

The spiral fitter consumes winding evidence from four sources (abs, relative,
same-winding point collections + patch overlap). Its own docs list desiderata
for any generator of such evidence: **accurate (or measurable confidence),
fast enough for entire volumes, easy to verify, general enough for global
fitters**. Everything below is organized against those four criteria.

## 2. How much do human annotations move the spiral? (the gate)

**Method.** Two z-windows of Paris 4 chosen by an annotation-density census
(annotations anti-correlate with patch area across the scroll): W1
z 8000–9000 (max patches, 1067 patches / 311 cm²) and W2 z 10000–11000
(adversarial, ann/area 12.2×). Two arms: (a) full human constraints,
(c) patch-overlap only. Two seeds per arm. Evaluation is **train-free**:
resume any checkpoint at step 30000 against any constraint set — geometry X
measured on ruler Y in ~10 min, no retraining. Denominators are the fit's own
(W1: 867 pcls / 10,891 pts; W2: 456 / 7,673).

**Results.**

| window | arm | satisfied pcls (%) | satisfied pcl points (%) |
|---|---|---|---|
| W1 (patch-dense) | (a) human | 83.7 | 82.2 |
| W1 | (c) none → eval | 81.3 / 83.3 | 80.8 / 80.8 |
| W2 (patch-sparse) | (a) human | 77.9 / 80.3 | 76.6 / 76.5 |
| W2 | (c) none → eval | 75.0 / 72.4 | 74.1 / 72.7 |

W1: spread between seeds (2.0 pp) swallows the arm difference — annotations
redundant. W2: complete separation (worst (a) > best (c) on both metrics;
Δ ≥ 2× seed spread) — annotations worth +3–5 pp exactly where patches thin.
Side observation: in W2, arm (a) trades patch satisfaction for pcl
satisfaction, suggesting a systematic error in one of the signals there.

The result suggests that annotation effort should be allocated spatially
rather than uniformly. Human winding evidence provides little additional value
in patch-dense regions, but becomes consistently useful where patch coverage
thins out. Measuring where annotations matter may be more important than
simply collecting more of them.

## 3. Anatomy of the winding signal (what 93% does and does not buy)

All numbers on virgin collection-level splits, calibration on train only.

- **Unit step (Δw = 1, magnitude + sign): 91–93%** across three virgin seeds
  and two estimator variants. Sign via umbilicus radial geometry: **100%**
  (650/650 test pairs; winding increases outward, learned from train with 100%
  agreement). Calibration k stable (2.77–3.03) across windows and splits.
- **Distance confound, decomposed:** a distance-only ruler scores 57.1%
  global / 83.6% at Δw=1. The field integral adds **+7 pp of genuine winding
  information** — real but thin.
- **Human λ (winding spacing): 18.76 working-grid voxels ≈ 180 µm** in W2
  (least-squares over 706 annotated pairs).
- **Ensemble labeler (E1):** linear combination of {distance, raw integral,
  cos-weighted integral, ray–normal alignment, local patch density} reaches
  **75–79% global / 93% at Δw=1**, stable on two virgin seeds
  (pre-registered criterion passed twice). Ablation:

| features | global % | Δw=1 % |
|---|---|---|
| distance | 52.9–54.8 | 61–70 |
| + integral | 70.7–73.4 | 87.5–91.3 |
| + cos + alignment | 75.3–77.7 | 93.2–93.3 |
| + patch density | 78.5–79.1 | 91.3–93.2 |

Independent physical sources add; features derived from the same volume
saturate. (A radius feature was unstable across seeds and dropped.)

## 4. Why 93% is not enough to generate: three generators, three root causes

All three trained arm-(b) style (patch-overlap + generated constraints,
30k steps) and evaluated train-free against the human ruler in W2.
Baseline (c) no-constraints = 73.7%.

| generator | design | eval | root cause |
|---|---|---|---|
| v1 | radial march, emit at integer crossings (847k pts) | 45.4 | anchors land off-sheet (1.3% in-sample satisfaction); slant inflation overcounts 1.8× (emission every 10.4 vx vs true λ 18.8) |
| v2 | track-point endpoints, integral labels (43k pairs) | 43.0 | anchors fixed (25% in-sample) but Δw=2/3 labels (59% of volume) poison; slant/skip indistinguishable at this resolution |
| v3 | ensemble labels + domain filter (align & distance within human p5–p95) + Δw=1 only + residual ≤ 0.25 (3.8k pairs) | 52.0 | best in-sample ever (66.5%) and emission spacing matches physics (21.5 vx ≈ λ) — yet ~1.3k wrong rigid pairs outweigh ~2.5k sparse correct ones |

Progression 1.3% → 25% → 66.5% in-sample shows each fix worked; the eval
plateau shows the ceiling is the signal, not the algorithm. Two auxiliary
negatives close side doors: cos-weighting with the published normals is a
global rescale, not a geometric correction (|cos| ≈ 0.81 uniform); and the
generator's ambiguity does **not** predict where annotations are needed
(difficulty-map AUC 0.46/0.48 — saturated ~0.80 across the whole window).

**Transferable lesson for any auto-annotator:** pairwise concordance measured
on perpendicular human clicks does not transfer to arbitrary sampling
geometries, and anchor placement is a separate failure axis that concordance
never tests. Early poison detector that would have saved us a GPU-day: 500-step
smoke run, read in-sample `satisfied_unattached_pcls`.

**Why the signal is the ceiling:** the lasagna fields are materialized at
data_level 4 (16× the native pipeline grid; confirmed in the pyramid writer) —
sheets sit ~4.7 lasagna voxels apart. The pipeline that produced them
(`lasagna/preprocess_cos_omezarr.py`, confirmed by the team) takes
`--scaledown` and `--crop-xyzwhd` as first-class flags.

## 5. Winding Atlas: first collection-wide measurement of winding geometry

**Method.** For every sample in open-data with an m7 surface prediction
(auto-discovered, latest per scroll): stream the level-2 pyramid over HTTP
(98–830 MB/scroll, 6.2 GB total), 16 slices × 64 radial rays from the
per-slice mask centroid (no umbilicus exists outside Paris 4), sheets counted
as mask runs, spacing = gaps between run centers. Converted to physical µm via
the bucket `metadata.json` (render volume_id parsed from the zarr name →
`pixel_size_um`), normalizing heterogeneous render bases (m7-L0 vs m7-L2 —
naive same-pyramid-level comparisons are wrong by 4×).

**Calibration against ground truth.** Gap-closing chosen by the Paris 4 human
anchor (λ = 4.69 level-voxels): gap_close=0 passes (5.5 measured, wraps 89 vs
100–130 true); any closing fuses sheets at this resolution. Declared method
bias +17%; estimator quantization ±9 µm. Raw m7 predictions detect ~70–85% of
sheets (floor estimate) — independently corroborated on PHerc1218: our
raw-prediction count (median 60) vs iyando's stitched-instance plateau
(37–46) quantifies what stitching merges/recovers.

**Result.** *(figure: winding_atlas_collection.png)*
**35 scrolls with full statistics: median pitch 207 µm, IQR 206–212, 34/35
within 190–242 µm.** Independent of scroll size (16–113 median wraps,
r = −0.21) and of scan campaign (2.4/8.64/9.36 µm; 78–116 keV). Anchored to
the human ground truth: **physical pitch ≈175–180 µm, remarkably uniform
across the collection.** Per-scroll table in `atlas_collection.csv`; the
wraps column doubles as a size census of the collection (largest: PHerc0268,
113 median wraps, also the only pitch outlier at 242 µm).

Immediate practical use: the fit's `initial_dr_per_winding` is currently a
guessed constant; the atlas replaces it with a measured per-scroll prior for
all 13 Grand Prize volumes and beyond. Credit: the (z,θ) ray formulation
follows Diego-dcv's technical note §2; per-scroll profiles complement
iyando's stitched-instance profiles.

## 6. Against the four criteria

- **Accurate / measurable confidence:** unit-step 93% with per-pair residual
  confidence; generation accuracy insufficient at the published resolution —
  measured, not assumed.
- **Fast:** ruler labels 123k candidate pairs in ~4 min CPU; atlas covers the
  collection in ~1 CPU-afternoon of streaming.
- **Easy to verify:** everything above is reproducible from open data +
  linked scripts; train-free evaluator makes any geometry×constraints check a
  10-minute operation.
- **General:** atlas runs on any scroll with an m7 prediction (35 today);
  generator generality is exactly what the pre-registered next experiment
  tests.

## 7. Pre-registered next experiment (August)

Regenerate the lasagna fields for the W2 crop at `--scaledown 2` (level-2
materialization, 8× finer than published) using the team's own
`preprocess_cos_omezarr.py` (checkpoint publication in progress per the team),
then re-run the identical concordance suite and generator v3.
**Success criterion, declared now:** ensemble Δw=1 ≥ 96% on virgin splits AND
arm-(b) eval ≥ 77% (parity with human annotations). Failure closes the
generation route at any resolution of this signal family; success reopens it
with the same 3.8k-pair architecture. Cost: hours on one consumer GPU.
The experiment is intentionally identical except for signal resolution. No
changes to the generator architecture are allowed.

## 8. Data & code

- Dataset: spiral-input PHercParis4, HF snapshot 2026-07-15 (pre coverage-fix
  of 07-17; 77 segments republished since — internal comparisons unaffected).
  villa @ 37c37de. Coordinate note for reproducers: the dataset README states
  (z,y,x); point-collection JSONs are (x,y,z) in practice (validated against
  the fit's own loaded counts); track dbm arrays are (z,y,x).
- Scripts: https://github.com/pscamillo/winding-ruler — organized as
  `concordance/` (v1–v1.5 + distance-confound test), `generators/`
  (v1/v2/v2.1/v3), `evaluation/` (ensemble E1, difficulty D1, train-free
  evaluator recipe, run tags & seeds), `atlas/` (winding_atlas v1.2),
  `results/` (atlas_collection.csv, figures, per-run satisfaction logs),
  `docs/` (this document + method notes).
- Every number above: fixed seeds, pre-registered criteria (including the two
  that killed our own preferred hypotheses), virgin-split confirmation.

## 9. Takeaway

The measurements suggest that winding evidence is not a binary question of
"more annotations versus fewer annotations", but a spatial one. Human
annotations are most valuable where the geometry is least constrained by
patch evidence, and that value can now be measured. The collection-wide
winding atlas additionally provides a physically grounded prior for scroll
geometry that can be used immediately by existing fitting pipelines. Finally,
the published lasagna signal appears to sit at an interesting boundary: rich
enough to recover human labels with high accuracy, but insufficient for fully
automatic constraint generation at its current resolution. Whether that
limitation is fundamental or simply one of materialization resolution is the
next experiment.

*Thanks: sean (bruniss) for the evaluator-shape guidance, waldkauz for the
pipeline provenance, Diego-dcv and iyando for the converging ray/pitch
formulation, Ben (Hari Seldon) for the priors framing.*
