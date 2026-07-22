# Silent data loss in predict3d: finding it, reporting it, and what the pre-registered experiment found

**Paulo Sergio Camillo (pscamillo) — July 2026 progress submission, second entry.**
Code & data: https://github.com/pscamillo/winding-ruler ·
Issue: [ScrollPrize/villa#1183](https://github.com/ScrollPrize/villa/issues/1183) ·
Fix (by another contributor): [#1192](https://github.com/ScrollPrize/villa/pull/1192)

---

## TL;DR

While running the experiment my July submission pre-registered, a feature that
scores 85.6% on intact data collapsed to 0.0%. I spent a day looking for the
bug in my own code. It was in the data: `predict3d` can silently drop z-slices
inside a chunk, with every chunk file present, every tile reported as
processed, and no warning.

I reported it with a two-command reproducer. Within a day another contributor,
@NanokodasKarolis, found the root cause and opened a fix. The scanner I built
to detect the failure was used to validate that fix, by them and by me.

The pre-registered experiment itself failed, with margin. Finer signal
resolution does not improve winding accuracy. That route is closed.

---

## 1. How it surfaced

My July submission pre-registered an experiment: regenerate the lasagna fields
at 2× finer scaledown, then re-run the identical concordance suite. Success
criterion declared in advance, Δw=1 ≥ 96%.

The fine-field run came back at 44.2% against 93.3% on the published field. One
feature — the raw integral of `grad_mag` along the ray between two annotated
points — had gone from 85.6% to **0.0%**.

0.0% is not noise. A feature carrying no information lands near the class
frequency; 0.0% means the fit never emits the right answer at all. Something
was structurally wrong.

I worked through four hypotheses and killed all four by measurement: an
absolute threshold on decoded values, a fixed sampling window, a grid-scale
error between pyramid levels, and an empty field. The field was there —
25.1% nonzero in the slab the estimator reads.

The fifth check was the one that mattered. Of 706 annotated pairs, 8.9% had
rays crossing voxels that were zero, and 4.7% were zero along their entire
length. In a least-squares fit those are leverage points: they flatten the
slope and lift the intercept until the prediction collapses onto a constant.
Excluding them, the feature returns to 84.6%, against 85.6% on the intact
field.

The zeros were not anatomy. They were missing output.

## 2. The bug

**Symptom.** 16 of the 32 z-slices inside an output chunk come out zero. Every
chunk file is present, every tile reports as processed, no warning is emitted.
Only `grad_mag` was affected in the case I hit; `nx`, `ny` and `cos` were
correct in the same slices of the same run.

**Minimal reproducer.** Same input, same checkpoint, same machine, same z0.
Only the crop depth differs.

    lasagna-preprocess predict3d --input paris4.zarr/2 \
      --unet-checkpoint lasagna_2um_ff85612.pt \
      --crop 0 0 40000 32693 32693  640 --cos-scaledown 2 --scaledown 2   # clean

    lasagna-preprocess predict3d --input paris4.zarr/2 \
      --unet-checkpoint lasagna_2um_ff85612.pt \
      --crop 0 0 40000 32693 32693 1280 --cos-scaledown 2 --scaledown 2   # loses z 5040-5055

The shallow run covers the same z range correctly, which proves the data exists
and the model produces it.

**Ruled out by measurement, not by argument:**

- *Crop alignment.* `z0 40000` and `z0 39936` have different chunk alignment
  and lose the same absolute range.
- *Missing input.* Level-2 input chunk-row 78 has 1289 chunk files, more than
  its neighbours.
- *Accumulator pressure.* The accumulator spans the full output in every run,
  holed or clean.

**Characterisation across seven runs at total scaledown 8:** the lost range is
always the second half of the second output chunk row, 16 of 32 slices. The two
clean runs are the shallow ones.

| crop z0 | depth | out z0 | 2nd row | lost |
|---|---|---|---|---|
| 32000 | 2368 | 4000 | 126 | 4048–4063 |
| 40000 | 2368 | 5000 | 157 | 5040–5055 |
| 42048 | 1952 | 5256 | 165 | 5296–5311 |
| 40000 | 1280 | 5000 | 157 | 5040–5055 |
| 39936 | 1280 | 4992 | 157 | 5040–5055 |
| 40000 | 640 | 5000 | 157 | clean |
| 39936 | 640 | 4992 | 157 | clean |

I could not isolate the mechanism. The issue was filed as symptom, reproducer,
and what had been ruled out — including one hypothesis of mine that turned out
to be wrong.

## 3. The detector

The QA I had been running counted chunk files, which is the right check for the
failure mode reported earlier in villa#1114: chunks elided entirely. It is
blind to this one, because the files are all present. It had passed the
corrupted output.

`concordance/qa_holescan.py` checks content instead. The hard part is telling
"zero because the writer dropped it" from "zero because there is no papyrus
there" — most of a scroll volume is legitimately empty. It uses cross-channel
comparison: `grad_mag`, `nx`, `ny` and `cos` share support, so a z-slice where
one channel is empty and another is not is a defect rather than mask. No
threshold, no per-scroll calibration. It exits nonzero, so it can gate a
pipeline.

Two limitations, both worth stating:

- The check must run **before any normalisation**. z-scoring moves raw zeros to
  some nonzero value and the evidence disappears — a point raised independently
  in villa#1173.
- Cross-channel comparison cannot see a defect that zeroes all four channels at
  the same z. It caught this bug because the affected z differs per channel.
  @NanokodasKarolis's byte-comparison against an unpatched run is the stronger
  check, and it exposed this limitation by finding damage in `nx`/`ny` that my
  issue had not reported.

Running it over the outputs I already had turned up a hole in a calibration
field I had used downstream and believed clean.

## 4. What happened after the report

The issue was filed on 20 July. Within a day @NanokodasKarolis — who had not
worked on this repository before — reproduced it from the two commands, found
the root cause, and opened #1192.

The cause: `predict3d` treated a completed z-band in a `(C, Z, Y, X)` memmap as
one contiguous byte range. The array is channel-major, so the hole-punch
released pages belonging to other channels and other slices. That explains the
channel specificity, why depth was the trigger, and why crop alignment was
irrelevant — all three of which the issue reported but could not account for.
It also corrects my guess in the issue, which pointed at the wrong line.

I validated the fix on my reproducer: the 1280 case passes, and the 640 output
is byte-identical to the unpatched run across all four channels, so the fix does
not disturb output that was already correct. They ran the scanner against a
deeper stress case and the default scaledown, on both patched and unpatched
builds, and archived the outputs.

At the time of writing the PR is open and awaiting maintainer review.

## 5. The pre-registered experiment

With the ruler repaired, the experiment could be answered.

**Pre-registered criterion:** ensemble Δw=1 ≥ 96% on virgin splits.
**Result: 91.3% / 89.8% on two seeds, 88.3% over 20 seeds. Fails with margin.**

An integrity filter was declared before the numbers were seen — exclude any
pair with more than 20% of its ray samples in zero voxels, applied identically
to both grids. It excluded zero pairs on both sides, so no number below is
filtered.

Stratifying Δw=1 pairs by A–B distance (which, at unit winding step, is the
local sheet spacing), 20 seeds, ~500 test pairs per bin:

| distance (vox) | coarse | fine | delta |
|---|---|---|---|
| 0–13 | 99.4 | 98.0 | −1.4 |
| 13–18 | 98.2 | 97.8 | −0.4 |
| 18–23 | 91.7 | 88.8 | −2.8 |
| >23 | 82.7 | 68.2 | −14.5 |
| all Δw=1 | 93.0 | 88.3 | −4.8 |

The dense regime has no headroom: the coarse grid is already at 99.4% where
sheets are closest. The residual error lives in Δw=1 pairs that are far apart,
and there the finer field is clearly worse. Across quartiles the median
distance goes 11.4 → 27.5 and the median integral 0.298 → 0.364 while the true
label stays at 1, so the fit overpredicts on far pairs. This is the distance
confound: distance predicts winding difference well globally and badly at fixed
Δw=1.

Unit-step concordance is 92.3% on **both** grids, delta 0 — a second,
methodologically independent measurement pointing the same way.

**Signal resolution is not the bottleneck.** The route my July submission
pre-registered — a constraint generator fed by a finer version of this signal
family — is closed. That was reported to the field owner with numbers, as
promised when the experiment was proposed.

## 6. What is reusable

- **`concordance/qa_holescan.py`** — content-level detector for silent slice
  loss in `predict3d` output. One file, numpy and zarr only, exits nonzero on a
  hole. Currently the check I would run before consuming any lasagna field.
- **The reproducer pattern.** The pair of commands differing in one parameter,
  one failing and one passing, is what let someone reproduce a
  crop-dependent bug on the first try. A single failing case would not have.
- **The negative result.** Anyone considering finer materialisation of this
  signal family to improve winding constraints now has the measurement instead
  of the intuition. Two independent estimators, twenty seeds, stratified by
  regime.
- **A caution.** Silent zeros in a field are indistinguishable from absence of
  material, and they poison least-squares fits through leverage rather than
  through noise. A metric collapsing to exactly 0.0% is a data integrity
  signal, not a modelling one.

## 7. Credit

@NanokodasKarolis found the root cause and wrote the fix, on their first
contribution to this repository. @waldkauz published the checkpoint that made
the experiment possible and asked the question about dense regions that led to
the stratified table in §5. @iyando's independent cross-check on PHerc1218
prompted a separate correction to the July atlas, published as an errata in the
same repository.
