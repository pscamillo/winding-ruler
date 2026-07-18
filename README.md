# winding-coordinates

**When does the "shifted coordinate" actually separate winding?**

The Vesuvius scroll-unrolling pipeline turns a hard 3D problem into an easy one with a single trick:
a `shifted = r − θ/2π·dr` coordinate that makes *"same turn"* become *"same value."* This project
isolates that trick and measures its exact limit — it works cleanly only when the winding is
**single**, and degrades when the winding is **nested**. Two spirals, one measured boundary.

> 🔗 **[Live interactive visualizer](https://pscamillo.github.io/winding-coordinates/)** — runs in your browser, no install.

![contrast](figures/contrast.png)

---

## The result

Both spirals get the *same* shifted coordinate. The difference is structural, not tuning:

| spiral | winding structure | shifted → winding-snap residual |
|---|---|---|
| **helix** (concentric turns) | **single** | **0.00000** — clean integer bands, one value per turn |
| **phyllotaxis** (sunflower disk) | **nested** (Fibonacci families coexist) | **0.24741** — scatters, best possible family |

The shifted coordinate is **~2×10⁸× cleaner** on single winding than on nested winding.

That gap *is the point*: it explains **why the scroll was tractable** by this technique. A scroll's
turns are concentric — single winding — so one coordinate isolates them. Phyllotaxis has a *nested
tower* of Fibonacci windings (k = 5, 8, 13, 21, 34 all coexist, each tighter than the last), so no
single shifted coordinate can separate them, because there is no single structure to separate.

---

## The trick, and the four techniques

The project reimplements — from scratch, tested — four techniques studied in the Vesuvius
`fit_spiral` pipeline, each an instance of *"guarantee the invariant by construction, don't penalize
violations after the fact."*

1. **Shifted coordinate** — `r − θ/2π·dr` removes the angular ramp so a turn becomes a constant.
2. **Winding unwrap with detached seam detection** — the θ=0 branch cut is stitched by detecting the
   jump (a *discrete* decision, frozen with `.detach()`) and accumulating `dr` (a *continuous*
   quantity, kept differentiable). Freeze the combinatorial, flow the geometric.
3. **Structural invertibility** — `r(θ)` is monotone, so the inverse is exact by construction:
   `forward(inverse(x)) ≈ x` to machine precision (measured error ~1e-13).
4. **Injectivity by construction** — only the golden angle (137.5°, the "most irrational" number)
   keeps florets from colliding. Rational angles collapse into radial arms. This is *structural*,
   not optimized — turn the slider in the visualizer and watch the packing fall apart.

![injectivity](figures/injectivity.png)

---

## Run it

```bash
pip install -r requirements.txt

python test_core.py     # phyllotaxis (nested winding) — 4 tests
python test_helix.py    # helix (single winding) + the contrast test
python figure.py        # regenerate the phyllotaxis figures
python figure_helix.py  # regenerate the contrast figure
```

Everything is CPU, float64, deterministic — the numbers reproduce bit-for-bit.

Internal-state panels (the debugging view — the spiral, the shifted coordinate, the winding
assignment, and the invertibility check, side by side):

![panels](figures/panels.png)

---

## What this is (and isn't)

This is a **reference implementation** of a geometric technique, applied to a clean domain outside
its origin, with the technique's *applicability boundary measured*. The individual mathematics is
classical (Vogel 1979 for the sunflower map; the golden-angle ↔ Fibonacci ↔ continued-fraction
connection dates to van Iterson 1907; phase unwrapping and monotone inverses are textbook). The
contribution here is not new mathematics — it is a clean, tested, visual demonstration of *when a
specific technique works and why*, learned by studying the `fit_spiral` pipeline and rebuilt
independently.

Credit: the techniques were learned by reading the Vesuvius Challenge `fit_spiral` / volume-
cartographer work. The code here is original and shares no source with it.

## License

MIT — see [LICENSE](LICENSE).
