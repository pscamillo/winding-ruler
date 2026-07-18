# Train-free evaluation: measure any geometry against any constraint set

The spiral fitter can be resumed at its final step, producing metrics without
training. This turns "how well does geometry X satisfy constraint set Y" into
a ~10-minute operation:

    export FIT_SPIRAL_RESUME_PATH=<run_dir>/checkpoint_fitted.ckpt
    export FIT_SPIRAL_RESUME_STEP=30000    # = num_training_steps
    python fit_spiral.py                   # with pcl_json_paths = set Y
    unset FIT_SPIRAL_RESUME_PATH FIT_SPIRAL_RESUME_STEP

Read the `satisfied_*` block at the end. Two failure modes to check every run:

1. **Validity line.** stdout must contain `resuming from ... at iteration
   30000`. If RESUME_PATH is unset/wrong (e.g. an unexpanded placeholder),
   the fit silently evaluates a randomly initialized model — no error raised.
2. **Constraint loading.** Count the `Loaded point collection` lines against
   your pcl_json_paths; a missing file loads as empty without error.

All arm evaluations in this repo used identical configs except pcl_json_paths,
same window, same denominators (the fit's own counts), 2 seeds per arm.
Early poison detector for generated constraints: run a 500-step smoke first
and read in-sample `satisfied_unattached_pcls` (our v1: 0.1%, v2: 15%,
v3: 35% — eval outcomes ranked in the same order).
