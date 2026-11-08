# Navigation and motion evaluation

Apply `integration/worldc.patch` to the dependency checkout before navigation
runs. The patch contains the navigation orchestration, contact/physics extensions,
and changes to scene eligibility and reporting; the base simulator stays in worldc.

```sh
python experiments/02_navigation_motion/run_navigation.py --worldc-root ../worldc select DATASET --n 10 --out outputs/selection
python experiments/02_navigation_motion/run_navigation.py --worldc-root ../worldc run outputs/selection/selection.json --out outputs/navigation
python experiments/02_navigation_motion/motion_analysis.py --worldc-root ../worldc --scenes DATASET --magni THOR_MAGNI/CSVs_Scenarios --out outputs/motion
```

`motion_analysis.py` compares generated worker paths with THOR-MAGNI helmet
centroids using common smoothing and non-overlapping eight-second windows.
It exports normalized speed variability, curvature, summary tables and plots.
Native simulation entry points are available through the runner's `isaac` mode.
