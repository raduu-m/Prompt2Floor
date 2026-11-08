# Scene-generation experiment

Run the paired prompt/repair protocol through the external worldc installation:

```sh
python experiments/01_scene_generation/run_generation.py --worldc-root ../worldc --models MODEL_ID --out outputs/scenes
```

The runner uses worldc's prompt definitions and generation API. `--variant plain`
selects plain prompts; `--models` accepts multiple models and `ollama:MODEL`
provider identifiers. Generation settings and logs are saved with the outputs.
Use `--dry-run` to inspect commands. Configure credentials in the environment.

Layout diversity, adherence, and plausibility analysis use worldc's
`scripts/e2_prelim.py` and evaluation tools.
