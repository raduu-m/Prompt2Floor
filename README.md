# From Prompt to Floor — experiments

Experiment contributions for scene generation, navigation/motion evaluation,
and person detection.

The scene compiler, assets, solver, and base simulation framework are provided
by [worldc](https://github.com/SheepSeb/worldc), maintained separately.

| Contribution | Folder |
| --- | --- |
| Generation experiment runner | `experiments/01_scene_generation/` |
| Navigation integration and motion analysis | `experiments/02_navigation_motion/` |
| Person-detection benchmark | `experiments/03_person_detection/` |

## Setup

Install worldc using its repository instructions. For the navigation extensions,
use the dependency revision in `integration/worldc.json` and apply the included
patch from this repository's root:

```powershell
git clone https://github.com/SheepSeb/worldc.git ../worldc
git -C ../worldc checkout d0ae72495ef5bbb0b82065f25f3c21dff86c3130
git -C ../worldc apply (Resolve-Path integration/worldc.patch).Path
pip install -r requirements.txt
```

Pass `--worldc-root ../worldc` to scripts that use the simulator. Person detection
uses its own requirements file and can run independently of worldc.
Each experiment folder contains its commands.

[Scene dataset](https://huggingface.co/datasets/RaduMoldoveanu/From-Prompt-to-Floor-Dataset)
