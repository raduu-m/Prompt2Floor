"""Prepare, train and evaluate paired person-detection experiments."""

import argparse
import hashlib
import json
from pathlib import Path

from data import prepare, summarize


def jobs(manifest, models=None, seeds=None):
    config = manifest["config"]
    selected = [m for m in config["models"] if not models or m["name"] in models]
    if not selected or models and set(models) - {m["name"] for m in selected}:
        raise ValueError("Unknown model selection")
    paired = config["seeds"] if seeds is None else seeds
    if set(paired) - set(config["seeds"]):
        raise ValueError("Unknown seed selection")
    return [
        (m, seed, condition)
        for m in selected
        for seed in paired
        for condition in ["id", "ood", "mixed"]
    ]


def train(manifest, root, models=None, seeds=None):
    from backends import train_rtdetr, train_yolo

    signature = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    for spec, seed, condition in jobs(manifest, models, seeds):
        folder = root / spec["name"] / f"seed-{seed}" / condition
        record = folder / "run.json"
        if record.is_file():
            existing = json.loads(record.read_text())
            if (
                existing["manifest_sha256"] != signature
                or not Path(existing["checkpoint"]).exists()
            ):
                raise ValueError(f"Checkpoint or configuration changed: {folder}")
            continue
        folder.mkdir(parents=True, exist_ok=False)
        backend = train_yolo if spec["backend"] == "yolo" else train_rtdetr
        checkpoint = backend(
            spec,
            manifest["cohorts"][str(seed)][condition],
            folder,
            seed,
            manifest["config"],
        )
        record.write_text(
            json.dumps(
                dict(
                    model=spec["name"],
                    seed=seed,
                    condition=condition,
                    checkpoint=checkpoint,
                    manifest_sha256=signature,
                ),
                indent=2,
            ),
            encoding="utf-8",
        )


def evaluate(manifest, root, models=None, seeds=None):
    from backends import predict, coco_ap

    planned = jobs(manifest, models, seeds)
    specs = {m["name"]: m for m, _, _ in planned}
    evaluation = [(m, None, "pretrained") for m in specs.values()] + planned
    results = []
    for spec, seed, condition in evaluation:
        folder = (
            root
            / spec["name"]
            / ("pretrained" if seed is None else f"seed-{seed}/{condition}")
        )
        checkpoint = (
            None
            if seed is None
            else json.loads((folder / "run.json").read_text())["checkpoint"]
        )
        folder.mkdir(parents=True, exist_ok=True)
        predictions = predict(spec, checkpoint, manifest["real"], manifest["config"])
        (folder / "predictions.json").write_text(
            json.dumps(predictions), encoding="utf-8"
        )
        metrics = dict(
            model=spec["name"],
            seed=seed,
            condition=condition,
            **coco_ap(root / "real.coco.json", predictions),
        )
        (folder / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        results.append(metrics)
    (root / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    summarize(results, root, manifest["config"]["seeds"] if seeds is None else seeds)
    return results


def compare(manifest, root, output, model, seed, image_ids, threshold):
    from PIL import Image, ImageDraw, ImageFont

    output.mkdir(parents=True, exist_ok=True)
    if not 0 <= threshold <= 1:
        raise ValueError("Confidence must lie in [0,1]")
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    for condition in ["pretrained", "id", "ood", "mixed"]:
        folder = (
            root
            / model
            / (
                "pretrained"
                if condition == "pretrained"
                else f"seed-{seed}/{condition}"
            )
        )
        predictions = json.loads((folder / "predictions.json").read_text())
        for image_id in image_ids:
            if not 1 <= image_id <= len(manifest["real"]):
                raise ValueError("Image ID is outside evaluation set")
            image = Image.open(manifest["real"][image_id - 1]["image"]).convert("RGB")
            draw = ImageDraw.Draw(image)
            for p in predictions:
                if p["image_id"] == image_id and p["score"] >= threshold:
                    x, y, w, h = p["bbox"]
                    draw.rectangle((x, y, x + w, y + h), outline="#18d89a", width=3)
                    draw.text(
                        (x, max(0, y - 20)),
                        f'person {p["score"]:.2f}',
                        fill="#18d89a",
                        font=font,
                    )
            target = output / f"image-{image_id}"
            target.mkdir(exist_ok=True)
            image.save(target / f"{condition}.jpg", quality=96)


def read_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    for source in ["id", "ood", "real"]:
        for name in ["images", "annotations"]:
            if name in config[source]:
                config[source][name] = str(
                    (path.parent / config[source][name]).resolve()
                )
    if len(config.get("seeds", [0, 1, 2])) != len(set(config.get("seeds", [0, 1, 2]))):
        raise ValueError("Seeds must be distinct")
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["prepare", "train", "evaluate", "run", "compare"]
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path("outputs/detection"))
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--model", default="RT-DETRv2-S")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-ids", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    root = args.work_dir.resolve()
    if args.command in ["prepare", "run"]:
        if not args.config:
            parser.error("--config is required")
        manifest = prepare(read_config(args.config), root)
    else:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if args.command in ["train", "run"]:
        train(manifest, root, args.models, args.seeds)
    if args.command in ["evaluate", "run"]:
        evaluate(manifest, root, args.models, args.seeds)
    if args.command == "compare":
        compare(
            manifest,
            root,
            args.output or root / "comparison",
            args.model,
            args.seed,
            args.image_ids,
            args.confidence,
        )


if __name__ == "__main__":
    main()
