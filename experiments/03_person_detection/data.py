"""Person annotations, paired synthetic cohorts, and held-out real evaluation data."""

import csv
import hashlib
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def load_coco(annotation_file, image_root):
    document = json.loads(Path(annotation_file).read_text(encoding="utf-8"))
    people = {
        c["id"]
        for c in document["categories"]
        if c["name"].lower() in {"person", "human", "pedestrian"}
    }
    if not people:
        raise ValueError("COCO categories must include person, human, or pedestrian")
    boxes = {}
    for annotation in document["annotations"]:
        if annotation["category_id"] in people:
            boxes.setdefault(annotation["image_id"], []).append(annotation)
    records = []
    for item in document["images"]:
        path = (Path(image_root) / item["file_name"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        width, height = int(item["width"]), int(item["height"])
        with Image.open(path) as image:
            if image.size != (width, height):
                raise ValueError(
                    f"Image dimensions disagree with COCO annotations: {path}"
                )
        annotations = []
        for a in boxes.get(item["id"], []):
            x, y, w, h = map(float, a["bbox"])
            if (
                w <= 0
                or h <= 0
                or x < 0
                or y < 0
                or x + w > width + 1
                or y + h > height + 1
            ):
                raise ValueError(f'Invalid person box in {path}: {a["bbox"]}')
            annotations.append(
                dict(bbox=[x, y, w, h], iscrowd=int(a.get("iscrowd", 0)))
            )
        records.append(
            dict(
                image=str(path),
                width=width,
                height=height,
                annotations=annotations,
                layout=item.get("layout_id"),
                camera=item.get("camera"),
                sha256=digest(path),
            )
        )
    return records


def load_voc(root):
    """Read Sheffield VOC directories, including paired *_xml / *_img folders."""
    root = Path(root)
    images = {}
    for path in root.rglob("*"):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            images.setdefault(path.name.lower(), []).append(path)
    records = []
    for xml in sorted(root.rglob("*.xml")):
        tree = ET.parse(xml).getroot()
        filename = Path(
            tree.findtext("filename", xml.stem + ".jpg").replace("\\", "/")
        ).name
        candidates = images.get(filename.lower(), [])
        if not candidates:
            candidates = [
                p for paths in images.values() for p in paths if p.stem == xml.stem
            ]
        nearby = [p for p in candidates if p.parent.parent == xml.parent.parent]
        if len(nearby) == 1:
            path = nearby[0]
        elif len(candidates) == 1:
            path = candidates[0]
        else:
            raise ValueError(f"Missing or ambiguous RGB image for {xml}")
        with Image.open(path) as image:
            width, height = image.size
        annotations = []
        for obj in tree.findall("object"):
            if obj.findtext("name", "").strip().lower() not in {
                "person",
                "human",
                "pedestrian",
            }:
                continue
            b = obj.find("bndbox")
            # Pascal VOC coordinates are inclusive and one-based.
            x1 = max(0, float(b.findtext("xmin")) - 1)
            y1 = max(0, float(b.findtext("ymin")) - 1)
            x2 = min(width, float(b.findtext("xmax")))
            y2 = min(height, float(b.findtext("ymax")))
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"Invalid VOC person box: {xml}")
            annotations.append(dict(bbox=[x1, y1, x2 - x1, y2 - y1], iscrowd=0))
        records.append(
            dict(
                image=str(path.resolve()),
                width=width,
                height=height,
                annotations=annotations,
                layout=None,
                camera=None,
                sha256=digest(path),
            )
        )
    if not records:
        raise ValueError(f"No annotated images found in {root}")
    return records


def sample(records, count, seed):
    if len(records) < count:
        raise ValueError(f"Need {count} images, found {len(records)}")
    return random.Random(seed).sample(records, count)


def prepare(config, output):
    pools = {
        key: load_coco(config[key]["annotations"], config[key]["images"])
        for key in ["id", "ood"]
    }
    real_spec = config["real"]
    real = (
        load_voc(real_spec["images"])
        if real_spec.get("format") == "voc"
        else load_coco(real_spec["annotations"], real_spec["images"])
    )
    expected = config.get("expected_real_images", 13949)
    if expected is not None and len(real) != expected:
        raise ValueError(f"Expected {expected} real images, found {len(real)}")
    for name, records in [*pools.items(), ("real", real)]:
        if len({r["image"] for r in records}) != len(records):
            raise ValueError(f"Duplicate image paths in {name}")
    sets = {k: {r["sha256"] for r in v} for k, v in {**pools, "real": real}.items()}
    for a, b in [("id", "ood"), ("id", "real"), ("ood", "real")]:
        if sets[a] & sets[b]:
            raise ValueError(f"Image leakage between {a} and {b}")
    budget = config.get("image_budget", 1800)
    if budget % 2:
        raise ValueError("The mixed image budget must be even")
    cohorts = {}
    for seed in config.get("seeds", [0, 1, 2]):
        cohorts[str(seed)] = {
            name: sample(pool, budget, seed) for name, pool in pools.items()
        }
        cohorts[str(seed)]["mixed"] = sample(pools["id"], budget // 2, seed) + sample(
            pools["ood"], budget // 2, seed
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(config=config, cohorts=cohorts, real=real)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    coco = dict(
        info={},
        images=[
            dict(id=i + 1, file_name=r["image"], width=r["width"], height=r["height"])
            for i, r in enumerate(real)
        ],
        categories=[dict(id=1, name="person")],
        annotations=[],
    )
    for i, r in enumerate(real, 1):
        for a in r["annotations"]:
            coco["annotations"].append(
                dict(
                    id=len(coco["annotations"]) + 1,
                    image_id=i,
                    category_id=1,
                    bbox=a["bbox"],
                    area=a["bbox"][2] * a["bbox"][3],
                    iscrowd=a["iscrowd"],
                )
            )
    (output / "real.coco.json").write_text(json.dumps(coco), encoding="utf-8")
    return manifest


def summarize(results, output, seeds):
    import numpy as np

    rows = []
    for model in sorted({r["model"] for r in results}):
        row = {"model": model}
        for condition in ["pretrained", "id", "ood", "mixed"]:
            group = [
                r
                for r in results
                if r["model"] == model and r["condition"] == condition
            ]
            expected = {None} if condition == "pretrained" else set(seeds)
            if len(group) != len(expected) or {r["seed"] for r in group} != expected:
                raise ValueError(
                    f"Incomplete or duplicate results for {model}/{condition}"
                )
            scores = [100 * r["ap"] for r in group]
            row[condition] = float(np.mean(scores))
            if condition != "pretrained":
                row[condition + "_std"] = (
                    float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
                )
        rows.append(row)
    output = Path(output)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "| Model | Pretrained | ID | OOD | Mixed |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        cells = [r["model"], f'{r["pretrained"]:.2f}'] + [
            f'{r[k]:.2f} ± {r[k+"_std"]:.2f}' for k in ["id", "ood", "mixed"]
        ]
        lines.append("| " + " | ".join(cells) + " |")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows
