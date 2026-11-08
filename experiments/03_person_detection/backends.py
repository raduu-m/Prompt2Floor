"""Head-only YOLO and decoder/prediction-layer RT-DETRv2 training backends."""

import json
import os
import random
import shutil
from pathlib import Path


def set_seed(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_device(config):
    device = str(config.get("device", "0"))
    return "cuda:" + device if device.isdigit() else device


def yolo_dataset(records, folder, names):
    import yaml

    folder = Path(folder).resolve()
    for name in ["images", "labels"]:
        (folder / name).mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(records):
        source = Path(r["image"])
        target = folder / "images" / f"{i:06d}{source.suffix.lower()}"
        if not target.exists():
            try:
                os.link(source, target)
            except OSError:
                shutil.copyfile(source, target)
        lines = []
        for a in r["annotations"]:
            if a.get("iscrowd"):
                continue
            x, y, w, h = a["bbox"]
            lines.append(
                f'0 {(x+w/2)/r["width"]:.9f} {(y+h/2)/r["height"]:.9f} {w/r["width"]:.9f} {h/r["height"]:.9f}'
            )
        (folder / "labels" / f"{i:06d}.txt").write_text(
            "\n".join(lines), encoding="utf-8"
        )
    path = folder / "dataset.yaml"
    path.write_text(
        yaml.safe_dump(
            dict(path=str(folder), train="images", val="images", names=names)
        ),
        encoding="utf-8",
    )
    return path


def train_yolo(spec, records, folder, seed, config):
    from ultralytics import YOLO

    set_seed(seed)
    model = YOLO(spec["checkpoint"])
    data = yolo_dataset(records, Path(folder) / "data", model.names)
    # Preserve COCO's class layout; person is class zero. Only the final Detect module is trained.
    frozen = list(range(len(model.model.model) - 1))

    def freeze_statistics(trainer):
        for index in frozen:
            trainer.model.model[index].eval()

    model.add_callback("on_train_epoch_start", freeze_statistics)
    options = dict(config.get("yolo_train", {}))
    options.update(
        data=str(data),
        epochs=config.get("epochs", 3),
        seed=seed,
        freeze=frozen,
        val=False,
        patience=0,
        plots=False,
        project=str(Path(folder).resolve()),
        name="fit",
        exist_ok=False,
        device=config.get("device", "0"),
        batch=config.get("batch_size", 8),
        imgsz=config.get("image_size", 640),
    )
    model.train(**options)
    checkpoint = Path(folder) / "fit/weights/last.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return str(checkpoint)


def load_rtdetr(spec, checkpoint=None):
    os.environ.setdefault("USE_TF", "0")
    from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

    source = checkpoint or spec["checkpoint"]
    options = (
        {} if checkpoint or not spec.get("revision") else {"revision": spec["revision"]}
    )
    processor = RTDetrImageProcessor.from_pretrained(source, **options)
    model = RTDetrV2ForObjectDetection.from_pretrained(source, **options)
    return model, processor


def freeze_rtdetr(model):
    trainable = []
    for name, p in model.named_parameters():
        enabled = (
            ".decoder." in name
            or "class_embed" in name
            or "bbox_embed" in name
            or "enc_score_head" in name
            or "enc_bbox_head" in name
        )
        p.requires_grad_(enabled)
        if enabled:
            trainable.append(name)
    if not trainable:
        raise ValueError("No RT-DETR decoder or prediction parameters found")
    return trainable


def rtdetr_training_mode(model):
    # Frozen feature/projection batch norms must retain their pretrained statistics.
    model.eval()
    model.training = True
    model.model.training = True
    model.model.decoder.train()


def train_rtdetr(spec, records, folder, seed, config):
    import numpy as np
    import torch
    from PIL import Image

    set_seed(seed)
    model, processor = load_rtdetr(spec)
    trainable = freeze_rtdetr(model)
    device = torch_device(config)
    model.to(device)
    options = config.get("rtdetr_train", {})
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=options.get("learning_rate", 1e-4),
        weight_decay=options.get("weight_decay", 1e-4),
    )
    amp = bool(config.get("amp", True) and device.startswith("cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    batch_size = config.get("batch_size", 8)
    generator = np.random.default_rng(seed)
    history = []
    for epoch in range(config.get("epochs", 3)):
        rtdetr_training_mode(model)
        losses = []
        for start in range(0, len(records), batch_size):
            if start == 0:
                order = generator.permutation(len(records))
            batch = [records[i] for i in order[start : start + batch_size]]
            images = [Image.open(r["image"]).convert("RGB") for r in batch]
            annotations = []
            for i, r in enumerate(batch):
                annotations.append(
                    dict(
                        image_id=i,
                        annotations=[
                            dict(
                                category_id=0,
                                bbox=a["bbox"],
                                area=a["bbox"][2] * a["bbox"][3],
                                iscrowd=0,
                            )
                            for a in r["annotations"]
                            if not a.get("iscrowd")
                        ],
                    )
                )
            inputs = processor(
                images=images,
                annotations=annotations,
                return_tensors="pt",
                size=dict(
                    height=config.get("image_size", 640),
                    width=config.get("image_size", 640),
                ),
            )
            inputs = {
                k: (
                    [{a: b.to(device) for a, b in x.items()} for x in v]
                    if k == "labels"
                    else v.to(device)
                )
                for k, v in inputs.items()
            }
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda" if device.startswith("cuda") else "cpu",
                dtype=torch.float16,
                enabled=amp,
            ):
                loss = model(**inputs).loss
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                options.get("max_grad_norm", 0.1),
            )
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        history.append(dict(epoch=epoch + 1, loss=float(np.mean(losses))))
        print(spec["name"], seed, epoch + 1, history[-1], flush=True)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(folder / "checkpoint")
    processor.save_pretrained(folder / "checkpoint")
    (folder / "training.json").write_text(
        json.dumps(
            dict(
                seed=seed,
                history=history,
                trainable_parameters=trainable,
                checkpoint_revision=getattr(model.config, "_commit_hash", None),
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return str(folder / "checkpoint")


def predict(spec, checkpoint, records, config):
    predictions = []
    threshold = config.get("evaluation_confidence", 0.001)
    if spec["backend"] == "yolo":
        from ultralytics import YOLO

        model = YOLO(checkpoint or spec["checkpoint"])
        for i, r in enumerate(records, 1):
            result = model.predict(
                r["image"],
                conf=threshold,
                classes=[0],
                max_det=100,
                imgsz=config.get("image_size", 640),
                device=config.get("device", "0"),
                verbose=False,
            )[0]
            for box, score in zip(
                result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()
            ):
                x1, y1, x2, y2 = box
                predictions.append(
                    dict(
                        image_id=i,
                        category_id=1,
                        bbox=[x1, y1, x2 - x1, y2 - y1],
                        score=score,
                    )
                )
    else:
        import torch
        from PIL import Image

        model, processor = load_rtdetr(spec, checkpoint)
        device = torch_device(config)
        model.to(device).eval()
        batch_size = config.get("batch_size", 8)
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            inputs = processor(
                images=[Image.open(r["image"]).convert("RGB") for r in batch],
                return_tensors="pt",
                size=dict(
                    height=config.get("image_size", 640),
                    width=config.get("image_size", 640),
                ),
            ).to(device)
            with torch.inference_mode():
                outputs = model(**inputs)
            results = processor.post_process_object_detection(
                outputs,
                threshold=threshold,
                target_sizes=torch.tensor(
                    [[r["height"], r["width"]] for r in batch], device=device
                ),
            )
            for offset, result in enumerate(results):
                for box, score, label in zip(
                    result["boxes"].cpu().tolist(),
                    result["scores"].cpu().tolist(),
                    result["labels"].cpu().tolist(),
                ):
                    if label != 0:
                        continue
                    x1, y1, x2, y2 = box
                    predictions.append(
                        dict(
                            image_id=start + offset + 1,
                            category_id=1,
                            bbox=[x1, y1, x2 - x1, y2 - y1],
                            score=score,
                        )
                    )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return predictions


def coco_ap(annotation_path, predictions):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    truth = COCO(str(annotation_path))
    if predictions:
        detections = truth.loadRes(predictions)
    else:
        detections = COCO()
        detections.dataset = dict(
            images=truth.dataset["images"],
            categories=truth.dataset["categories"],
            annotations=[],
        )
        detections.createIndex()
    evaluator = COCOeval(truth, detections, "bbox")
    evaluator.params.catIds = [1]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return dict(
        ap=float(evaluator.stats[0]),
        ap50=float(evaluator.stats[1]),
        ap75=float(evaluator.stats[2]),
    )
