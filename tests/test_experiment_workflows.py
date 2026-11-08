"""Standalone checks: python tests/test_experiment_workflows.py."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/03_person_detection"))
import data
import benchmark
from backends import coco_ap, freeze_rtdetr

spec = importlib.util.spec_from_file_location(
    "motion_analysis", ROOT / "experiments/02_navigation_motion/motion_analysis.py"
)
motion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(motion)


class DetectionTests(unittest.TestCase):
    def fixture(self, root, name, count):
        folder = root / name
        folder.mkdir()
        images = []
        annotations = []
        for i in range(count):
            path = folder / f"{i}.png"
            Image.new("RGB", (32, 24), (len(name) * 17, i * 30, 11)).save(path)
            images.append(dict(id=i, width=32, height=24, file_name=path.name))
            annotations.append(
                dict(
                    id=i + 1,
                    image_id=i,
                    category_id=7,
                    bbox=[2, 3, 8, 12],
                    area=96,
                    iscrowd=0,
                )
            )
        path = folder / "annotations.json"
        path.write_text(
            json.dumps(
                dict(
                    info={},
                    images=images,
                    annotations=annotations,
                    categories=[dict(id=7, name="person")],
                )
            )
        )
        return dict(images=str(folder), annotations=str(path))

    def test_paired_budgets_and_no_real_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = {
                name: self.fixture(root, name, 4) for name in ["id", "ood", "real"]
            }
            config.update(
                image_budget=4,
                expected_real_images=4,
                seeds=[0, 1, 2],
                models=[dict(name="one", backend="rtdetr")],
            )
            manifest = data.prepare(config, root / "out")
            self.assertEqual(len(benchmark.jobs(manifest)), 9)
            for cohorts in manifest["cohorts"].values():
                self.assertTrue(all(len(v) == 4 for v in cohorts.values()))
                mixed = [Path(x["image"]).parent.name for x in cohorts["mixed"]]
                self.assertEqual(mixed.count("id"), 2)
                self.assertEqual(mixed.count("ood"), 2)
                self.assertFalse(
                    {r["image"] for v in cohorts.values() for r in v}
                    & {r["image"] for r in manifest["real"]}
                )
            self.assertEqual(
                data.prepare(config, root / "out")["cohorts"], manifest["cohorts"]
            )

    def test_content_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = {
                name: self.fixture(root, name, 2) for name in ["id", "ood", "real"]
            }
            config.update(image_budget=2, expected_real_images=2, seeds=[0])
            (root / "real/0.png").write_bytes((root / "id/0.png").read_bytes())
            with self.assertRaisesRegex(ValueError, "leakage"):
                data.prepare(config, root / "out")

    def test_missing_seed_is_not_silently_averaged(self):
        results = [dict(model="M", condition="pretrained", seed=None, ap=0.5)]
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                data.summarize(results, temp, [0, 1, 2])

    def test_coco_perfect_and_empty_predictions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.fixture(root, "real", 1)
            document = json.loads(Path(source["annotations"]).read_text())
            document["categories"] = [dict(id=1, name="person")]
            document["annotations"][0]["category_id"] = 1
            path = root / "truth.json"
            path.write_text(json.dumps(document))
            self.assertAlmostEqual(
                coco_ap(
                    path,
                    [dict(image_id=0, category_id=1, bbox=[2, 3, 8, 12], score=0.99)],
                )["ap"],
                1.0,
            )
            self.assertEqual(coco_ap(path, [])["ap"], 0.0)

    def test_voc_person_mapping_and_coordinate_origin(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "1_img").mkdir()
            (root / "1_xml").mkdir()
            Image.new("RGB", (30, 20), "red").save(root / "1_img/0000_color.jpg")
            (root / "1_xml/0000_color.xml").write_text(
                "<annotation><filename>0000_color.jpg</filename><object><name>person</name><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>10</xmax><ymax>12</ymax></bndbox></object></annotation>"
            )
            result = data.load_voc(root)
            self.assertEqual(result[0]["annotations"][0]["bbox"], [0, 1, 10, 11])

    def test_only_decoder_and_prediction_parameters_update(self):
        import torch
        from torch import nn

        class Detector(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.backbone = nn.Linear(2, 2)
                self.model.encoder = nn.Linear(2, 2)
                self.model.decoder = nn.Linear(2, 2)
                self.model.enc_score_head = nn.Linear(2, 1)

            def forward(self, x):
                return self.model.enc_score_head(
                    self.model.decoder(self.model.encoder(self.model.backbone(x)))
                )

        model = Detector()
        freeze_rtdetr(model)
        before = {k: v.detach().clone() for k, v in model.named_parameters()}
        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad], lr=0.1
        )
        model(torch.ones(3, 2)).sum().backward()
        optimizer.step()
        for name, p in model.named_parameters():
            self.assertEqual(torch.equal(before[name], p), not p.requires_grad, name)


class MotionTests(unittest.TestCase):
    def test_constant_speed_straight_line(self):
        t = np.arange(0, 16.01, 0.05)
        xy = np.column_stack([t, np.zeros_like(t)])
        rows = motion.window_metrics(t, xy, smoothing_s=0)
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[0]["mean_speed"], 1.0, places=10)
        self.assertAlmostEqual(rows[0]["normalized_speed_sd"], 0.0, places=10)
        self.assertAlmostEqual(rows[0]["curvature_p95"], 0.0, places=10)
        self.assertAlmostEqual(rows[0]["end_s"], rows[1]["start_s"])

    def test_unit_circle_curvature(self):
        t = np.arange(0, 16.01, 0.05)
        xy = np.column_stack([np.cos(t), np.sin(t)])
        rows = motion.window_metrics(t, xy, smoothing_s=0)
        self.assertAlmostEqual(rows[0]["curvature_p95"], 1.0, places=2)

    def test_gaps_are_not_interpolated_into_windows(self):
        t = np.r_[np.arange(0, 5, 0.05), np.arange(8, 13, 0.05)]
        self.assertEqual(
            motion.window_metrics(t, np.column_stack([t, t * 0]), smoothing_s=0), []
        )

    def test_magni_centroids_units_and_robot_exclusion(self):
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / "track.csv"
            file.write_text(
                "BODY_NAMES,Helmet_1,DARKO_Robot\nFrame,Time,Helmet_1 Centroid_X,Helmet_1 Centroid_Y,DARKO_Robot Centroid_X,DARKO_Robot Centroid_Y\n1,0,1000,2000,8000,9000\n2,0.05,1100,2100,8000,9000\n"
            )
            tracks = list(motion.magni_tracks(file))
            self.assertEqual(len(tracks), 1)
            np.testing.assert_allclose(tracks[0][-1], [[1, 2], [1.1, 2.1]])


if __name__ == "__main__":
    unittest.main()
