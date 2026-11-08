# Person detection

Install `requirements.txt` in a separate environment with CUDA-enabled PyTorch.
Copy `benchmark.example.json` to `benchmark.json` and set the dataset paths.
ID and OOD inputs use COCO annotations; real images accept COCO or Sheffield VOC.
Paths are relative to the configuration file.

```sh
python experiments/03_person_detection/benchmark.py run --config benchmark.json --work-dir outputs/detection
python experiments/03_person_detection/benchmark.py compare --work-dir outputs/detection --model RT-DETRv2-S --seed 0 --image-ids 1 2 3
```

The default protocol uses YOLO26s, YOLO26l, RT-DETRv2-S and RT-DETRv2-L; 1,800
images per condition; 900 ID + 900 OOD for Mixed; three epochs and three paired
seeds. Only the YOLO detection head or RT-DETRv2 decoder and prediction layers
are fine-tuned. Real images are excluded from training and checkpoint selection.

Evaluation reports person AP at IoU 0.50–0.95 on the real set and exports per-seed
results, mean/standard-deviation tables, raw predictions and comparison images.
`prepare.py`, `train.py` and `infer.py` also run the stages separately.
Configuration controls seeds, checkpoints, batch size, image size and optimizer
settings. Use a fresh work directory when changing the protocol.
