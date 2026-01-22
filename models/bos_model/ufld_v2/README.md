# Ultra-Fast-Lane-Detection-v2 (UFLD v2)

## Introduction
Ultra-Fast-Lane-Detection-v2 (UFLD v2) is a fast and efficient lane detection model based on hybrid anchor-driven ordinal classification.
This repository contains a TT-NN / TT-Metal optimized implementation supporting:

- Functional correctness tests (PCC)
- End-to-end performance benchmarks (Trace + 2CQ)
- Standalone performance runner
- Interactive demos (image / video / dataset)
- Dataset-level evaluation

Reference implementation:
https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2

---

## Set environment variables
```bash
# at $TT_METAL_HOME
source env_set.sh
```

## Model Variants

| Dataset  | Input Resolution | Model |
|---------|------------------|-------|
| TuSimple | 320 x 800 | TuSimple34 |
| CULane | 320 x 1600 | CULane34 |

Both variants are supported end-to-end (functional, perf, runner, demo).

---

## Model Weights Download

UFLD v2 requires pretrained **ResNet-34** model weights for the **TuSimple** and **CULane** datasets.
A helper script is provided to download the required weights automatically.

### Download Command

By default, the script downloads **both TuSimple and CULane ResNet-34 weights**:

```bash
    ./models/bos_model/ufld_v2/weights_download.sh
```

You can also download weights selectively using command-line arguments:

```bash
    # Download TuSimple (ResNet-34) weights only
    ./models/bos_model/ufld_v2/weights_download.sh --tusimple34

    # Download CULane (ResNet-34) weights only
    ./models/bos_model/ufld_v2/weights_download.sh --culane34
```

### Downloaded Files

Running the script downloads (or reuses, if already present) the following files:
    ```text
    models/bos_model/ufld_v2/tusimple_res34.pth
    models/bos_model/ufld_v2/culane_res34.pth
    ```

---

## Functional Tests (PCC)

### TuSimple

```bash
    pytest --disable-warnings \
      models/bos_model/ufld_v2/tests/pcc/test_ttnn_ufld_v2.py::test_ufld_v2_model
```

### CULane

```bash
    pytest \
      models/bos_model/ufld_v2/tests/pcc/test_ttnn_ufld_v2.py::test_ufld_v2_culane_model
```

---

## End-to-End Performance Tests (Trace + 2CQ)

### TuSimple

```bash
    pytest --disable-warnings \
      models/bos_model/ufld_v2/tests/perf/test_ufld_v2_e2e_performant.py::test_ufldv2_e2e_performant
```

### CULane

```bash
    pytest --disable-warnings \
      models/bos_model/ufld_v2/tests/perf/test_ufld_v2_e2e_performant.py::test_ufldv2_e2e_performant_culane
```

Notes:
- Uses Trace + 2 Command Queues
- Input mode: DRAM interleaved
- Batch size: 1

---

## Standalone Runner (Performance Runner)

UFLD v2 provides a standalone runner entry point for measuring pure inference performance (FPS) without demo or dataset logic.

The runner supports both TuSimple and CULane models and is intended for repeatable performance benchmarking.

### TuSimple Runner

```bash
    python -m models.bos_model.ufld_v2.runner.ufld_v2_runner \
      --device-id 0 \
      --num-iters 200 \
      --warmup-iters 15 \
      --model-type tusimple \
      --trace-region-size 6500000
```

### CULane Runner

```bash
    python -m models.bos_model.ufld_v2.runner.ufld_v2_runner \
      --device-id 0 \
      --num-iters 200 \
      --warmup-iters 15 \
      --model-type culane \
      --trace-region-size 6500000
```

---

### Runner Arguments

| Argument | Description |
|--------|-------------|
| --device-id | Target device ID |
| --num-iters | Number of measured inference iterations |
| --warmup-iters | Number of warm-up iterations (not included in timing) |
| --model-type | Model variant: tusimple or culane |
| --trace-region-size | Trace buffer size for Trace + CQ execution |

---

### Runner Output Format

The standalone runner returns a Python dictionary named result.
This dictionary always contains the following keys:

| Key | Type | Description |
|----|------|-------------|
| fps | float | Average frames per second measured over num-iters |
| input_tensor | torch.Tensor | Input tensor used for inference |
| output_tensor | torch.Tensor | Output tensor produced by the model |

Result structure:

```text
    result = {
        "fps": float,
        "input_tensor": torch.Tensor,
        "output_tensor": torch.Tensor,
    }
```

Notes:
- input_tensor and output_tensor are PyTorch tensors
- Only tensor shape and dtype are typically inspected to avoid excessive output
- Output tensor shape depends on the model:
  - TuSimple: (1, total_dim)
  - CULane: (1, total_dim)

---

## Demo (PyTest-based)

### TuSimple Demo

```bash
    pytest --disable-warnings \
      models/bos_model/ufld_v2/demo/demo.py::test_ufld_v2_demo
```

---
## Dataset Configuration (IMPORTANT)

Before running any UFLD v2 demos, dataset evaluation, or tests, **you must ensure that
the dataset root is correctly configured**.

UFLD v2 uses a **single source of truth** for dataset locations:

    models/bos_model/ufld_v2/demo/model_config.py

The dataset root is defined as:

    ```python
    # model_config.py
    data_root = "models/bos_model/ufld_v2/demo"
    ```

All dataset paths are resolved relative to this value.

If you want to place datasets in a different location, **update `data_root` before running**
any demo or test code.

### Expected Directory Structure

Given the `data_root` above, datasets are expected at:

    ```text
    <data_root>/image_data/
    ├── tusimple/
    │   └── test_label.json
    └── culane/
        └── list/test.txt
    ```

---

## Dataset Download (TuSimple / CULane)

UFLD v2 demos and dataset-based evaluations require the **TuSimple** and **CULane** datasets.
A helper script downloads the datasets via **kagglehub** and prepares symbolic links
under the configured dataset root.

### Download Command

    ```bash
    python models/bos_model/ufld_v2/demo/data_download.py
    ```

### Dataset Cache Location

The actual dataset files are downloaded and cached by kagglehub under:

    ```text
    ~/.cache/kagglehub/datasets/manideep1108/tusimple   (~23 GB)
    ~/.cache/kagglehub/datasets/manideep1108/culane     (~45 GB)
    ```

The download script then creates symbolic links under:

    ```text
    <data_root>/image_data/
    ```

- Ensure sufficient disk space (~70 GB) before downloading datasets.

---

## Dataset Evaluation

### TuSimple Dataset Evaluation (F1-score)

```bash
    pytest --disable-warnings \
      models/bos_model/ufld_v2/demo/dataset_evaluation.py::test_ufld_v2_dataset_inference
```

- Dataset: TuSimple
- Default number of samples: 100

---

## Interactive Demo: Dataset Mode

### CULane Dataset Demo

```bash
    python models/bos_model/ufld_v2/demo/demo_video.py \
      --dataset culane \
      --mode dataset \
      --loop
```

### TuSimple Dataset Demo

```bash
    python models/bos_model/ufld_v2/demo/demo_video.py \
      --dataset tusimple \
      --mode dataset \
      --loop
```

---

## Interactive Demo: Video Mode

- Video file is required in `demo/videos` . Please make "videos" folder.
- You can download the video from the below links.
  - Internal link: "https://bos-semi.atlassian.net/wiki/spaces/AIMultimed/pages/102400287/Yolov8s+demo+video".
  - External link: "https://bos-semi-demo-contents.s3.ap-northeast-2.amazonaws.com/public/demo_logo_text_v1.1.mp4" (Right click & Save video as ...)

### TuSimple Video Demo

```bash
    python models/bos_model/ufld_v2/demo/demo_video.py \
      --dataset tusimple \
      --mode video \
      --video_path models/bos_model/ufld_v2/demo/videos/demo_logo_text_v1.1.mp4
```

### CULane Video Demo

```bash
    python models/bos_model/ufld_v2/demo/demo_video.py \
      --dataset culane \
      --mode video \
      --video_path models/bos_model/ufld_v2/demo/videos/demo_logo_text_v1.1.mp4
```

---

## Video Demo: Save Output

### CULane Video Save Mode

```bash
    python models/bos_model/ufld_v2/demo/demo_video.py \
      --dataset culane \
      --mode video_save \
      --video_path models/bos_model/ufld_v2/demo/videos/demo_logo_text_v1.1.mp4
```

Optional argument:

```bash
    --max_frames N
```

- If --max_frames is specified, only the first N frames are processed and saved
- If --max_frames is not specified, the entire video is processed and saved

---

## Notes

- TuSimple and CULane differ in:
  - Input resolution
  - Anchor configuration
  - Output tensor dimensions
- The same runner infrastructure is shared across datasets
- Pretrained weights:
  - tusimple_res34.pth
  - culane_res34.pth
