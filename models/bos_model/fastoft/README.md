# FastOFT (Orthographic Feature Transform) Model

OFT is 3d object detection model that uses orthographic feature transforms to detect objects in 3D space. The model combines a ResNet-based frontend with specialized orthographic feature transformation layers and a topdown refinement network.

This FastOFT model is a customized model for better performance. The differences are as follows:
- **Frontend**: ResNetFeatures layer has only 2 layers, not 4. Every GroupNorm has been replaced by BatchNorm2d
- **Lateral Layers**: Every GroupNorm has been replaced by BatchNorm2d
- **OFT Layers**: Now there is a only one OFT layer (OFT8), not 3 OFT layers (OFT8, OFT16, OFT32)
  - The hidden feature size of conv3d has been changed from 256 to 128.
  - The grid sample in OFT will proceed in 'align_corners=True' and 'nearest' mode
- **Topdown Network**: Still contain 8 layers but the hidden channel size is reduced from 256 to 128
- **Detection Head**: also has 128 hidden channel size according to the Topdown network
- **The newly trained model weight is required to run this FastOFT model**

## Model Architecture

The OFT model consists of several key components:

- **Frontend**: ResNet-18/34 backbone for feature extraction at multiple scales (8x, 16x, 32x downsampling)
- **Lateral Layers**: Convert ResNet outputs to a common ~~256~~128-channel feature representation
- **OFT Layers**: Orthographic Feature Transform modules that project features into bird's-eye view
- **Topdown Network**: 8-layer refinement network using BasicBlock modules
- **Detection Head**: Final convolutional layer that outputs object scores, positions, dimensions, and angles
- **Decoder** Additional module that is used to decode encoded outputs into objects

The model outputs:
- **Scores**: Object detection confidence scores
- **Position Offsets**: 3D position predictions (x, y, z)
- **Dimension Offsets**: Object size predictions (width, height, length)
- **Angle Offsets**: Object orientation predictions (sin, cos components)
- **Objects**: Decoded outputs into list of detected objects.

## Project Structure

```
models/bos_model/fastoft/
├── demo/              # Demo scripts and visualization
├── reference/         # PyTorch reference implementation
├── resources/         # Test images and calibration files
├── tests/             # All tests together
    └── pcc/           # Unit tests for individual components
└── tt/                # TenstorrentNN (TTNN) optimized implementation
```

## Section 1: Demo Scripts

Input Requirements:
Both demos require:
- env variable CHECKPOINTS_PATH with pre-trained checkpoint file (e.g., `export CHECKPOINTS_PATH="/home/mbezulj/checkpoint-best-no-dist_01.pth.gz"`)
- Input images in JPG format (located in `resources/`)
- Corresponding calibration files in TXT format (camera intrinsic parameters)

#### demo.py
Full end-to-end inference demo that runs both PyTorch reference and TTNN implementations, comparing their outputs and generating visualizations.

Features:
- Loads pre-trained model weights from checkpoint
- Processes input images with calibration data
- Runs full OFT inference pipeline on both CPU (PyTorch) and device (TTNN)
- **Executes complete pipeline on TTNN**: OFTNet model inference + object decoder/encoder
- Compares intermediate outputs and final predictions
- Generates detection visualizations and heatmaps
- Supports various precision modes (float32, bfloat16)
- Configurable fallback modes for debugging

Usage:
```bash
TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="4,3" pytest models/bos_model/fastoft/demo/demo.py
```

## Section 2: Test Files

The test suite validates individual components of the OFT model, ensuring correctness of both reference and TTNN implementations.

### PCC (Pearson Correlation Coefficient) Tests

Located in `models/bos_model/fastoft/tests/pcc/`, these tests validate the accuracy of TTNN implementations against PyTorch reference models using PCC metrics.

#### test_basicblock.py
Tests the fundamental building block of the ResNet backbone and topdown network.

What it tests:
- TTBasicBlock forward pass correctness against PyTorch reference
- Memory layout conversions (NCHW ↔ NHWC)
- Sharding configurations for device execution
- Sequential execution of multiple BasicBlocks (topdown layers)

Key test cases:
- Single BasicBlock with various input dimensions
- 8 sequential BasicBlocks (mimicking topdown network)
- Different sharding strategies ("HS" - Height Sharding)

Usage:
```bash
TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="4,3" pytest models/bos_model/fastoft/tests/pcc/test_basicblock.py
```

#### test_encoder.py
Tests the object detection decoder/encoder that converts model outputs to final object detections.

What it tests:
- Peak detection in score heatmaps
- Non-maximum suppression (NMS)
- Object position, dimension, and angle decoding
- Score smoothing and filtering operations
- Object creation from decoded parameters

Key features:
- Loads pre-computed OFT outputs for consistent testing
- Validates intermediate processing steps
- Generates debug visualizations for manual inspection
- Tests both host and device implementations

Usage:
```bash
TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="4,3" pytest models/bos_model/fastoft/tests/pcc/test_encoder.py
```

#### test_oft.py
Tests the core Orthographic Feature Transform modules at different scales.

What it tests:
- OFT forward pass at 8x ~~16x, and 32x~~ scales
- Integral image computation
- Bounding box corner calculations
- Grid-based feature sampling
- Precision modes (float32, bfloat16)
- Pre-computed vs. on-demand grid calculation

Key test parameters:
- Different input resolutions corresponding to feature scales
- Various precision and grid computation modes
- Expected PCC (Pearson Correlation Coefficient) thresholds for each scale

Usage:
```bash
TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="4,3" pytest models/bos_model/fastoft/tests/pcc/test_oft.py
```

#### test_oftnet.py
Tests the OFTNet model (without decoder).

What it tests:
- Full model inference pipeline
- Integration of all components (ResNet + OFT + Topdown + Head)
- Host fallback mechanisms for debugging
- Multiple precision modes
- Real image processing with pre-trained weights

Key features:
- Uses real checkpoint weights for realistic testing
- Tests with actual images from the resources directory
- Configurable fallback modes (feedforward, lateral, OFT)
- Comprehensive intermediate output validation
- Output serialization for debugging encoder issues

Usage:
```bash
TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="4,3" pytest models/bos_model/fastoft/tests/pcc/test_oftnet.py
```

#### test_resnet.py
Tests the ResNet backbone feature extractor.

What it tests:
- ResNet-18 frontend implementation
- Multi-scale feature extraction (feats8, feats16, feats32)
- Memory layout handling for TTNN compatibility
- All intermediate activations in the ResNet pipeline

Key features:
- Tests with real images
- Validates all ResNet layers and operations
- Memory layout conversions for device execution
- Feature extraction at multiple downsampling rates

Usage:
```bash
TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="4,3" pytest models/bos_model/fastoft/tests/pcc/test_resnet.py
```

## Running All Tests
To run the complete test suite:

Usage:
```bash
# Run all PCC tests
TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="4,3" pytest models/bos_model/fastoft/tests/pcc/
```

## Environment Setup
The tests require:
- Pre-trained model checkpoint
- Test images and calibration files

## Expected Outputs

All tests generate:
- **Console logs**: Detailed PCC comparisons and validation results
- **Visualizations**: Debug plots and comparison images (saved to `outputs/` directories)

## Section 3: Device Performance Tests

The device performance test suite benchmarks the OFT model components on TT hardware, measuring execution time and ensuring performance regression detection.

#### test_device_perf_oft.py
Comprehensive performance benchmarking for OFT model components on TT device hardware.

What it tests:
- **OFTNet Model Performance**: Full model inference excluding decoder (test_device_perf_oft_oftnet)
- **Decoder Performance**: Object detection decoder performance in isolation (test_device_perf_oft_decoder)
- **Full Pipeline Performance**: End-to-end inference including complete demo pipeline (test_device_perf_oft_full)

Usage:
```bash
# Run OFTNet performance test
TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="4,3" pytest models/bos_model/fastoft/tests/test_device_perf_oft.py::test_device_perf_oft -k device_perf_oft_oftnet
```
```bash
# Run decoder performance test
TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="4,3" pytest models/bos_model/fastoft/tests/test_device_perf_oft.py::test_device_perf_oft -k device_perf_oft_decoder
```
```bash
# Run full oft performance test
TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="4,3" pytest models/bos_model/fastoft/tests/test_device_perf_oft.py::test_device_perf_oft -k device_perf_oft_full_demo_oft8
```
```bash
# Run all performance tests
TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="4,3" pytest models/bos_model/fastoft/tests/test_device_perf_oft.py::test_device_perf_oft
```


#### test_perf_e2e_oft.py
Optimised end-to-end performance benchmarking for OFT model on TT device hardware

What it tests:
- Multi-command queue model executor that overlaps input transfers with model execution (1cq, 2cqs)
- Same multi-command queue execution with a traced model for improved throughput (trace)
- The performance values are the average of 50 iterations
- This test use own executor and pipeline - oft_executor.py and oft_pipeline.py

Usage:
```bash
# Run multi-commend queue performance test
TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="4,3" pytest models/bos_model/fastoft/tests/test_perf_e2e_oft.py::test_perf_oft -k oft_1cq
TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="4,3" pytest models/bos_model/fastoft/tests/test_perf_e2e_oft.py::test_perf_oft -k oft_2cqs
```
```bash
# Run multi-commend queue performance test with traced model
TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="4,3" pytest models/bos_model/fastoft/tests/test_perf_e2e_oft.py::test_perf_oft -k oft_trace_1cq
TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="4,3" pytest models/bos_model/fastoft/tests/test_perf_e2e_oft.py::test_perf_oft -k oft_trace_2cqs
```


## Profiling

#### Run with Tracy

Tracy is a performance profiling tool that provides visual analysis of execution time and memory usage for each operation during model execution. It is useful for identifying performance bottlenecks and optimizing the FastOFT model.

Usage:
```bash
python -m tracy -r -p -v -m  pytest models/bos_model/fastoft/demo/demo.py
```

#### Run for ttnn-visualizer Profiler
- First, export ENV using script file
  - ```$EXPERIMENT_NAME```: input anythings (for example, ```fastoft```)
```
source models/bos_model/export_l1_vis.sh $EXPERIMENT_NAME
```

- Second, run model
  - If the model has finished running successfully, the result report will be generated in the following path (```generated/ttnn/reports/$EXPERIMENT_NAME_MMDD_hhmm/```)
```
pytest models/bos_model/fastoft/demo/demo.py
```

- Third, run ttnn-visualizer
    - ```$REPORT_PATH```: It is the path mentioned in the previous step
    - visit ```http://localhost:8000/``` using your web-browser
```
ttnn-visualizer --profiler-path $REPORT_PATH
```

- If the experiment has finished, please run the following command to clear the environment variables
```
source models/bos_model/unset_l1_vis.sh
```

## Visualization Demo

#### Run for image demo

Visualize the image inference results.

Options:
  - `-i`, `--input`: Input images directory or a single image path
  - `-n`, `--num_iter`: Number of iterations to process (default: `1`). Use `-1` for an infinite loop
  - `-c`, `--calib`: Calibration file directory or a single calibration file path
  - `--prep`: Input preprocessing mode: `padding`, `stretch`, `crop_top`, `crop_bottom`, or `crop_center` (default: `padding`)
  - `--output`: Output file path to save images
  - `--crop_k`: Additional vertical pixel offset applied to the crop starting position when using `--prep crop_*` (default: `0`).
  - `--full_res`: Display full-resolution images instead of resized output.

Example:
```bash
# live single image demo
python models/bos_model/fastoft/run_demo_image.py -i models/bos_model/fastoft/resources/000013.jpg -c models/bos_model/fastoft/resources/000013.txt --output models/bos_model/fastoft/demo/outputs

# live infinite loop
python models/bos_model/fastoft/run_demo_image.py -i models/bos_model/fastoft/resources -c models/bos_model/fastoft/resources -n 1

# live with different resolution - crop
python models/bos_model/fastoft/run_demo_image.py --input models/bos_model/fastoft/resources/720p.jpg --calib models/bos_model/fastoft/resources/720p.txt --prep crop_bottom --crop_k -90 --output models/bos_model/fastoft/demo/outputs/720p_out.jpg
```

#### Run for video

Visualize the results in video. Please, download the videos from here - https://www.kaggle.com/datasets/eeemmm1234/kitti-track-video. You can set the number of videos as `-1` and it means infinite loop too.

Usage:
```bash
python models/bos_model/fastoft/run_demo_video.py -i <your video directory> -n <num_video>
```
