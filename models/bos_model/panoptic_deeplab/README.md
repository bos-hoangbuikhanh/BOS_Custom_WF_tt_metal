# Panoptic DeepLab

## Platforms:
    Made for BOS chips, mostly tested on Blackhole with a core grid of 20 cores.

## Introduction
Panoptic DeepLab is a unified model for panoptic segmentation that combines semantic segmentation and instance segmentation into a single framework. The model uses a shared ResNet backbone with separate heads for semantic segmentation and instance embedding prediction, enabling comprehensive scene understanding by simultaneously identifying both "stuff" (background regions like road, sky) and "things" (countable objects like cars, people).

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## Setup

### Download Weights

Download the Panoptic-DeepLab CityScapes weights file:
[https://dl.fbaipublicfiles.com/detectron2/PanopticDeepLab/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32/model_final_bd324a.pkl]

Place the downloaded `model_final_bd324a.pkl` file in `models/bos_model/panoptic_deeplab/weights/`

## How to Run

### Run the Full Model Test
```bash
# From tt-metal root directory
pytest models/bos_model/panoptic_deeplab/tests/pcc/test_tt_model.py
```

### Run Component Tests
```bash
# Test ASPP component
pytest models/bos_model/panoptic_deeplab/tests/pcc/test_aspp.py

# Test ResNet backbone
pytest models/bos_model/panoptic_deeplab/tests/pcc/test_resnet.py

# Test semantic segmentation head
pytest models/bos_model/panoptic_deeplab/tests/pcc/test_semseg.py

# Test instance embedding head
pytest models/bos_model/panoptic_deeplab/tests/pcc/test_insemb.py
```

### Run Device Performance Tests
```bash
# Test full model performance
pytest models/bos_model/panoptic_deeplab/tests/test_pipeline_e2e.py
```

### Run the Demo
```bash
# Single image with custom output directory
python models/bos_model/panoptic_deeplab/run_panoptic_deeplab.py <image_or_directory_path> <weights_path> <output_dir>
```
Use the following arguments for convenience
- `--trace` enables running in Trace mode.
- `-p` enables Persistent cache
- `--enable_logger` enables Logger output
- `--model_category` selects the model variant:
  - `DEEPLAB_V3_PLUS`: (default): Semantic segmentation only. It's faster because it doesn't compute center and offset heads, which the full panoptic model uses for instance segmentation
  - `PANOPTIC_DEEPLAB` : Full model with semantic and instance segmentation (center and offset heads)

### Demo Output Files

The demo generates several output files for each processed image in separate directories:

**TTNN Output** (`ttnn_output/`):
- `{image_name}_original.jpg`: Original input image
- `{image_name}_panoptic.jpg`: TTNN panoptic segmentation visualization (blended with original)

**PyTorch Reference** (`pytorch_output/`):
- `{image_name}_original.jpg`: Original input image
- `{image_name}_panoptic.jpg`: PyTorch reference panoptic segmentation visualization

### Image Requirements

- **Best Results**: Street scene images similar to Cityscapes dataset
- **Supported Formats**: jpg, jpeg, png, bmp, tiff
- **Auto-Resize**: All images are automatically resized to 512x1024 for inference
- **Classes**: Predicts 19 Cityscapes classes (road, car, person, etc.)

## Details

- The entry point to the TTNN Panoptic DeepLab model is `TtPanopticDeepLab` in `models/experimental/panoptic_deeplab/tt/tt_model.py`. The model uses weights from the Detectron2 implementation.

**Input Size: 512x1024**
- Input size is optimized for Cityscapes dataset format with height=512 and width=1024.

**Batch Size: 1**
- Current implementation uses batch size of 1 for optimal memory usage with L1 memory configuration.

**Memory Configuration**
- The model is currently configured to mostly run in DRAM memory, as we optimize it we will be mostly in L1.

### Model Components

1. **ResNet Backbone**: Provides hierarchical feature extraction with multiple resolution levels
2. **ASPP Module**: Atrous Spatial Pyramid Pooling for multi-scale feature aggregation
3. **Semantic Segmentation Head**: Predicts pixel-wise semantic classes (19 Cityscapes categories)
4. **Instance Embedding Head**: Generates center heatmaps and offset vectors for instance segmentation
5. **Panoptic Fusion**: Combines semantic and instance predictions into unified panoptic segmentation

### Outputs

The model produces:
- **Semantic Logits**: Per-pixel classification scores for semantic classes
- **Center Heatmaps**: Instance center point predictions
- **Offset Vectors**: Pixel-to-instance-center displacement vectors
- **Panoptic Visualization**: Combined semantic and instance segmentation results


## Tracy profiling

Tracy is a performance profiling tool that provides visual analysis of execution time and memory usage for each operation during model execution. It is useful for identifying performance bottlenecks and optimizing the PDL model.

Usage:
```bash
python -m tracy -r -p -v -m pytest models/bos_model/panoptic_deeplab/tests/pcc/test_tt_model.py
```



## L1 Visualizer

L1 visualizer is a profiling tool that generates detailed reports on memory usage, buffer allocations, and tensor operations during model execution. It helps in understanding memory consumption patterns and optimizing memory usage for better performance.

<!-- ```
Usage:
export TTNN_CONFIG_OVERRIDES='{
    "enable_fast_runtime_mode": false,
    "enable_logging": true,
    "report_name": "pdl",
    "enable_graph_report": false,
    "enable_detailed_buffer_report": false,
    "enable_detailed_tensor_report": false,
    "enable_comparison_mode": false
}'
pytest models/bos_model/panoptic_deeplab/tests/pcc/test_tt_model.py
``` -->

- First, export ENV using script file
  - ```$EXPERIMENT_NAME```: input anythings (for example, ```pdl```)
```
source models/bos_model/export_l1_vis.sh $EXPERIMENT_NAME
```

- Second, run model
  - If the model has finished running successfully, the result report will be generated in the following path (```generated/ttnn/reports/<$EXPERIMENT_NAME>_MMMDD_hhmm/```)
```
pytest models/bos_model/panoptic_deeplab/tests/pcc/test_tt_model.py
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
