# Ultra-Fast-Lane-Detection-v2

## Platforms:
    A0

## Introduction
The Ultra-Fast-Lane-Detection-v2 is a PyTorch-based implementation designed for fast and efficient deep lane detection using hybrid anchor-driven ordinal classification. It enhances the speed and accuracy of lane detection models with a focus on real-time applications.

Resource link - [source](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2)

## Set environment variables
```
# at $TT_METAL_HOME
source env_set.sh
```

## How to Run:
### To run the model e2e test:
  ```
  pytest --disable-warnings models/bos_model/ufld_v2/tests/pcc/test_ttnn_ufld_v2.py::test_ufld_v2_model
  ```

### To capture Tracy report:
  ```
  python -m tracy -r -p -v -m pytest models/bos_model/ufld_v2/tests/pcc/test_ttnn_ufld_v2.py::test_ufld_v2_model
  ```

### To run the demo:
- To run the demo with torch backend:
```
python models/bos_model/ufld_v2/demo/demo.py --backend torch
```
The outputs will be saved under this directory: `models/bos_model/ufld_v2/demo/outputs/torch_outputs`

- To run the demo with ttnn backend with trace:
```
python models/bos_model/ufld_v2/demo/demo.py --backend ttnn
```
The ouputs will be saved under this directory: `models/bos_model/ufld_v2/demo/outputs/ttnn_outputs`

### To run the demo on your data:
- Add your images to the `images` directory under `demo` folder.

## Details
- The entry point of the model is located at ```models/bos_model/ufld_v2/ttnn/ttnn_ufld_v2.py```
- Batch Size : `1` (Single Device).
- Supported Input Resolution : `(320, 800)` - (Height, Width).
