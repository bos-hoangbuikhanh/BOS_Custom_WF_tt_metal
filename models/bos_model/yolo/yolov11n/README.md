# Yolov11n

## Introduction
**YOLOv11n** is the smallest variant in the YOLOV11 series, it offers improvements in accuracy, speed, and efficiency for real-time object detection. It features enhanced architecture and optimized training methods, suitable for various computer vision tasks.


## How to Run
Use the following command to run the model:
```
pytest --disable-warnings models/bos_model/yolo/yolov11n/tests/pcc/test_ttnn_yolov11.py::test_yolov11
```

### Model performant running with Trace+2CQ
#### Single Device (BS=1):
```
pytest --disable-warnings models/bos_model/yolo/yolov11n/tests/perf/test_e2e_performant.py::test_e2e_performant
```

### Demo with Trace+2CQ

#### Single Device (BS=1)
##### Custom Images:
- Use the following command to run demo for `320x320` resolution :
  ```bash
  pytest --disable-warnings models/bos_model/yolo/yolov11n/demo/demo.py::test_demo
  ```
  - To use a different image(s) for demo, replace your image(s) in the image path `models/bos_model/yolo/yolov11n/demo/images` and run the same command.

#### COCO-2017 dataset:
- Use the following command to run demo for `320x320` resolution :
  ```
  pytest --disable-warnings models/bos_model/yolo/yolov11n/demo/demo.py::test_demo_dataset
  ```

Note: Output images will be saved in the `models/bos_model/yolo/yolov11n/demo/runs` folder.

## Details
- The entry point to the `yolov11n` is located at : `models/bos_model/yolo/yolov11n/tt/ttnn_yolov11.py`
- Batch Size : `1` (Single Device)
- Supported Input Resolution : `(320, 320)` - (Height, Width).
