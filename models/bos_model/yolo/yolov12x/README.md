# Yolo12X

## Introduction:

Yolov12 has an attention-centric architecture that moves away from the traditional CNN-based approaches of previous YOLO models while preserving the real-time inference speed crucial for many applications. This model leverages innovative attention mechanisms and a redesigned network architecture to achieve state-of-the-art object detection accuracy without compromising real-time performance.

## How to Run:

Use the following command to run the Yolo12x model with pre-trained weights :
```sh
pytest --disable-warnings models/bos_model/yolo/yolov12x/tests/pcc/test_ttnn_yolov12x.py::test_yolov12x[pretrained_weight_true-0]
```

### Model performant running with Trace+2CQ

#### Single Device (BS=1):
  ```
  pytest --disable-warnings models/bos_model/yolo/yolov12x/tests/perf/test_e2e_performant.py::test_e2e_performant
  ```

### Demo with Trace+2CQ:

##### Note: Output images will be saved in the `models/bos_model/yolo/yolov12x/demo/outputs` folder.

#### Single Device (BS=1):

#### Custom Images:

- Use the following command to run demo for `320x320` resolution :

    ```bash
    pytest --disable-warnings models/bos_model/yolo/yolov12x/demo/demo.py::test_demo
    ```

- To use a different image(s) for demo, replace your image(s) in the image path `models/bos_model/yolo/yolov12x/demo/images`.

#### COCO-2017 dataset:

- Use the following command to run demo for `320x320` resolution :

  ```
  pytest --disable-warnings models/bos_model/yolo/yolov12x/demo/demo.py::test_demo_dataset
  ```

- To use a different image(s) for demo, replace your image(s) in the image path `models/bos_model/yolo/yolov12x/demo/images`.

### Details:

The model picks up certain configs and weights from Ultralytics pretrained model. We've used weights available [here](https://docs.ultralytics.com/models/yolo12/#performance-metrics) under YOLO12x.

- The entry point to `yolov12x` model is `YoloV12x` in `models/bos_model/yolo/yolov12x/tt/yolov12x.py`.
- Batch Size : `1` (Single Device)
- Supported Input Resolution - `(320, 320)` - (Height, Width).
