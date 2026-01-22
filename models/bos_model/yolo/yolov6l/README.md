# Yolov6l

### Introduction:
YOLOv6-L is a large variant of the YOLOv6 family—an advanced real-time object detection model developed by Meituan. YOLOv6 is designed to offer high performance in both accuracy and speed, making it suitable for industrial applications like autonomous driving, surveillance, and robotics. Resource link - [source](https://github.com/meituan/YOLOv6)

### Get model weight
```
./models/bos_model/yolo/yolov6l/weights_download.sh
```

### How to Run:

Use the following command to run the model :
```
pytest --disable-warnings models/bos_model/yolo/yolov6l/tests/pcc/test_ttnn_yolov6l.py
```

### Model Performant with Trace+2CQ

#### Single Device (BS=1) :

- For `320x320`,

  ```
  pytest --disable-warnings models/bos_model/yolo/yolov6l/tests/perf/test_e2e_performant.py::test_perf_yolov6l
  ```

### Demo:

#### Single Device (BS=1):

##### Custom Images:

- Use the following command to run demo for `320x320` resolution :

    ```
    pytest --disable-warnings models/bos_model/yolo/yolov6l/demo/demo.py::test_demo
    ```


##### Coco-2017 dataset:

- Use the following command to run demo for `320x320` resolution :

  ```
  pytest --disable-warnings models/bos_model/yolo/yolov6l/demo/demo.py::test_demo_dataset
  ```

### Details
- The entry point to yolov6l model is TtYolov6l in `models/bos_model/yolo/yolov6l/tt/ttnn_yolov6l.py`.
- Batch Size : `1` (Single Device)
- Supported Input Resolution : `(320, 320)` - (Height, Width).
- Dataset used for evaluation : **COCO-2017**
- Note: The post-processing is performed using PyTorch.

### Inputs
The demo receives inputs from `models/bos_model/yolo/yolov6l/demo/images` dir by default. To test the model on different input data, it is recommended to add a new image file to this directory.

### Outputs
A runs folder will be created inside the `models/bos_model/yolo/yolov6l/demo` directory. For reference, the model output will be stored in the `torch_model` directory, while the TTNN model output will be stored in the `tt_model` directory.
