# YOLOV10x

## Introduction:
Demo showcasing Yolov10x running on - A0 using ttnn.

YOLOv10x introduces a new approach to real-time object detection, addressing both the post-processing and model architecture deficiencies found in previous YOLO versions. By eliminating non-maximum suppression (NMS) and optimizing various model components, YOLOv10x achieves state-of-the-art performance with significantly reduced computational overhead. Extensive experiments demonstrate its superior accuracy-latency trade-offs across multiple model scales. We've used weights available [here](https://docs.ultralytics.com/models/yolov10x/#performance) under YOLOV10x.


## How to Run
Use the following command to run the Yolov10x model:
```
pytest --disable-warnings models/bos_model/yolov10x/tests/pcc/test_ttnn_yolov10x.py::test_yolov10x_model
```

### Model Performant with Trace+2CQ
- For `224x224`,

  ```bash
  pytest --disable-warnings models/bos_model/yolov10x/tests/perf/test_e2e_performant.py::test_e2e_performant -k res224
  ```

- For `320x320`,

  ```bash
  pytest --disable-warnings models/bos_model/yolov10x/tests/perf/test_e2e_performant.py::test_e2e_performant -k res320
  ```

### Demo
#### Custom Images:
- Use the following command to run demo for `224x224` resolution:

    ```bash
    pytest --disable-warnings models/bos_model/yolov10x/demo/demo.py::test_demo -k res224
    ```
- Use the following command to run demo for `320x320` resolution:

    ```bash
    pytest --disable-warnings models/bos_model/yolov10x/demo/demo.py::test_demo -k res320
    ```

- To use a different image(s) for demo, replace your image(s) in the image path `models/bos_model/yolov10x/demo/images`

Note: Output images will be saved in the `models/bos_model/yolov10x/demo/outputs` folder.


## Details
- The entry point to yolov10x model is YoloV10x in `models/bos_model/yolov10x/tt/ttnn_yolov10x.py`.
- Batch Size : `1` (Single Device)
- Supported Input Resolution : `(224, 224), (320, 320)` - (Height, Width).
- Note: The post-processing is performed using PyTorch.

### Inputs
The demo receives inputs from `models/bos_model/yolov10x/demo/images` dir by default. To test the model on different input data, it is recommended to add a new image file to this directory.

### Outputs
A runs folder will be created inside the `models/bos_model/yolov10x/demo/` directory. For reference, the model output will be stored in the torch_model directory, while the TTNN model output will be stored in the tt_model directory.

### Additional Information:
The tests can be run with  randomly initialized weights and pre-trained real weights.  To run only for the pre-trained weights, specify pretrained_weight_true when running the tests.
