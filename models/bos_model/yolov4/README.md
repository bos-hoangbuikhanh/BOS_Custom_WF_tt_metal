# Yolov4

## Introduction
YOLOv4 is a state-of-the-art real-time object detection model introduced in 2020 as an improved version of the YOLO (You Only Look Once) series. Designed for both speed and accuracy, YOLOv4 leverages advanced techniques such as weighted residual connections, cross-stage partial connections, and mosaic data

## Set environment variables
```
# at $TT_METAL_HOME
source env_set.sh
```

## How to Run
### For 320x320:
```
pytest models/bos_model/yolov4/tests/pcc/test_ttnn_yolov4.py::test_yolov4[0-pretrained_weight_true-0]
```

### Model performant running with Trace+2CQ
#### Single Device (BS=1):
  ```
  pytest models/bos_model/yolov4/tests/perf/test_e2e_performant.py::test_e2e_performant
  ```

### Demo
**Note:** Output images will be saved in the `models/bos_model/yolov4/demo/outputs` folder.

#### Single Device (BS=1):
##### Custom Images:
- Use the following command to run demo for `320x320` resolution :
  ```
  pytest models/bos_model/yolov4/demo/demo.py::test_yolov4[resolution0-1-act_dtype0-weight_dtype0-models/bos_model/yolov4/resources-device_params0]
  ```

- To use a different image(s) for demo, replace your image(s) in the image path `models/bos_model/yolov4/resources/` and run:
  ```
  pytest models/bos_model/yolov4/demo/demo.py::test_yolov4[resolution0-1-act_dtype0-weight_dtype0-models/bos_model/yolov4/resources-device_params0]
  ```

##### Coco-2017 dataset:
- Use the following command to run demo for `320x320` resolution :
  ```
  pytest models/bos_model/yolov4/demo/demo.py::test_yolov4_coco[resolution0-1-act_dtype0-weight_dtype0-device_params0]
  ```

### To capture Tracy report:
```
python -m tracy -m -r -p -v "pytest models/bos_model/yolov4/tests/pcc/test_ttnn_yolov4.py::test_yolov4"
```

## Details
- The entry point to the `yolov4` is located at:`models/bos_model/yolov4/tt/yolov4.py`.
- Batch Size : `1` (Single Device).
- Supported Input Resolution - `(320, 320)` - (Height, Width).
