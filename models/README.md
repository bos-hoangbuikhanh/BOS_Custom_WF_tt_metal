# Model (BOS Support)
The model performance below was measured based on FC_RC5 firmware.

## Model List

### LLM

| Model                                                         | Batch | Hardware | ttft (ms) | t/s/u | t/s    |
|---------------------------------------------------------------|-------|----------|-----------|-------|--------|
| [Qwen2.5 VL 3B](./bos_model/qwen25_vl)                        | 2     | A0       | 1186      | 10.73 | 21.47  |
| [Qwen2.5 VL 7B](./bos_model/qwen25_vl)                        | 2     | A0       | 1695      | 5.06  | 10.12  |
| [Qwen3 VL 2B](./bos_model/qwen3_vl)                           | 16    | A0       | 4406      | 16.02 | 256.5  |
| [Qwen3 VL 4B](./bos_model/qwen3_vl)                           | 16    | A0       | 5545      | 8.21  | 131.4  |
| [Qwen3 VL 8B](./bos_model/qwen3_vl)                           | 16    | A0       | 10541     | 4.77  | 76.34  |
| [Llama3.2 1B](./bos_model/llama32)                            | 16    | A0       | 52        | 22.77 | 356.9  |
| [Llama3.2 3B](./bos_model/llama32)                            | 16    | A0       | 116       | 9.98  | 159.75 |
| [Llama3.1 8B](./bos_model/llama32)                            | 16    | A0       | 233       | 4.83  | 77.28  |


> **Notes:**
>
> - ttft = time to first token | t/s/u = tokens/second/user | t/s = tokens/second; where t/s = t/s/u * batch.
> - The t/s/u reported is the throughput of the first token generated after prefill, i.e. 1 / inter token latency.


### Vision

#### Classification
| Model                                                         | Batch | Hardware | Frame/sec (FPS) |
|---------------------------------------------------------------|-------|----------|-----------------|
| [ResNet-50 (224x224)](./bos_model/resnet50)                   | 4     | A0       | 969.31          |
| [ViT-base (224x224)](./bos_model/vit)                         | 5     | A0       | 485             |


#### Detection
| Model                                                         | Batch | Hardware | Frame/sec (FPS) |
|---------------------------------------------------------------|-------|----------|-----------------|
| [YOLOv4 (320x320)](./bos_model/yolov4)                        | 1     | A0       | 73              |
| [YOLOv6l (320x320)](./bos_model/yolo/yolov6l)                 | 1     | A0       | 90              |
| [YOLOv8s (256x256)](./bos_model/yolov8s)                      | 1     | A0       | 140             |
| [YOLOv8s (320x320)](./bos_model/yolov8s)                      | 1     | A0       | 120             |
| [YOLOv10s (224x224)](./bos_model/yolo/yolov10s)               | 1     | A0       | 149             |
| [YOLOv10s (320x320)](./bos_model/yolo/yolov10s)               | 1     | A0       | 122             |
| [YOLOv10x (224x224)](./bos_model/yolov10x)                    | 1     | A0       | 53              |
| [YOLOv10x (320x320)](./bos_model/yolov10x)                    | 1     | A0       | 42              |
| [YOLOv11n (320x320)](./bos_model/yolo/yolov11n)               | 1     | A0       | 142             |
| [YOLOv12x (320x320)](./bos_model/yolo/yolov12x)               | 1     | A0       | 15              |
| [OFT (384x1280)](./bos_model/oft)                             | 1     | A0       | 1.19            |
| [FastOFT (384x1280)](./bos_model/fastoft)                     | 1     | A0       | TBL             |
| [UFLD_v2 (320x800) (tusimple)](./bos_model/ufld_v2)           | 1     | A0       | 47.03           |
| [UFLD_v2 (320x1600) (culane)](./bos_model/ufld_v2)            | 1     | A0       | 19.60           |


#### Segmentation
| Model                                                         | Batch | Hardware | Frame/sec (FPS) |
|---------------------------------------------------------------|-------|----------|-----------------|
| [Panoptic DeepLab (512x1024)](./bos_model/panoptic_deeplab)   | 1     | A0       | TBL             |
| [DeepLab V3 (512x1024)](./bos_model/panoptic_deeplab)         | 1     | A0       | TBL             |


#### Autonomous Driving
| Model                                                         | Batch | Hardware | Frame/sec (FPS) |
|---------------------------------------------------------------|-------|----------|-----------------|
| [SSR (6x640x384)](./bos_model/ssr)                            | 1     | A0       | 4.6             |
