# OFT

## Platform
   N1 A0

## Introduction
Orthogonal Finetuning (OFT) is a method developed for adapting text-to-image diffusion models. It works by reparameterizing the pretrained weight matrices with it’s orthogonal matrix to preserve information in the pretrained model. To reduce the number of parameters, OFT introduces a block-diagonal structure in the orthogonal matrix.

## Download Model Weights & Test data

- Before running the demo, download and save the pretrained weights as .pt file from [here](http://web.bos-semi.com:8093/tt_metal_demo/bos_model_mcw/-/blob/feature/oft_model/oft/reference/checkpoint-0600.pth) or below url.

   ```bash
   http://web.bos-semi.com:8093/tt_metal_demo/bos_model_mcw/-/blob/feature/oft_model/oft/reference/checkpoint-0600.pth
   ```

- Move the weights inside `models/bos_model/oft/reference/` so that the final location should be

   ```bash
   models/bos_model/oft/reference/checkpoint-0600.pth
   ```

## Download the test data

- For sample data from kitti dataset, download from [here](https://bossemi.sharepoint.com/:f:/s/AIMMTeam/ElXpUgFd5U5GkF14fBR8Y1sBKXmLKfLNOS-YMVanfCl-pA?e=HIWRN0) or below url.

   ```bash
   https://bossemi.sharepoint.com/:f:/s/AIMMTeam/ElXpUgFd5U5GkF14fBR8Y1sBKXmLKfLNOS-YMVanfCl-pA?e=HIWRN0
   ```

- Paste data inside `models/bos_model/oft/data/` directory. The final location looks like

   ```bash
   models/bos_model/oft/data/kitti/object/testing/calib
   models/bos_model/oft/data/kitti/object/testing/image_2
   models/bos_model/oft/data/kitti/object/testing/label_2
   ```

## Set environment variables
```bash
source python_env/bin/activate  # You must already have
source env_set.sh
```


## Run demo
```bash
pytest models/bos_model/oft/demo/demo.py
```

__Outputs__

A runs folder will be created inside the `models/bos_model/oft/demo/runs` directory.
```bash
models/bos_model/oft/demo/runs/Torch_model  # result from torch model
models/bos_model/oft/demo/runs/Tt_model     # result from ttnn model
```

## Run visual demo (opencv windows)
```bash
pytest models/bos_model/oft/demo/run_demo_image.py -n -1
```
__Option__
```
  -h, --help            show this help message and exit
  -n NUM_IMAGES, --num_images NUM_IMAGES
                        number of images to process. -1 is infinite loop
  -i {180,256,384}, --image_size {180,256,384}
                        image height processed by bos model, width will be adjusted accordingly
  --torch               run torch model, not ttnn model
  --save_result         save result images
```

## Details
- The entry point to the `oft` is located at : `models/bos_model/oft/ttnn/ttnn_oftnet.py`.
- Batch Size : `1`
- Supported Input Resolution - `(384, 1248)` - (Height, Width).
- The post-processing is performed using PyTorch.
- The demo receives inputs from `models/bos_model/oft/data/kitti/object/testing/image_2` dir by default. To test the model on different input data, it is recommended to add a new image file to this directory.
