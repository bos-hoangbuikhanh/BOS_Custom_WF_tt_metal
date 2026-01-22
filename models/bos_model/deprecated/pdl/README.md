# Panoptic-DeepLab (PDL)

## Platform
   N1 A0

## Introduction
Panoptic-DeepLab is a state-of-the-art bottom-up method for panoptic segmentation, where the goal is to assign semantic labels (e.g., person, road, building and so on) to every pixel in the input image as well as instance labels (e.g. an id of 1, 2, 3, etc) to pixels belonging to each class.

## Set environment variables
```bash
source ./python_env/bin/activate
source env_set.sh
```

## Install Required Packages and Download Model Weights
Before running the demo, download and save the pretrained weights as .pt file.

```bash
chmod +x models/bos_model/pdl/weights_downloader.sh
models/bos_model/pdl/weights_downloader.sh
```
This also installs all necessary packages.

## To run model in TTNN:
```bash
python models/bos_model/pdl/run_pdl.py -n 10
```
The argument *-n* denotes the number of images the model will be executing on.

## To run model using Trace:
```bash
python models/bos_model/pdl/run_pdl.py -n 10 --trace -p
```
Executing with *-p* sets Persistent Cache on. Persistent cache is off by default.

## To run demo video:
```bash
python models/bos_model/pdl/demo_pdl.py -p
python models/bos_model/pdl/demo_pdl.py -p --labels
```
Executing without labels is much faster. So for a faster FPS, execute without *--labels*.

## To run model on a video offline:
```bash
python models/bos_model/pdl/offline_demo.py -f 600
python models/bos_model/pdl/offline_demo.py -f 600 --torch
```
*-f* denotes the number of frames to be executed. Leave it blank to run for the entire video. Running using *--torch* flag uses the Torch model instead of TTNN model.


## Details
- Batch Size : `1`
- Supported Input Resolution - `(256, 512)` - (Height, Width).
- The post-processing is performed using PyTorch.

### Inputs
The demo receives inputs from `models/bos_model/pdl/test_images` dir by default. The Images have to be named *image1.png*, *image2.png* and so on. Use the *--source* argument when running to point to a different directory.
For video demo, the video is stored in `videos/car.mp4` by default. Use *--source* to point to a different video file.
You can download *test_images* from [here](https://bossemi.sharepoint.com/:f:/s/AIMMTeam/EvpSUruInpRLhm9OMMHEQpcBo0VnLE-usj-GX9qhu5gteQ?e=ZzkXal), and *car.mp4* from [here](https://bossemi.sharepoint.com/:v:/s/AIMMTeam/EUZFUUe6HuFPqiUCNHMxSbkBc2xIUInLFamDpNF2TiVxzA?e=Nl38dd).

### Outputs
Output images will be stored inside the `models/bos_model/pdl/output` directory. For offline compiled video, it will also be stored in the same directory under the same directory as `annotated_car.mp4`. If the video does not play, try using VLC Media Player, or any other media player with an MPEG-4 decoder. When using Torch model, the output will be saved as `torch_annotated_car.mp4`.
