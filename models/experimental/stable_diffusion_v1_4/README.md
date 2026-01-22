# Stable_diffusion

## Introduction
Stable Diffusion is a latent text-to-image diffusion model capable of generating photo-realistic images given any text input.

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## How to Run

- Run the submodules:
```sh
pytest models/experimental/stable_diffusion_v1_4/tests/"<submodule_name>"
```

- Example:
```sh
pytest models/experimental/stable_diffusion_v1_4/tests/test_transformer_2d_model.py
```

- Currently all the submodules are running on N1 A0.
- Please refer the submodule names in [architecture.txt](architecture.txt)
- Geglu, Feedforward block, Time_Embedding , Downsample2d, Upsample2d, Upsample with nearest 2d, DownBlock2D, Upblock2D, UNetMidBlock2DCrossAttn are passing with full pcc.
  - `test_geglu.py`
  - `test_feedforward.py`
  - `test_embedding.py`
  - `test_downsample_2d.py`
  - `test_upsample_2d.py`
  - `test_upsample_nearest_2d.py`
  - `test_downblock_2d.py`
  - `test_upblock_2d.py`
  - `test_cross_attn_midblock_2d.py`
- BasicTransformerBlock
  - `test_basic_transformer_block.py` : 2/16 cases (1 up and 1 down) passing with 0.97 pcc.
- Transformer2DModel
  - `test_transformer_2d_model.py` : Higher resolution test cases are running in fallback groupnorm, 3 cases running 0.98 pcc.
- Cross_attention
  - `test_cross_attention.py` : 5/32 cases (3 up and 2 down) passing with 0.97 pcc.
- CrossAttnDownBlock2D
  - `test_cross_attn_downblock_2d.py` : 2/3 cases passing with 0.97 pcc.
- CrossAttnUpBlock2D
  - `test_cross_attn_up_block_2d.py` : 1/3 cases passing with 0.82 pcc.
- ResnetBlock2D
  - `test_resnet_block_2d.py` : Fixed all the SCB issues, all test 22/22 cases are passing with 0.99 pcc, (groupnorm fallback).
- UNet2DConditionModel
  - `test_unet_2d_condition_model.py` : [WIP] Currently the test hangs after the Down Blocks.
- Deom
  - `test_demo.py` : [WIP] Not fully supported for 5x4 grid yet

## Details
Note: ttnn stable diffusion utilizes `PNDMScheduler` and requires `num_inference_steps to be greater than or equal to 4`. [Reference](https://arxiv.org/pdf/2202.09778)

### Inputs
Inputs by default are provided from `input_data.json`. If you wish to change the inputs, provide a different path to test_demo.We do not recommend modifying `input_data.json` file.
The entry point to  functional_stable_diffusion model is UNet2DConditionModel in `models/demos/wormhole/stable_diffusion/tt/ttnn_functional_unet_2d_condition_model.py`. The model picks up certain configs and weights from huggingface pretrained model. We have used `CompVis/stable-diffusion-v1-4` version from huggingface as our reference.

### Metrics  Interpretation
`FID Score (Fréchet Inception Distance)` evaluates the quality of generated images by measuring the similarity between their feature distributions and those of real images. A lower FID score indicates better similarity between generated and real images.
For more information, refer [FID Score](https://lightning.ai/docs/torchmetrics/stable/image/frechet_inception_distance.html).

`CLIP Score` measures the similarity between the generated images and the input prompts. Higher CLIP scores indicate better alignment between the generated images and the provided text prompts.
For more information, refer [CLIP Score](https://lightning.ai/docs/torchmetrics/stable/multimodal/clip_score.html).

## WIP 2025-12-19

- Run and check the tests found in vae folder. (Change the import paths accordingly).
- Check the e2e model run (`test_unet_2d_condition_model`). Currently the test hangs after the Down Blocks.
- Run the demo and generate the images.

- To run the demo:
```sh
pytest --disable-warnings --input-path="models/demos/wormhole/stable_diffusion/demo/input_data.json" models/demos/wormhole/stable_diffusion/demo/demo.py::test_demo
```

- If you wish to run the demo with a different input:
```sh
pytest --disable-warnings --input-path="<address_to_your_json_file.json>" models/demos/wormhole/stable_diffusion/demo/demo.py::test_demo
```

- If you would like to run an interactive demo which will prompt you for the input:
```sh
pytest models/demos/wormhole/stable_diffusion/demo/demo.py::test_interactive_demo
```

- Our second demo is designed to run poloclub/diffusiondb dataset, run this with:
```sh
pytest --disable-warnings models/demos/wormhole/stable_diffusion/demo/demo.py::test_demo_diffusiondb
```

- If you wish to run for `num_prompts` samples and `num_inference_steps` denoising steps:
```sh
pytest --disable-warnings models/demos/wormhole/stable_diffusion/demo/demo.py::test_demo_diffusiondb[<num_prompts>-<num_inference_steps>]
```
