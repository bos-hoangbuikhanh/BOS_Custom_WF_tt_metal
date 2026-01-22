import pytest
import torch
from loguru import logger
from PIL import Image
from torchvision.transforms.functional import to_tensor

import ttnn
from models.bos_model.oft.demo.demo import adjust_calibration, transform_image
from models.bos_model.oft.demo.visualization.kitti import KittiObjectDataset
from models.bos_model.oft.reference.oftnet import OftNet
from models.bos_model.oft.ttnn.model_preprocessing import (
    create_OFT_model_parameters,
    create_OFT_model_parameters_oft,
    create_OFT_model_parameters_resnet,
    prepare_ttnn_input,
    preprocessing,
)
from models.bos_model.oft.ttnn.ttnn_oft import TtOft
from models.bos_model.oft.ttnn.ttnn_oftnet import TtOftNet
from models.bos_model.oft.ttnn.ttnn_resnet import TtBasicBlock, TtResNetFeatures
from models.common.utility_functions import disable_persistent_kernel_cache
from tests.ttnn.utils_for_testing import assert_with_pcc, comp_allclose

SliceHeight = ttnn.Conv2dSliceHeight
SliceWidth = ttnn.Conv2dSliceWidth

TEST_IMAGE_PATH = "models/bos_model/oft/data/kitti/object/testing/image_2/000001.png"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "input_tensor, path, inplanes, planes, stride, height_sharding, use_conv_dram",
    [
        # Resolution 180*540
        # (torch.rand((1, 64, 45, 135)), "frontend.layer1.0", 64, 64, 1, False, False),  # frontend.layer1.0 #0.999 pcc
        # (torch.rand((1, 64, 45, 135)), "frontend.layer1.1", 64, 64, 1, False, False),  # frontend.layer1.1 #0.999
        # (torch.rand((1, 64, 45, 135)), "frontend.layer2.0", 64, 128, 2, False, False),  # frontend.layer2.0 #0.999
        # (torch.rand((1, 128, 23, 68)), "frontend.layer2.1", 128, 128, 1, False, False),  # frontend.layer2.1 #0.999
        # (torch.rand((1, 128, 23, 68)), "frontend.layer3.0", 128, 256, 2, False, False),  # frontend.layer3.0 #0.999
        # (torch.rand((1, 256, 12, 34)), "frontend.layer3.1", 256, 256, 1, False, False),  # frontend.layer3.1 #0.999
        # (torch.rand((1, 256, 12, 34)), "frontend.layer4.0", 256, 512, 2, False, False),  # frontend.layer4.0 #0.999
        # (torch.rand((1, 512, 6, 17)), "frontend.layer4.1", 512, 512, 1, False, False),  # frontend.layer4.1 #0.999
        # (torch.rand((1, 256, 159, 159)), "topdown.0", 256, 256, 1, False, True),  # topdown.0 #0.999
        # (torch.rand((1, 256, 159, 159)), "topdown.1", 256, 256, 1, False, True),  # topdown.1 #0.999
        # (torch.rand((1, 256, 159, 159)), "topdown.2", 256, 256, 1, False, True),  # topdown.2 #0.999
        # (torch.rand((1, 256, 159, 159)), "topdown.3", 256, 256, 1, False, True),  # topdown.3 #0.999
        # (torch.rand((1, 256, 159, 159)), "topdown.4", 256, 256, 1, False, True),  # topdown.4 #0.999
        # (torch.rand((1, 256, 159, 159)), "topdown.5", 256, 256, 1, False, True),  # topdown.5 #0.999
        # (torch.rand((1, 256, 159, 159)), "topdown.6", 256, 256, 1, False, True),  # topdown.6 #0.999
        # (torch.rand((1, 256, 159, 159)), "topdown.7", 256, 256, 1, False, True),  # topdown.7 #0.999
        # # Resolution 256*832
        # (torch.rand((1, 64, 64, 208)), "frontend.layer1.0", 64, 64, 1, False, False),  # frontend.layer1.0 #0.999 pcc
        # (torch.rand((1, 64, 64, 208)), "frontend.layer1.1", 64, 64, 1, False, False),  # frontend.layer1.1 #0.999
        # (torch.rand((1, 64, 64, 208)), "frontend.layer2.0", 64, 128, 2, False, False),  # frontend.layer2.0 #0.999
        # (torch.rand((1, 128, 32, 104)), "frontend.layer2.1", 128, 128, 1, False, False),  # frontend.layer2.1 #0.999
        # (torch.rand((1, 128, 32, 104)), "frontend.layer3.0", 128, 256, 2, False, False),  # frontend.layer3.0 #0.999
        # (torch.rand((1, 256, 16, 52)), "frontend.layer3.1", 256, 256, 1, False, False),  # frontend.layer3.1 #0.999
        # (torch.rand((1, 256, 16, 52)), "frontend.layer4.0", 256, 512, 2, False, False),  # frontend.layer4.0 #0.999
        # (torch.rand((1, 512, 8, 26)), "frontend.layer4.1", 512, 512, 1, False, False),  # frontend.layer4.1 #0.999
        # (torch.rand((1, 256, 159, 159)), "topdown.0", 256, 256, 1, False, True),  # topdown.0 #0.999
        # (torch.rand((1, 256, 159, 159)), "topdown.1", 256, 256, 1, False, True),  # topdown.1 #0.999
        # (torch.rand((1, 256, 159, 159)), "topdown.2", 256, 256, 1, False, True),  # topdown.2 #0.999
        # (torch.rand((1, 256, 159, 159)), "topdown.3", 256, 256, 1, False, True),  # topdown.3 #0.999
        # (torch.rand((1, 256, 159, 159)), "topdown.4", 256, 256, 1, False, True),  # topdown.4 #0.999
        # (torch.rand((1, 256, 159, 159)), "topdown.5", 256, 256, 1, False, True),  # topdown.5 #0.999
        # (torch.rand((1, 256, 159, 159)), "topdown.6", 256, 256, 1, False, True),  # topdown.6 #0.999
        # (torch.rand((1, 256, 159, 159)), "topdown.7", 256, 256, 1, False, True),  # topdown.7 #0.999
        # Resolution 384*1248
        (torch.rand((1, 64, 96, 312)), "frontend.layer1.0", 64, 64, 1, False, False),  # frontend.layer1.0 #0.999 pcc
        (torch.rand((1, 64, 96, 312)), "frontend.layer1.1", 64, 64, 1, False, False),  # frontend.layer1.1 #0.999
        (torch.rand((1, 64, 96, 312)), "frontend.layer2.0", 64, 128, 2, False, False),  # frontend.layer2.0 #0.999
        (torch.rand((1, 128, 48, 156)), "frontend.layer2.1", 128, 128, 1, False, False),  # frontend.layer2.1 #0.999
        (torch.rand((1, 128, 48, 156)), "frontend.layer3.0", 128, 256, 2, False, False),  # frontend.layer3.0 #0.999
        (torch.rand((1, 256, 24, 78)), "frontend.layer3.1", 256, 256, 1, False, False),  # frontend.layer3.1 #0.999
        (torch.rand((1, 256, 24, 78)), "frontend.layer4.0", 256, 512, 2, False, False),  # frontend.layer4.0 #0.999
        (torch.rand((1, 512, 12, 39)), "frontend.layer4.1", 512, 512, 1, False, False),  # frontend.layer4.1 #0.999
        (torch.rand((1, 256, 159, 159)), "topdown.0", 256, 256, 1, False, True),  # topdown.0 #0.999
        (torch.rand((1, 256, 159, 159)), "topdown.1", 256, 256, 1, False, True),  # topdown.1 #0.999
        (torch.rand((1, 256, 159, 159)), "topdown.2", 256, 256, 1, False, True),  # topdown.2 #0.999
        (torch.rand((1, 256, 159, 159)), "topdown.3", 256, 256, 1, False, True),  # topdown.3 #0.999
        (torch.rand((1, 256, 159, 159)), "topdown.4", 256, 256, 1, False, True),  # topdown.4 #0.999
        (torch.rand((1, 256, 159, 159)), "topdown.5", 256, 256, 1, False, True),  # topdown.5 #0.999
        (torch.rand((1, 256, 159, 159)), "topdown.6", 256, 256, 1, False, True),  # topdown.6 #0.999
        (torch.rand((1, 256, 159, 159)), "topdown.7", 256, 256, 1, False, True),  # topdown.7 #0.999
    ],
)
def test_basic_block(device, input_tensor, path, inplanes, planes, stride, height_sharding, use_conv_dram, reset_seeds):
    disable_persistent_kernel_cache()

    model = OftNet(
        num_classes=1,
        frontend="resnet18",
        topdown_layers=8,
        grid_res=0.5,
        grid_height=4.0,
    )

    # use trained weights
    ckpt = torch.load("models/bos_model/oft/reference/checkpoint-0600.pth", map_location="cpu")
    model.load_state_dict(ckpt["model"])

    torch_module = model.get_submodule(path)

    parameters = create_OFT_model_parameters_resnet(torch_module, input_tensor, device=device)

    ttnn_input = input_tensor.permute((0, 2, 3, 1))
    ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, device=device)

    with torch.inference_mode():
        tt_module = TtBasicBlock(
            device,
            parameters,
            parameters.conv_args,
            inplanes=inplanes,
            planes=planes,
            stride=stride,
            height_sharding=height_sharding,
            slice_type=SliceHeight if use_conv_dram else None,
            num_slices=4 if use_conv_dram else None,
            # layer="topdown" if "topdown" in path else None,
            layer_name=path,
        )
        ttnn_output = tt_module(device, ttnn_input)
        ttnn_output = ttnn.to_torch(ttnn_output)
        ttnn_output = ttnn_output.permute((0, 3, 1, 2))

    with torch.inference_mode():
        torch_output = torch_module(input_tensor)

    logger.info(comp_allclose(torch_output, ttnn_output))
    passing, pcc = assert_with_pcc(ttnn_output, torch_output, 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 72916}], indirect=True)
@pytest.mark.parametrize(
    "input_tensor",
    [(to_tensor(Image.open(TEST_IMAGE_PATH)))],
    ids=["image"],
)
@pytest.mark.parametrize(
    "res",
    [(384, 1248)],
)
def test_resnet_features(device, input_tensor, res, reset_seeds):
    disable_persistent_kernel_cache()

    input_tensor = transform_image(input_tensor, res=res)
    input_tensor = input_tensor.unsqueeze(0)  # 4D tensor

    logger.info(f"Running ResNet submodule with resolution {res}")

    model = OftNet(
        num_classes=1,
        frontend="resnet18",
        topdown_layers=8,
        grid_res=0.5,
        grid_height=4.0,
    )

    # use trained weights
    ckpt = torch.load("models/bos_model/oft/reference/checkpoint-0600.pth", map_location="cpu")
    model.load_state_dict(ckpt["model"])

    torch_module = model.get_submodule("frontend")
    parameters = create_OFT_model_parameters_resnet(torch_module, input_tensor, device=device)

    ttnn_input = input_tensor.permute((0, 2, 3, 1))
    ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    with torch.inference_mode():
        torch_output = torch_module(input_tensor)

    with torch.inference_mode():
        tt_module = TtResNetFeatures(
            device, parameters, parameters.conv_args, TtBasicBlock, [2, 2, 2, 2], layer_name="frontend"
        )
        ttnn_output = tt_module(device, ttnn_input)

        for i in range(len(ttnn_output)):
            ttnn_output[i] = ttnn.to_torch(ttnn_output[i])
            ttnn_output[i] = ttnn_output[i].permute((0, 3, 1, 2))

    for i in range(len(torch_output)):
        passing, pcc = assert_with_pcc(ttnn_output[i], torch_output[i], 0.98)
        logger.info(f"Passing {i}th output: {passing}, PCC: {pcc}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "features, scale, layer_name",
    [
        (torch.rand((1, 256, 48, 156)), 1 / 8.0, "oft8"),
        (torch.rand((1, 256, 24, 78)), 1 / 16.0, "oft16"),
        (torch.rand((1, 256, 12, 39)), 1 / 32.0, "oft32"),
    ],
)
@pytest.mark.parametrize(
    "grid_res, grid_height",
    [(0.5, 4.0)],
)
@pytest.mark.parametrize(
    "res",
    [(384, 1248)],
)
def test_oft_module(device, features, grid_res, grid_height, scale, layer_name, res):
    disable_persistent_kernel_cache()

    dataset = KittiObjectDataset("models/bos_model/oft/data/kitti", "test", (80.0, 80.0), 0.5, 1.74)

    idx, image, calib, objects, grid = dataset[0]
    calib = adjust_calibration(calib, orig_size=(370, 1224), new_size=res)
    calib = calib.unsqueeze(0)
    grid = grid.unsqueeze(0)

    # Preprocess
    pre_config = preprocessing(calib, grid, grid_height, grid_res, res=res, device=device)

    model = OftNet(
        num_classes=1,
        frontend="resnet18",
        topdown_layers=8,
        grid_res=grid_res,
        grid_height=grid_height,
    ).eval()

    # use trained weights
    ckpt = torch.load("models/bos_model/oft/reference/checkpoint-0600.pth", map_location="cpu")
    model.load_state_dict(ckpt["model"])

    ttnn_input = ttnn.from_torch(features, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    torch_module = model.get_submodule(layer_name)
    parameters = create_OFT_model_parameters_oft(torch_module, (features, calib, grid), device=device)

    with torch.inference_mode():
        torch_output = torch_module(features, calib, grid)

    with torch.inference_mode():
        tt_module = TtOft(
            device,
            parameters,
            grid_res,
            grid_height,
            scale,
        )
        ttnn_output = tt_module(ttnn_input, pre_config[layer_name])
        ttnn_output = ttnn.to_torch(ttnn_output)

    logger.info(comp_allclose(torch_output, ttnn_output))
    passing, pcc = assert_with_pcc(torch_output, ttnn_output, 0.97)
    logger.info(f"Passing: {passing}, PCC: {pcc}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 52768}], indirect=True)
@pytest.mark.parametrize(
    "topdown, cell_size, grid_height",
    [(8, 0.5, 4.0)],
)
@pytest.mark.parametrize(
    "res",
    [
        # (180, 540),  # Uncomment to test 180*540 resolution
        # (256, 832),  # Uncomment to test 256*832 resolution
        (384, 1248)  # Default resolution
    ],
)
def test_oftnet(device, topdown, grid_height, cell_size, res):
    disable_persistent_kernel_cache()
    device.enable_program_cache()
    torch.manual_seed(42)

    # Load validation dataset to visualise
    dataset = KittiObjectDataset("models/bos_model/oft/data/kitti", "test", (80.0, 80.0), 0.5, 1.74)
    idx, image, calib, objects, grid = dataset[0]

    input_tensor = transform_image(to_tensor(image), res=res)
    calib = adjust_calibration(calib, orig_size=(370, 1224), new_size=res)

    logger.info(f"Running OftNet model with resolution {res}")

    # Add batch dimension
    input_tensor = input_tensor.unsqueeze(0)
    calib = calib.unsqueeze(0)
    grid = grid.unsqueeze(0)

    # Preprocess
    pre_config = preprocessing(calib, grid, grid_height, cell_size, res=res, device=device)

    torch_model = OftNet(
        num_classes=1,
        frontend="resnet18",
        topdown_layers=topdown,
        grid_res=cell_size,
        grid_height=grid_height,
    )

    ckpt = torch.load("models/bos_model/oft/reference/checkpoint-0600.pth", map_location="cpu")
    torch_model.load_state_dict(ckpt["model"])

    parameters = create_OFT_model_parameters(torch_model, (input_tensor, calib, grid), device=device)
    ttnn_input = prepare_ttnn_input(input_tensor, device)

    with torch.inference_mode():
        torch_output = torch_model(input_tensor, calib, grid)

    with torch.inference_mode():
        tt_model = TtOftNet(device, parameters, parameters.conv_args, TtBasicBlock, [2, 2, 2, 2])

        ttnn_output = tt_model(device, ttnn_input, pre_config)
        for i in range(len(ttnn_output)):
            ttnn_output[i] = ttnn.to_torch(ttnn_output[i])

    for i in range(len(torch_output)):
        logger.info(comp_allclose(torch_output[i], ttnn_output[i]))
        passing, pcc = assert_with_pcc(ttnn_output[i], torch_output[i], 0.95)
        logger.info(f"Passing {i}th output: {passing}, PCC: {pcc}")
