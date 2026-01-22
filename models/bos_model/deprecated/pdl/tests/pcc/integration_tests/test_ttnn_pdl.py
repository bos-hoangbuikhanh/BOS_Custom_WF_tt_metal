import copy

import pytest
import torch
from detectron2.config import get_cfg
from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config
from loguru import logger

import ttnn
from models.bos_model.pdl.reference.panoptic_seg import PanopticDeepLab
from models.bos_model.pdl.tt.model_processing import create_pdl_model_parameters
from models.bos_model.pdl.tt.ttnn_panoptic_seg import (
    TtASPP,
    TtAvgPool2d,
    TtBottleneckBlock,
    TtConv2d,
    TtDeepLabStem,
    TtDeepLabV3PlusHead,
    TtPanopticDeepLab,
    TtPanopticDeepLabInsEmbedHead,
    TtPanopticDeepLabSemSegHead,
    TtResNet,
)

# Load tt modules
from tests.ttnn.utils_for_testing import assert_with_pcc, comp_allclose

# Global Scope
CONFIG_FILE_PATH = "models/bos_model/pdl/reference/configs/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024.yaml"
WEIGHT_FILE_PATH = "models/bos_model/pdl/reference/pdl_weights.pt"


def setup_cfg(config_file, weights_path, new_res=(256, 512)):
    cfg = get_cfg()
    add_panoptic_deeplab_config(cfg)

    cfg.merge_from_file(config_file)

    new_height, new_width = new_res

    cfg.INPUT.MIN_SIZE_TEST = new_height
    cfg.INPUT.MAX_SIZE_TEST = new_width

    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.CROP.TYPE = "absolute"
    cfg.INPUT.CROP.SIZE = (new_height, new_width)

    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = "cpu"
    cfg.freeze()
    return cfg


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "input_tensor, layer_name, shard_type",
    [
        (torch.rand(1, 2048, 16, 32), "sem_seg_head.decoder.res5.project_conv.convs.0", None),  # 0.999
        (torch.rand(1, 2048, 16, 32), "sem_seg_head.decoder.res5.project_conv.convs.1", None),  # 0.999
        (torch.rand(1, 2048, 16, 32), "sem_seg_head.decoder.res5.project_conv.convs.2", None),  # 0.999
        (torch.rand(1, 2048, 16, 32), "sem_seg_head.decoder.res5.project_conv.convs.3", None),  # 0.999
        (torch.rand(1, 288, 32, 64), "sem_seg_head.decoder.res2.fuse_conv.0", None),  # 0.999
        (torch.rand(1, 256, 32, 64), "sem_seg_head.decoder.res2.fuse_conv.1", None),  # 0.999
    ],
)
def test_TtConv2d(device, input_tensor, layer_name, shard_type, reset_seeds):
    # Torch model setup
    cfg = setup_cfg(CONFIG_FILE_PATH, WEIGHT_FILE_PATH)
    torch_model = PanopticDeepLab(cfg)
    state_dict = torch.load(cfg.MODEL.WEIGHTS, map_location="cpu")
    torch_model.load_state_dict(state_dict)
    torch_model.eval()

    # Fetch submodule
    torch_submodule = torch_model.get_submodule(layer_name)
    parameters = create_pdl_model_parameters(torch_submodule, input_tensor, device=device)

    # prepare inputs
    in_n, in_c, in_h, in_w = input_tensor.shape
    ttnn_input = input_tensor.permute((0, 2, 3, 1))
    ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, device=device)

    with torch.inference_mode():
        torch_output = torch_submodule(input_tensor)

    with torch.inference_mode():
        tt_submodule = TtConv2d(device, parameters, parameters["model_args"], shard_type=shard_type)
        ttnn_output, out_h, out_w = tt_submodule(ttnn_input, in_h, in_w)
        ttnn_output = ttnn.to_torch(ttnn_output)
        ttnn_output = ttnn_output.reshape((1, out_h, out_w, ttnn_output.shape[-1]))
        ttnn_output = ttnn_output.permute((0, 3, 1, 2))

    # PCC Check
    logger.info(comp_allclose(ttnn_output, torch_output))
    passing, pcc = assert_with_pcc(ttnn_output, torch_output, 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("input_tensor", [(torch.rand(1, 2048, 16, 32))])  # sem_seg_head.decoder.res5.project_conv
def test_TtAvgPool2d(device, input_tensor, reset_seeds):
    # Torch model setup
    cfg = setup_cfg(CONFIG_FILE_PATH, WEIGHT_FILE_PATH)
    torch_model = PanopticDeepLab(cfg)
    state_dict = torch.load(cfg.MODEL.WEIGHTS, map_location="cpu")
    torch_model.load_state_dict(state_dict)
    torch_model.eval()

    # Fetch submodule
    torch_submodule = torch_model.get_submodule("sem_seg_head.decoder.res5.project_conv.convs.4.0")

    # prepare inputs
    in_n, in_c, in_h, in_w = input_tensor.shape
    ttnn_input = input_tensor.permute((0, 2, 3, 1))
    ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, device=device)
    ttnn_input = ttnn.reshape(ttnn_input, (1, 1, in_n * in_h * in_w, in_c))

    with torch.inference_mode():
        torch_output = torch_submodule(input_tensor)

    with torch.inference_mode():
        tt_submodule = TtAvgPool2d(device, channels=2048)
        ttnn_output = tt_submodule(ttnn_input, inp_h=16, inp_w=32)
        ttnn_output = ttnn.to_torch(ttnn_output)
        ttnn_output = ttnn_output.reshape((1, torch_output.shape[2], torch_output.shape[3], ttnn_output.shape[-1]))
        ttnn_output = ttnn_output.permute((0, 3, 1, 2))

    # PCC Check
    logger.info(comp_allclose(ttnn_output, torch_output))
    passing, pcc = assert_with_pcc(ttnn_output, torch_output, 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("input_tensor", [(torch.rand(1, 2048, 16, 32))])  # sem_seg_head.decoder.res5.project_conv
def test_TtASPP(device, input_tensor, reset_seeds):
    # Torch model setup
    cfg = setup_cfg(CONFIG_FILE_PATH, WEIGHT_FILE_PATH)
    torch_model = PanopticDeepLab(cfg)
    state_dict = torch.load(cfg.MODEL.WEIGHTS, map_location="cpu")
    torch_model.load_state_dict(state_dict)
    torch_model.eval()

    # Fetch submodule
    torch_submodule = torch_model.get_submodule("sem_seg_head.decoder.res5.project_conv")
    parameters = create_pdl_model_parameters(torch_submodule, input_tensor, device=device)

    # prepare inputs
    in_n, in_c, in_h, in_w = input_tensor.shape
    ttnn_input = input_tensor.permute((0, 2, 3, 1))
    ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, device=device)
    ttnn_input = ttnn.reshape(ttnn_input, (1, 1, in_n * in_h * in_w, in_c))

    with torch.inference_mode():
        torch_output = torch_submodule(input_tensor)

    with torch.inference_mode():
        tt_submodule = TtASPP(device, parameters, parameters["model_args"])
        ttnn_output, out_h, out_w = tt_submodule(ttnn_input, inp_h=in_h, inp_w=in_w)
        ttnn_output = ttnn.to_torch(ttnn_output)
        ttnn_output = ttnn_output.reshape((1, out_h, out_w, ttnn_output.shape[-1]))
        ttnn_output = ttnn_output.permute((0, 3, 1, 2))

    # PCC Check
    logger.info(comp_allclose(ttnn_output, torch_output))
    passing, pcc = assert_with_pcc(ttnn_output, torch_output, 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc}")


@pytest.fixture
def features():
    return {
        "res2": torch.rand(1, 256, 64, 128),
        "res3": torch.rand(1, 512, 32, 64),
        "res5": torch.rand(1, 2048, 16, 32),
    }


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.skip(
    reason="There is no seprate DeepLabV3PlusHead torch submodule to fetch and test, so test is done by returning tensor in between from PanopticDeepLabSemSegHead."
)
def test_TtDeepLabV3PlusHead(device, features, reset_seeds):
    # Torch model setup
    cfg = setup_cfg(CONFIG_FILE_PATH, WEIGHT_FILE_PATH)
    torch_model = PanopticDeepLab(cfg)
    state_dict = torch.load(cfg.MODEL.WEIGHTS, map_location="cpu")
    torch_model.load_state_dict(state_dict)
    torch_model.eval()

    # Fetch submodule
    torch_submodule = torch_model.get_submodule("sem_seg_head")
    parameters = create_pdl_model_parameters(torch_submodule, features, device=device)

    ttnn_features = copy.deepcopy(features)

    # prepare inputs
    for layer_name in ["res2", "res3", "res5"]:
        ttnn_input = ttnn_features[layer_name]
        in_n, in_c, in_h, in_w = ttnn_input.shape
        ttnn_input = ttnn_input.permute((0, 2, 3, 1))
        ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, device=device)
        ttnn_input = ttnn.reshape(ttnn_input, (1, 1, in_n * in_h * in_w, in_c))
        ttnn_features[layer_name] = (ttnn_input, in_h, in_w)

    with torch.inference_mode():
        torch_output = torch_submodule(features)

    with torch.inference_mode():
        tt_submodule = TtDeepLabV3PlusHead(
            device, parameters.decoder, parameters["model_args"].decoder, input_shape=["res2", "res3", "res5"]
        )
        ttnn_output, out_h, out_w = tt_submodule(ttnn_features)
        ttnn_output = ttnn.to_torch(ttnn_output)
        ttnn_output = ttnn_output.reshape((1, out_h, out_w, ttnn_output.shape[-1]))
        ttnn_output = ttnn_output.permute((0, 3, 1, 2))

    # PCC Check
    logger.info(comp_allclose(ttnn_output, torch_output))
    passing, pcc = assert_with_pcc(ttnn_output, torch_output, 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_TtPanopticDeepLabSemSegHead(device, features, reset_seeds):
    # Torch model setup
    cfg = setup_cfg(CONFIG_FILE_PATH, WEIGHT_FILE_PATH)
    torch_model = PanopticDeepLab(cfg)
    state_dict = torch.load(cfg.MODEL.WEIGHTS, map_location="cpu")
    torch_model.load_state_dict(state_dict)
    torch_model.eval()

    # Fetch submodule
    torch_submodule = torch_model.get_submodule("sem_seg_head")
    parameters = create_pdl_model_parameters(torch_submodule, features, device=device)

    ttnn_features = copy.deepcopy(features)

    # prepare inputs
    for layer_name in ["res2", "res3", "res5"]:
        ttnn_input = ttnn_features[layer_name]
        in_n, in_c, in_h, in_w = ttnn_input.shape
        ttnn_input = ttnn_input.permute((0, 2, 3, 1))
        ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, device=device)
        ttnn_input = ttnn.reshape(ttnn_input, (1, 1, in_n * in_h * in_w, in_c))
        ttnn_features[layer_name] = (ttnn_input, in_h, in_w)

    with torch.inference_mode():
        torch_output = torch_submodule(features)[0]

    with torch.inference_mode():
        tt_submodule = TtPanopticDeepLabSemSegHead(device, parameters, parameters["model_args"])
        ttnn_output = tt_submodule(ttnn_features)[0]
        ttnn_output = ttnn.to_torch(ttnn_output)
        ttnn_output = ttnn_output.permute((0, 3, 1, 2))

    # PCC Check
    logger.info(comp_allclose(ttnn_output, torch_output))
    passing, pcc = assert_with_pcc(ttnn_output, torch_output, 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_TtPanopticDeepLabInsEmbedHead(device, features, reset_seeds):
    # Torch model setup
    cfg = setup_cfg(CONFIG_FILE_PATH, WEIGHT_FILE_PATH)
    torch_model = PanopticDeepLab(cfg)
    state_dict = torch.load(cfg.MODEL.WEIGHTS, map_location="cpu")
    torch_model.load_state_dict(state_dict)
    torch_model.eval()

    # Fetch submodule
    torch_submodule = torch_model.get_submodule("ins_embed_head")
    parameters = create_pdl_model_parameters(torch_submodule, features, device=device)

    ttnn_features = copy.deepcopy(features)

    # prepare inputs
    for layer_name in ["res2", "res3", "res5"]:
        ttnn_input = ttnn_features[layer_name]
        in_n, in_c, in_h, in_w = ttnn_input.shape
        ttnn_input = ttnn_input.permute((0, 2, 3, 1))
        ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        ttnn_input = ttnn.reshape(ttnn_input, (1, 1, in_n * in_h * in_w, in_c))
        ttnn_features[layer_name] = (ttnn_input, in_h, in_w)

    with torch.inference_mode():
        torch_output = torch_submodule(features)

    with torch.inference_mode():
        tt_submodule = TtPanopticDeepLabInsEmbedHead(device, parameters, parameters["model_args"])
        ttnn_output = tt_submodule(ttnn_features)

    # PCC Check
    for i in range(len(ttnn_output)):
        tt_output = ttnn.to_torch(ttnn_output[i])
        tt_output = tt_output.permute((0, 3, 1, 2))

        ref_output = torch_output[i]

        logger.info(comp_allclose(ref_output, tt_output))
        passing, pcc = assert_with_pcc(tt_output, ref_output, 0.99)
        logger.info(f"Passing: {passing}, PCC: {pcc}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "input_tensor, layer_name",
    [
        (torch.rand(1, 3, 256, 512), "backbone.stem"),  # 0.999
    ],
)
def test_TtDeepLabStem(device, input_tensor, layer_name, reset_seeds):
    # Torch model setup
    cfg = setup_cfg(CONFIG_FILE_PATH, WEIGHT_FILE_PATH)
    torch_model = PanopticDeepLab(cfg)
    state_dict = torch.load(cfg.MODEL.WEIGHTS, map_location="cpu")
    torch_model.load_state_dict(state_dict)
    torch_model.eval()

    # Fetch submodule
    torch_submodule = torch_model.get_submodule(layer_name)
    parameters = create_pdl_model_parameters(torch_submodule, input_tensor, device=device)

    # prepare inputs
    in_n, in_c, in_h, in_w = input_tensor.shape
    ttnn_input = input_tensor.permute((0, 2, 3, 1))
    ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, device=device)

    with torch.inference_mode():
        torch_output = torch_submodule(input_tensor)

    with torch.inference_mode():
        tt_submodule = TtDeepLabStem(device, parameters, parameters["model_args"])
        ttnn_output, out_h, out_w = tt_submodule(ttnn_input, in_h, in_w)
        ttnn_output = ttnn.to_torch(ttnn_output)
        ttnn_output = ttnn_output.reshape((1, out_h, out_w, ttnn_output.shape[-1]))
        ttnn_output = ttnn_output.permute((0, 3, 1, 2))

    # PCC Check
    logger.info(comp_allclose(ttnn_output, torch_output))
    passing, pcc = assert_with_pcc(ttnn_output, torch_output, 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "input_tensor, layer_name, shard_type",
    [
        (torch.rand(1, 128, 64, 128), "backbone.res2.0", None),  # 0.999
        (torch.rand(1, 256, 64, 128), "backbone.res2.1", None),  # 0.999
        (torch.rand(1, 256, 64, 128), "backbone.res2.2", None),  # 0.999
        (torch.rand(1, 256, 64, 128), "backbone.res3.0", None),  # 0.999
        (torch.rand(1, 512, 32, 64), "backbone.res3.1", None),  # 0.999
        (torch.rand(1, 512, 32, 64), "backbone.res3.2", None),  # 0.999
        (torch.rand(1, 512, 32, 64), "backbone.res3.3", None),  # 0.999
        # (torch.rand(1, 512, 32, 64), "backbone.res4.0", None),  # 0.999 # Device hangs
        (torch.rand(1, 1024, 16, 32), "backbone.res4.1", None),  # 0.999
        (torch.rand(1, 1024, 16, 32), "backbone.res4.3", None),  # 0.999
        (torch.rand(1, 1024, 16, 32), "backbone.res4.2", None),  # 0.999
        (torch.rand(1, 1024, 16, 32), "backbone.res4.4", None),  # 0.999
        (torch.rand(1, 1024, 16, 32), "backbone.res4.5", None),  # 0.999
        (torch.rand(1, 1024, 16, 32), "backbone.res5.0", None),  # 0.999
        (torch.rand(1, 2048, 16, 32), "backbone.res5.1", None),  # 0.999
        (torch.rand(1, 2048, 16, 32), "backbone.res5.2", None),  # 0.999
    ],
)
def test_TtBottleneckBlock(device, input_tensor, layer_name, shard_type, reset_seeds):
    # Torch model setup
    cfg = setup_cfg(CONFIG_FILE_PATH, WEIGHT_FILE_PATH)
    torch_model = PanopticDeepLab(cfg)
    state_dict = torch.load(cfg.MODEL.WEIGHTS, map_location="cpu")
    torch_model.load_state_dict(state_dict)
    torch_model.eval()

    # Fetch submodule
    torch_submodule = torch_model.get_submodule(layer_name)
    parameters = create_pdl_model_parameters(torch_submodule, input_tensor, device=device)

    # prepare inputs
    in_n, in_c, in_h, in_w = input_tensor.shape
    ttnn_input = input_tensor.permute((0, 2, 3, 1))
    ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    ttnn_input = ttnn.reshape(ttnn_input, (1, 1, in_n * in_h * in_w, in_c))

    with torch.inference_mode():
        torch_output = torch_submodule(input_tensor)

    with torch.inference_mode():
        tt_submodule = TtBottleneckBlock(device, parameters, parameters["model_args"], shard_type=shard_type)
        ttnn_output, out_h, out_w = tt_submodule(ttnn_input, in_h, in_w)
        ttnn_output = ttnn.to_torch(ttnn_output)
        ttnn_output = ttnn_output.reshape((1, out_h, out_w, ttnn_output.shape[-1]))
        ttnn_output = ttnn_output.permute((0, 3, 1, 2))

    # PCC Check
    logger.info(comp_allclose(ttnn_output, torch_output))
    passing, pcc = assert_with_pcc(ttnn_output, torch_output, 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "input_tensor, layer_name",
    [
        (torch.rand(1, 3, 256, 512), "backbone"),  # 0.999
    ],
)
def test_TtResNet(device, input_tensor, layer_name):
    # Torch model setup
    cfg = setup_cfg(CONFIG_FILE_PATH, WEIGHT_FILE_PATH)
    torch_model = PanopticDeepLab(cfg)
    state_dict = torch.load(cfg.MODEL.WEIGHTS, map_location="cpu")
    torch_model.load_state_dict(state_dict)
    torch_model.eval()

    # Fetch submodule
    torch_submodule = torch_model.get_submodule(layer_name)
    parameters = create_pdl_model_parameters(torch_submodule, input_tensor, device=device)

    # prepare inputs
    in_n, in_c, in_h, in_w = input_tensor.shape
    ttnn_input = input_tensor.permute((0, 2, 3, 1))
    ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, device=device)
    ttnn_input = ttnn.reshape(ttnn_input, (1, 1, in_n * in_h * in_w, in_c))

    with torch.inference_mode():
        torch_output = torch_submodule(input_tensor)

    with torch.inference_mode():
        tt_submodule = TtResNet(device, parameters, parameters["model_args"])
        ttnn_output = tt_submodule(ttnn_input, in_h, in_w)

    # PCC Check

    for layer_name in ["res2", "res3", "res5"]:
        tt_output, out_h, out_w = ttnn_output[layer_name]
        tt_output = ttnn.to_torch(tt_output)
        tt_output = tt_output.reshape((1, out_h, out_w, tt_output.shape[-1]))
        tt_output = tt_output.permute((0, 3, 1, 2))

        ref_output = torch_output[layer_name]
        logger.info(comp_allclose(tt_output, ref_output))
        passing, pcc = assert_with_pcc(tt_output, ref_output, 0.99)
        logger.info(f"Passing: {passing}, PCC: {pcc}")


# This test is to get the FPS numbers only.
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "input_tensor",
    [
        (torch.rand(1, 3, 256, 512)),
    ],
)
def test_TtPanopticDeepLab_model(device, input_tensor):
    # Torch model setup
    cfg = setup_cfg(CONFIG_FILE_PATH, WEIGHT_FILE_PATH)
    torch_model = PanopticDeepLab(cfg)
    state_dict = torch.load(cfg.MODEL.WEIGHTS, map_location="cpu")
    torch_model.load_state_dict(state_dict)
    torch_model.eval()

    # Fetch submodule
    parameters = create_pdl_model_parameters(torch_model, input_tensor, device=device)

    # prepare inputs
    in_n, in_c, in_h, in_w = input_tensor.shape
    ttnn_input = input_tensor.permute((0, 2, 3, 1))
    ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, device=device)
    ttnn_input = ttnn.reshape(ttnn_input, (1, 1, in_n * in_h * in_w, in_c))

    with torch.inference_mode():
        tt_submodule = TtPanopticDeepLab(device, parameters, parameters["model_args"])
        ttnn_output = tt_submodule(ttnn_input, in_h, in_w)

    logger.info(f"Test Ran Successfully")
