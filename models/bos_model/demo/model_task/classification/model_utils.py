from transformers import ViTForImageClassification

from models.bos_model.resnet50.tt.ttnn_functional_resnet50 import BenchmarkModelRunnerWrapper as BMTtResNet50
from models.bos_model.resnet50.tt.ttnn_functional_resnet50 import ModelRunnerWrapper as TtResNet50
from models.bos_model.vit.ttnn_optimized_sharded_vit_a0 import TtViT

SUPPORTED_MODEL_IDS = {
    "google/vit-base-patch16-224": {
        "model": {"functional": TtViT, "benchmark": None},
        "model_name": "ViT",
    },  # TODO: Need to implementation for benchmark
    "microsoft/resnet-50": {"model": {"functional": TtResNet50, "benchmark": BMTtResNet50}, "model_name": "ResNet50"},
}


def get_model(device, model_id: str, batch_size: int, use_trace: bool, use_2cq: bool, model_mode):
    if model_id == "google/vit-base-patch16-224":
        torch_model = ViTForImageClassification.from_pretrained(model_id)
    elif model_id == "microsoft/resnet-50":
        torch_model = None
    else:
        raise NotImplementedError(f"{model_id} is not supported yet")
    model_mode_key = (
        "functional" if model_mode == [True, False] else "benchmark" if model_mode == [False, True] else None
    )
    model = SUPPORTED_MODEL_IDS[model_id]["model"][model_mode_key](device, batch_size, torch_model, use_trace, use_2cq)
    return model
