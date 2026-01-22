from argparse import ArgumentParser

import torch
from transformers import ViTForImageClassification

import ttnn
from models.bos_model.vit.ttnn_optimized_sharded_vit_a0 import TtViT


def run_vit(batch_size, device, torch_pixel_values):
    assert batch_size == 5, "The current vit implementation supports only batch size of 5."
    model_id = "google/vit-base-patch16-224"
    torch_vit = ViTForImageClassification.from_pretrained(model_id).eval()
    ttvit = TtViT(device, batch_size, torch_vit)

    with torch.inference_mode():
        out = ttvit(torch_pixel_values)
        pred = ttnn.to_torch(ttnn.from_device(out)).to(torch.float)
        class_id = pred[:, 0, :1000].argmax(dim=-1)

    return pred[:, 0, :1000], class_id


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size for inference")
    parser.add_argument("--device_id", type=int, default=0, help="Device ID for inference")
    args = parser.parse_args()

    device = ttnn.open_device(device_id=args.device_id, l1_small_size=32768)
    torch_pixel_values = torch.rand([args.batch_size, 3, 224, 224], dtype=torch.bfloat16)

    logits, class_id = run_vit(args.batch_size, device, torch_pixel_values)
    ttnn.close_device(device)
