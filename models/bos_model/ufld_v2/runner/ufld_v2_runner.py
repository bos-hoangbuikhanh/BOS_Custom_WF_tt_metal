import time

import torch

import ttnn
from models.bos_model.ufld_v2.common import UFLD_V2_L1_SMALL_SIZE
from models.bos_model.ufld_v2.runner.performant_runner import UFLDPerformantRunner


def ufld_runner(device_id: int, batch_size: int, num_iters: int, **kwargs):
    """
    UFLD model runner that follows the same contract as ViT / ResNet runners.

    This function:
      - Measures FPS using repeated inference
      - Runs a single inference to return an output tensor
      - Uses internally generated synthetic input (no dataset dependency)

    Returns:
        dict:
            {
                "fps": float,
                "input_tensor": torch.Tensor,
                "output_tensor": torch.Tensor,
            }
    """

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------
    assert batch_size == 1, "UFLDPerformantRunner is currently configured for batch_size=1"

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    model_type = kwargs.get("model_type", "tusimple")  # "tusimple" | "culane"
    input_mode = kwargs.get("input_mode", "dram_interleaved")
    warmup_iters = kwargs.get("warmup_iters", 15)
    model_location_generator = kwargs.get("model_location_generator", None)

    # Input resolution depends on dataset type
    if model_type == "culane":
        H, W = 320, 1600
    else:
        H, W = 320, 800

    # ------------------------------------------------------------------
    # Device creation
    # ------------------------------------------------------------------
    device = ttnn.CreateDevice(
        device_id=device_id,
        l1_small_size=UFLD_V2_L1_SMALL_SIZE,
        trace_region_size=kwargs.get("trace_region_size", ttnn._ttnn.device.DEFAULT_TRACE_REGION_SIZE),
        num_command_queues=kwargs.get("num_command_queues", 2),
    )

    # ------------------------------------------------------------------
    # Synthetic input generation
    # ------------------------------------------------------------------
    torch_input = torch.randn(
        (batch_size, 3, H, W),
        dtype=torch.float32,
    )

    # ------------------------------------------------------------------
    # Runner initialization
    # ------------------------------------------------------------------
    runner = UFLDPerformantRunner(
        device=device,
        model_location_generator=model_location_generator,
        device_batch_size=batch_size,
        resolution=(H, W),
        torch_input_tensor=torch_input,
        input_mode=input_mode,
        model_type=model_type,
    )

    # ------------------------------------------------------------------
    # Warmup iterations (not measured)
    # ------------------------------------------------------------------
    for _ in range(warmup_iters):
        _ = runner.run(torch_input)

    ttnn.synchronize_device(device)

    # ------------------------------------------------------------------
    # FPS measurement
    # ------------------------------------------------------------------
    start_time = time.time()
    out_dev = None
    list_out_dev = []

    for _ in range(num_iters):
        out_dev = runner.run(torch_input)
        list_out_dev.append(ttnn.from_device(out_dev, blocking=False))  # to measure H2D

    ttnn.synchronize_device(device)
    end_time = time.time()

    elapsed_time = max(end_time - start_time, 1e-12)
    fps = (batch_size * num_iters) / elapsed_time

    # ------------------------------------------------------------------
    # Convert output tensor to torch
    # ------------------------------------------------------------------
    mesh_composer = getattr(getattr(runner, "runner_infra", None), "output_mesh_composer", None)
    if mesh_composer is not None:
        output_tensor = ttnn.to_torch(
            out_dev,
            mesh_composer=mesh_composer,
        )
    else:
        output_tensor = ttnn.to_torch(out_dev)

    # Match existing UFLD post-processing behavior
    output_tensor = output_tensor.squeeze(1).squeeze(1)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    runner.release()
    ttnn.CloseDevice(device)

    return {
        "fps": fps,
        "input_tensor": torch_input,
        "output_tensor": output_tensor,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="UFLD runner (fps + output tensor)")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-iters", type=int, default=100)
    parser.add_argument("--warmup-iters", type=int, default=15)
    parser.add_argument("--model-type", type=str, default="tusimple", choices=["tusimple", "culane"])
    parser.add_argument("--trace-region-size", type=int, default=6500000)

    args = parser.parse_args()

    kwargs = dict(
        model_type=args.model_type,
        warmup_iters=args.warmup_iters,
    )
    if args.trace_region_size is not None:
        kwargs["trace_region_size"] = args.trace_region_size

    result = ufld_runner(
        device_id=args.device_id,
        batch_size=args.batch_size,
        num_iters=args.num_iters,
        **kwargs,
    )

    print(f"FPS: {result['fps']}")
    print(f"Input tensor shape: {tuple(result['input_tensor'].shape)} dtype={result['input_tensor'].dtype}")
    # output_tensor could be torch tensor; print shape & dtype only (avoid huge prints)
    print(f"Output tensor shape: {tuple(result['output_tensor'].shape)} dtype={result['output_tensor'].dtype}")
