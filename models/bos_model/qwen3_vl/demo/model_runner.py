# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import llama_models.llama3.reference_impl.generation as llama_reference_generation
from llama_models.llama3.api.chat_format import ChatFormat
from llama_models.llama3.api.datatypes import ImageMedia, UserMessage
from llama_models.llama3.api.tokenizer import Tokenizer
from loguru import logger
from PIL import Image as PIL_Image
from pkg_resources import resource_filename

from models.bos_model.qwen3_vl.tt.generator import create_submeshes

IMG_PATH = Path(resource_filename("llama_models", "scripts/resources/"))

import os
import time

import numpy as np
import torch

import ttnn
from models.bos_model.qwen3_vl.tt.common import hf_multimodal_encode
from models.bos_model.qwen3_vl.tt.generator import Generator
from models.bos_model.qwen3_vl.tt.model_config import CheckpointType, DecodersPrecision
from models.demos.utils.llm_demo_utils import create_benchmark_data, verify_perf
from models.perf.benchmarking_utils import BenchmarkProfiler

try:
    from tracy import signpost
except ModuleNotFoundError:

    def signpost(header: str, **kwargs):
        pass


def check_hf_online():
    import requests

    try:
        requests.get("https://huggingface.co", timeout=3)
        return True
    except:
        return False


def get_batch_sampler(temperature, top_p, tokenizer):
    def sample(logits):
        if temperature > 0:
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            next_token = llama_reference_generation.sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(logits[:, -1], dim=-1)

        next_tokens = next_token.reshape(-1)
        texts = [tokenizer.decode([next_tokens[i].item()]) for i in range(len(next_tokens))]
        return next_tokens, texts

    return sample


def create_random_image(width, height):
    """Create a random RGB image of specified dimensions."""
    # Generate random RGB values
    random_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return PIL_Image.fromarray(random_array, "RGB")


def create_multimodal_model(
    mesh_device,
    max_batch_size,
    max_seq_len,
    dtype=ttnn.bfloat16,
    use_paged_kv_cache=False,
    checkpoint=None,
    optimizations=None,
):
    from models.bos_model.qwen3_vl.tt.model_config import ModelArgs
    from models.bos_model.qwen3_vl.tt.vision.qwen_e2e_model import TtQwen_Model
    from models.tt_transformers.tt.multimodal.llama_vision_model import CrossAttentionTransformer

    tt_model_args = ModelArgs(mesh_device, max_batch_size=max_batch_size, optimizations=optimizations)
    assert tt_model_args.is_vision(), "This model is multimodal"

    # limit length or we'll run out of space
    tt_model_args.max_seq_len = max_seq_len
    if tt_model_args.is_90b:
        assert tt_model_args.device_name == "T3K", "90B model only supported on T3K right now"
        # for 90B model on T3K, use bfp8 and performance optimizations or the model won't fit in memory
        dtype = ttnn.bfloat8_b
        logger.info(f"Setting dtype to bfloat8_b for 90B model on T3K to fit model in memory")

    if checkpoint is None:
        checkpoint = tt_model_args.load_state_dict()

    if "Qwen3-VL" in tt_model_args.base_model_name:
        model = TtQwen_Model(
            mesh_device=mesh_device,
            state_dict=checkpoint,
            weight_cache_path=tt_model_args.weight_cache_path(ttnn.bfloat8_b),
            dtype=ttnn.bfloat8_b,
            args=tt_model_args,
            use_paged_kv_cache=use_paged_kv_cache,
        )
    else:
        model = CrossAttentionTransformer(
            mesh_device,
            state_dict=checkpoint,
            weight_cache_path=tt_model_args.weight_cache_path(dtype),
            dtype=dtype,
            configuration=tt_model_args,
            use_paged_kv_cache=use_paged_kv_cache,
        )
    return tt_model_args, model, checkpoint


def prepare_generator_args(
    num_devices,
    data_parallel,
    mesh_device,
    max_batch_size,
    max_seq_len,
    dtype=ttnn.bfloat16,
    use_paged_kv_cache=False,
    optimizations=None,
):
    submesh_devices = create_submeshes(mesh_device, data_parallel)
    state_dict = None

    model_args = []
    model = []

    for submesh in submesh_devices:
        model_args_i, model_i, state_dict = create_multimodal_model(
            mesh_device=submesh,
            max_batch_size=max_batch_size // data_parallel,
            max_seq_len=max_seq_len,
            dtype=dtype,
            use_paged_kv_cache=use_paged_kv_cache,
            checkpoint=state_dict,
            optimizations=optimizations,
        )
        model_args.append(model_args_i)
        model.append(model_i)

    return model_args, model


def get_mesh_device(param=1, device_params=None):
    if isinstance(param, tuple):
        grid_dims = param
        num_devices_requested = grid_dims[0] * grid_dims[1]
        mesh_shape = ttnn.MeshShape(*grid_dims)
    else:
        num_devices_requested = min(param, ttnn.get_num_devices())
        mesh_shape = ttnn.MeshShape(1, num_devices_requested)

    # Open the mesh device
    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape, **device_params)
    return mesh_device


def qwen3_runner(device_id, batch_size, num_iters, **kwargs):
    """
    Simple multimodal demo with limited dependence on reference code.
    """
    # device_id
    device_params = kwargs.get("device_params", {"trace_region_size": 14951424, "num_command_queues": 2})
    device_ids = ttnn.get_device_ids()
    assert device_id in device_ids, f"device_id {device_id} must be one of available device ids: {device_ids}"
    mesh_device = {"N150": (1, 1), "N300": (1, 2), "N150x4": (1, 4), "T3K": (1, 8), "TG": (8, 4)}.get(
        os.environ.get("MESH_DEVICE"), len([device_id])
    )
    mesh_device = get_mesh_device(mesh_device, device_params=device_params)
    max_batch_size = batch_size
    if "model" in kwargs:
        os.environ["HF_MODEL"] = kwargs["model"]
    elif os.environ.get("HF_MODEL") is None:
        os.environ["HF_MODEL"] = "Qwen/Qwen3-VL-8B-Instruct"
    enable_trace = kwargs.get("enable_trace", True)
    data_parallel = kwargs.get("data_parallel", 1)
    max_seq_len = kwargs.get("max_seq_len", 1024)
    is_ci_env = os.environ.get("CI") == "true"
    optimizations = kwargs.get("optimizations", "performance")
    optimizations_set = {
        "accuracy": lambda model_args: DecodersPrecision.accuracy(model_args.n_layers, model_args.model_name),
        "performance": lambda model_args: DecodersPrecision.performance(model_args.n_layers, model_args.model_name),
    }
    assert optimizations in optimizations_set, f"Invalid optimizations: {optimizations}"
    optimizations = optimizations_set[optimizations]

    image_list = kwargs.get(
        "images",
        [
            os.path.join(os.path.dirname(__file__), "images/car.png"),
            os.path.join(os.path.dirname(__file__), "images/car.png"),
            os.path.join(os.path.dirname(__file__), "images/dog.jpg"),
            os.path.join(os.path.dirname(__file__), "images/pasta.jpg"),
            os.path.join(os.path.dirname(__file__), "images/person.jpeg"),
        ],
    )
    query_list = kwargs.get(
        "queries",
        [
            "avoid image. can you count from 1 to 50 ?",
            "explain the main objects, their actions, and the scene context in detail.",
            "Is there an animal in the image? If yes, what is it? Describe the scene in detail.",
            "what is the dish in the image? Is there a fork in the image? ",
            "Is the dog wearing a strap? What is the color of the strap? Describe the exact view.",
        ],
    )
    res_list = kwargs.get("res", [(224, 224)])
    assert isinstance(image_list, list), f"images must be a list, got: {image_list}"
    assert isinstance(query_list, list), f"queries must be a list, got: {query_list}"
    assert len(image_list) == 0 or len(image_list) == len(
        query_list
    ), f"Number of images and queries must match. if you want text-only prompts, set image as None"
    if len(image_list) == 0:
        image_list = [None] * len(query_list)
    assert len(res_list) == 1 or len(res_list) == len(
        image_list
    ), f"Number of resolutions and images must match or be 1"
    if len(res_list) == 1:
        res_list = res_list * len(image_list)
    temperature = kwargs.get("temperature", 0)
    top_p = kwargs.get("top_p", 0.9)
    max_gen_len = kwargs.get("max_gen_len", 256)
    print_to_file = kwargs.get("print_to_file", False)
    output_dir = kwargs.get("output_dir", "models/bos_model/qwen3_vl/demo/outputs")

    if not check_hf_online():
        logger.info(f">>> HuggingFace is not reachable. Setting offline mode for demo.")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # Start profiler
    logger.info(f"Start profiler")
    profiler = BenchmarkProfiler()
    profiler.start("run")

    num_devices = mesh_device.get_num_devices() if isinstance(mesh_device, ttnn.MeshDevice) else 1
    max_batch_size *= data_parallel  # input batch_size is interpreted as size per DP group

    model_args, model = prepare_generator_args(
        num_devices=num_devices,
        data_parallel=data_parallel,
        mesh_device=mesh_device,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        optimizations=optimizations,
    )

    # Prepare output directory and file (only once)
    output_path = None
    if print_to_file:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"qwen3_vl_outputs.txt")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(
                f"Model Outputs (Model Name = {model_args[0].model_name}, Batch size = {max_batch_size}, Max gen len = {max_gen_len})\n"
            )
            f.write("=" * 80 + "\n\n")

    HF_MODEL = model_args[0].checkpoint_type == CheckpointType.HuggingFace

    if not HF_MODEL:
        ckpt_dir = os.environ["LLAMA_DIR"]
        tokenizer_path = str(Path(ckpt_dir) / "tokenizer.model")

        tokenizer = Tokenizer(model_path=tokenizer_path)
        formatter = ChatFormat(tokenizer)
    else:
        from transformers import AutoProcessor

        try:
            processor = AutoProcessor.from_pretrained(model_args[0].CKPT_DIR, local_files_only=True)
        except:
            processor = AutoProcessor.from_pretrained(model_args[0].CKPT_DIR)
    generator = Generator(model, model_args, mesh_device)

    xattn_caches = [
        model.setup_cache(model_args[i].max_batch_size) if not HF_MODEL else None
        for i, model in enumerate(generator.model)
    ]

    # prepare input
    trace_dialogs = []
    for img_path, res, query in zip(image_list, res_list, query_list):
        if img_path:
            with open(img_path, "rb") as f:
                h, w = res
                trace_img = PIL_Image.open(f).convert("RGB").resize((h, w))
            trace_dialogs.append([UserMessage(content=[ImageMedia(image=trace_img), query])])
        else:
            trace_dialogs.append([UserMessage(content=[query])])

    if len(trace_dialogs) < max_batch_size:
        trace_dialogs *= max_batch_size // len(trace_dialogs) + 1
    trace_dialogs = trace_dialogs[:max_batch_size]

    sampler = get_batch_sampler(temperature, top_p, model_args[0].tokenizer)
    _num_prefill_tokens = 0
    _num_decode_tokens = 0

    prompt_encoder = hf_multimodal_encode if HF_MODEL else formatter.encode_dialog_prompt

    logger.info(f"Starting inference...")
    logger.info(f"Num iterations: {num_iters}")

    final_outputs = []
    for batch_idx in range(num_iters):
        signpost(header=f"inference_batch_{batch_idx} start")
        logger.info(f"\n--- Processing batch {batch_idx + 1}/{num_iters} ---")

        for dialog in trace_dialogs:
            for msg in dialog:
                logger.info(f"{msg.role.capitalize()}: {msg.content}")

        batch_model_input = [
            prompt_encoder(dialog, processor) if HF_MODEL else prompt_encoder(dialog, tool_prompt_format=False)
            for dialog in trace_dialogs
        ]

        if HF_MODEL:
            tokenizer = processor.tokenizer
            image_grid_thw = [model_input.image_grid_thw for model_input in batch_model_input]
        else:
            image_grid_thw = None

        # Prefill inputs
        vision_images = [model_input.vision.images if model_input.vision else None for model_input in batch_model_input]
        vision_mask = [model_input.vision.mask if model_input.vision else None for model_input in batch_model_input]
        prompt_tokens = [model_input.tokens for model_input in batch_model_input]
        prefill_lens = torch.tensor([len(tokens) for tokens in prompt_tokens], dtype=torch.long)
        _num_prefill_tokens += prefill_lens.sum().item()
        total_lens = prefill_lens + max_gen_len

        pad_id = tokenizer.pad_token_id if HF_MODEL else tokenizer.pad_id
        bsz = len(prompt_tokens)
        tokens = torch.full((bsz, max(total_lens)), pad_id, dtype=torch.long)
        for i, seq in enumerate(prompt_tokens):
            tokens[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)

        # Prefill phase
        logger.info("Starting compile prefill...")
        signpost(header=f"compile_prefill_{batch_idx} start")
        prefill_start = time.perf_counter()
        with profiler("compile_prefill", iteration=batch_idx):
            (
                batch_logits,
                prefill_batch_xattn_masks,
                prefill_batch_text_masks,
                decode_batch_xattn_masks,
                decode_batch_text_masks,
            ) = generator.prefill_forward(
                vision_images,
                vision_mask,
                tokens,
                xattn_caches,
                total_lens,
                prefill_lens,
                image_grid_thw=image_grid_thw,
            )
        prefill_end = time.perf_counter()
        signpost(header=f"compile_prefill_{batch_idx} end")
        prefill_time_ms = (prefill_end - prefill_start) * 1000
        logger.info(f"compile prefill complete: {prefill_time_ms:.2f} ms")

        # Prefill phase
        logger.info("Starting inference prefill...")
        signpost(header=f"inference_prefill_{batch_idx} start")
        prefill_start = time.perf_counter()
        with profiler("inference_prefill", iteration=batch_idx):
            (
                batch_logits,
                prefill_batch_xattn_masks,
                prefill_batch_text_masks,
                decode_batch_xattn_masks,
                decode_batch_text_masks,
            ) = generator.prefill_forward(
                vision_images,
                vision_mask,
                tokens,
                xattn_caches,
                total_lens,
                prefill_lens,
                image_grid_thw=image_grid_thw,
            )
        prefill_end = time.perf_counter()
        signpost(header=f"inference_prefill_{batch_idx} end")
        prefill_time_ms = (prefill_end - prefill_start) * 1000
        logger.info(f"inference prefill complete: {prefill_time_ms:.2f} ms")

        # First decode token
        next_tokens, next_texts = sampler(batch_logits)
        for i, (next_token, next_text) in enumerate(zip(next_tokens, next_texts)):
            tokens[i, prefill_lens[i]] = next_token
        logger.info(f"Initial sampled tokens: {next_texts}")

        decode_times = []
        logger.info("Starting decode loop...")
        signpost(header=f"inference_decode_{batch_idx} start")
        profiler.start(f"inference_decode", iteration=batch_idx)

        all_outputs = [[*t[: prefill_lens[i]].tolist(), next_tokens[i].item()] for i, t in enumerate(tokens)]
        users_decoding = [True] * bsz
        gen_idx = 0

        while any(users_decoding) and gen_idx < max_gen_len - 1:
            if batch_idx == 0 and gen_idx == 0:  # First decode accounts for compile time
                profiler.start(f"compile_decode", iteration=batch_idx)

            decode_start = time.perf_counter()
            position_id = prefill_lens + gen_idx
            next_token_tensor = next_tokens.reshape(max_batch_size, 1)

            logits = generator.decode_forward(
                position_id,
                next_token_tensor,
                prefill_batch_xattn_masks,
                prefill_batch_text_masks,
                decode_batch_xattn_masks,
                decode_batch_text_masks,
                xattn_caches,
                enable_trace=enable_trace,
            )

            next_tokens, next_texts = sampler(logits)
            tokens[torch.arange(max_batch_size), position_id + 1] = next_tokens

            decode_end = time.perf_counter()
            iter_time = decode_end - decode_start
            decode_times.append(iter_time)

            if batch_idx == 0 and gen_idx == 0:
                profiler.end(f"compile_decode", iteration=batch_idx)

            tokens_per_sec_user = 1 / iter_time
            throughput = bsz * tokens_per_sec_user
            logger.info(
                f"Decode iteration {gen_idx}: {iter_time*1000:.0f}ms "
                f"@ {tokens_per_sec_user:.2f} tok/s/user ({throughput:.2f} tok/s total)"
            )

            # Append tokens and decode partial text for preview
            for i in range(bsz):
                if not users_decoding[i]:
                    continue
                user_tok = next_tokens[i].item()
                all_outputs[i].append(user_tok)
                text = tokenizer.decode(all_outputs[i])
                short_text = text[-100:].replace("\n", " ")
                logger.info(f"[User {i}] partial: ...{short_text}")

                if next_texts[i] in ["<|eot_id|>", "<|eom_id|>", "<|im_end|>", "<|im_start|>", "addCriterion"]:
                    users_decoding[i] = False

            gen_idx += 1

        profiler.end(f"inference_decode", iteration=batch_idx)
        signpost(header=f"inference_decode_{batch_idx} end")
        _num_decode_tokens += gen_idx * bsz

        avg_decode_time_ms = sum(decode_times) / len(decode_times) * 1000 if decode_times else 0
        logger.info(f"Average decode time per token: {avg_decode_time_ms:.2f} ms")
        logger.info("Final outputs for batch:")

        batch_outputs = []
        for i in range(bsz):
            tokens_out = [t for t in all_outputs[i] if t not in [tokenizer.pad_token_id]]
            text = tokenizer.decode(tokens_out)
            batch_outputs.append(text.strip())
            logger.info(f"[User {i}] Output: {text.strip()}\n")

            # ✅ Append each user's output directly to the single file
            if print_to_file and output_path:
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(f"[User {i}] Output:\n{text}\n\n")
            logger.info(f"📁 Saved to output file: {output_path}")
        final_outputs.append(batch_outputs)

    logger.info("=== All iterations complete ===")
    logger.info(f"Total prefill tokens: {_num_prefill_tokens}, decode tokens: {_num_decode_tokens}")

    # End profiling
    profiler.end("run")

    # Calculate measurements
    compile_prefill_time = profiler.get_duration("compile_prefill")
    compile_decode_time = profiler.get_duration("compile_decode")
    total_inference_prefill_time = profiler.get_duration_sum("inference_prefill")
    total_inference_decode_time = profiler.get_duration_sum("inference_decode", start_iteration=0) - compile_decode_time
    avg_ttft = total_inference_prefill_time / num_iters  # One first token per batch
    avg_prefill_t_s = _num_prefill_tokens / total_inference_prefill_time
    avg_decode_t_s = _num_decode_tokens / total_inference_decode_time
    avg_decode_t_s_u = _num_decode_tokens / total_inference_decode_time / max_batch_size

    measurements = {
        # Required measurements
        "compile_prefill": compile_prefill_time,
        "compile_decode": compile_decode_time,
        "inference_prefill": total_inference_prefill_time,
        "inference_decode": total_inference_decode_time,
        "prefill_time_to_token": avg_ttft,
        "prefill_t/s": avg_prefill_t_s,
        "decode_t/s/u": avg_decode_t_s_u,
        "decode_t/s": avg_decode_t_s,
    }

    # Print performance metrics
    logger.info("")
    logger.info(f"Performance metrics for batch 0")
    logger.info(f"Prefill compile time: {round(measurements['compile_prefill'], 4)}s")
    logger.info(f"Decode compile time: {round(measurements['compile_decode'], 4)}s")
    logger.info(f"Prefill inference time per user: {round(avg_ttft, 4)}s")
    logger.info(
        f"Total Decode inference time ({max_gen_len} iterations): {round(measurements['inference_decode'], 4)}s"
    )
    logger.info("")
    logger.info(f"Time to first token: {round(measurements['prefill_time_to_token']* 1000, 2)}ms")
    logger.info(f"Prefill t/s: {round(measurements['prefill_t/s'], 2)} tok/s")
    logger.info(
        f"Average speed: {round(1/avg_decode_t_s_u * 1000, 2)}ms @ {round(avg_decode_t_s_u, 2)} tok/s/user ({round(avg_decode_t_s, 2)} tok/s throughput)"
    )
    logger.info("")

    return {
        "tps": avg_decode_t_s,
        "ttft": measurements["prefill_time_to_token"],
        "output": final_outputs,
    }

    # # Tenstorrent performance CI verification code
    # logger.info(f"is_ci_env: {is_ci_env}")
    # if is_ci_env and max_batch_size == 1 and enable_trace:  # Only profiling these parametrizations
    #     tt_device_name = model_args[0].device_name
    #     base_model_name = model_args[0].base_model_name
    #     target_prefill_tok_s = {
    #         "N300_Llama-3.2-11B": 23.5,
    #         "T3K_Llama-3.2-11B": 21.5,
    #         "T3K_Llama-3.2-90B": 3,
    #     }[f"{tt_device_name}_{base_model_name}"]
    #
    #     target_decode_tok_s_u = {
    #         "N300_Llama-3.2-11B": 21.5,
    #         "T3K_Llama-3.2-11B": 37,
    #         "T3K_Llama-3.2-90B": 6,
    #     }[f"{tt_device_name}_{base_model_name}"]
    #
    #     target_decode_tok_s = target_decode_tok_s_u * max_batch_size
    #     targets = {
    #         "prefill_t/s": target_prefill_tok_s,
    #         "decode_t/s": target_decode_tok_s,
    #         "decode_t/s/u": target_decode_tok_s_u,
    #     }
    #
    #     # Save benchmark data for CI
    #     N_warmup_iter = {"inference_prefill": 0, "inference_decode": 0}
    #     benchmark_data = create_benchmark_data(profiler, measurements, N_warmup_iter, targets)
    #     benchmark_data.save_partial_run_json(
    #         profiler,
    #         run_type=f"{tt_device_name}-demo",
    #         ml_model_name=f"{base_model_name}-Vision",
    #         ml_model_type="vlm",
    #         num_layers=model_args[0].n_layers,
    #         batch_size=max_batch_size,
    #         input_sequence_length=max(prefill_lens).item(),
    #         output_sequence_length=max_gen_len,
    #     )
    #
    #     verify_perf(measurements, targets, high_tol_percentage=1.15)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", type=int, default=0, help="device id to use for inference")
    parser.add_argument("--batch_size", type=int, default=1, help="Size of batch")
    parser.add_argument("--num_iters", type=int, default=5, help="number of iteration to process. -1 is infinite loop")
    parser.add_argument(
        "--images",
        type=str,
        nargs="?",
        default=[
            os.path.join(os.path.dirname(__file__), "images/car.png"),
            os.path.join(os.path.dirname(__file__), "images/car.png"),
            os.path.join(os.path.dirname(__file__), "images/dog.jpg"),
            os.path.join(os.path.dirname(__file__), "images/pasta.jpg"),
            os.path.join(os.path.dirname(__file__), "images/person.jpeg"),
        ],
        help="input image path",
    )
    parser.add_argument(
        "--queries",
        type=str,
        nargs="?",
        default=[
            "avoid image. can you count from 1 to 50 ?",
            "explain the main objects, their actions, and the scene context in detail.",
            "Is there an animal in the image? If yes, what is it? Describe the scene in detail.",
            "what is the dish in the image? Is there a fork in the image? ",
            "Is the dog wearing a strap? What is the color of the strap? Describe the exact view.",
        ],
        help="Questions to process",
    )

    def parse_tuple(s) -> List[Tuple[int, int]]:
        import ast

        s = s.strip()

        dlim = ","
        dlim_count = s.count(dlim)
        assert dlim_count % 2 == 1, f"Resolution must be separated by a comma, got: {s!r}"
        assert not ("][" in s or ")(" in s), f"Resolution tuples must be separated by a comma, got: {s!r}"
        _ss = s.split(dlim)
        ss = [dlim.join(_ss[i * 2 : i * 2 + 2]) for i in range(dlim_count // 2 + 1)]

        res_list = []
        for s in ss:
            s = s.strip()
            s = s.replace("[", "(").replace("]", ")")
            if not (s.startswith("(") and s.endswith(")")):
                s = f"({s})"
            try:
                val = ast.literal_eval(s)  # safe for tuples/ints/strings
            except Exception as e:
                raise argparse.ArgumentTypeError(f"Invalid resolution {s!r}: {e}")
            if not (isinstance(val, tuple) and len(val) == 2):
                raise argparse.ArgumentTypeError(f"Resolution must be a tuple of 2 ints, got: {val!r}")
            w, h = val
            if not (isinstance(w, int) and isinstance(h, int)):
                raise argparse.ArgumentTypeError(f"Resolution must be 2 ints, got: {val!r}")
            if w <= 0 or h <= 0:
                raise argparse.ArgumentTypeError(f"Resolution must be positive, got: {val!r}")
            res_list.append((w, h))
        return res_list

    parser.add_argument(
        "--res",
        nargs="?",
        type=parse_tuple,
        default=[(224, 224)],
        help="Input image resolution e.g., --res (224,224), (384,384), ... or --res 224,224",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HF_MODEL name (default: Qwen/Qwen3-VL-8B-Instruct)",
    )

    args = parser.parse_args()

    if args.model is not None:
        os.environ["HF_MODEL"] = args.model
    elif os.environ.get("HF_MODEL") is not None:
        args.model = os.environ["HF_MODEL"]
    else:
        args.model = os.environ["HF_MODEL"] = "Qwen/Qwen3-VL-8B-Instruct"

    cfg = {
        "images": args.images,
        "queries": args.queries,
        "model": args.model,
        "enable_trace": True,  # default
        "device_params": {
            "trace_region_size": 14951424,  # default
            "num_command_queues": 2,  # default
        },
        "data_parallel": 1,  # default
        "max_seq_len": 1024,  # default
        "optimizations": "performance",  # default
        "res": args.res,
        "temperature": 0,  # default
        "top_p": 0.9,  # default  # if the temperature is 0, then top_p means nothing
        "max_gen_len": 256,  # default
        "print_to_file": False,  # default
        "output_dir": "models/bos_model/qwen3_vl/demo/outputs",  # Optional save path
    }
    qwen3_runner(device_id=args.device_id, batch_size=args.batch_size, num_iters=args.num_iters, **cfg)
