# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import argparse
from pathlib import Path
from typing import Optional

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
import warnings
from contextlib import contextmanager

import numpy as np
import pytest
import torch

import ttnn
from models.bos_model.qwen3_vl.tt.common import hf_multimodal_encode
from models.bos_model.qwen3_vl.tt.generator import Generator
from models.bos_model.qwen3_vl.tt.model_config import CheckpointType
from models.demos.utils.llm_demo_utils import create_benchmark_data, verify_perf
from models.perf.benchmarking_utils import BenchmarkProfiler

# import logging
# logging.disable(logging.CRITICAL)

# from loguru import logger
# logger.remove()  # Removes the default stdout sink


def check_hf_online():
    import requests

    try:
        requests.get("https://huggingface.co", timeout=3)
        return True
    except:
        return False


if not check_hf_online():
    logger.info(f">>> HuggingFace is not reachable. Setting offline mode for demo.")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


@contextmanager
def suppress_loguru():
    """Temporarily disable all Loguru logging and Python warnings."""
    # Remember which modules were enabled
    logger.disable("__main__")
    logger.disable("models")  # optionally disable other modules if needed

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            yield
        finally:
            logger.enable("__main__")
            logger.enable("models")


def parse_args(argv=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--context", action="store_true", help="Enables Llama with context mode")
    parser.add_argument(
        "-g", "--max_generated_tokens", default=256, type=int, help="Maxiimum number of generated tokens"
    )
    parser.add_argument("-i", "--image-path", type=str, required=True, help="Path to the input image")
    parser.add_argument("-d", "--display", action="store_true", help="Display the image")

    args, _ = parser.parse_known_args(argv)
    return args


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


## image helper functions
from urllib.parse import urlparse


def is_url(s: str) -> bool:
    try:
        p = urlparse(s)
        return p.scheme in ("http", "https") and bool(p.netloc)
    except Exception:
        return False


def fetch_image_from_url(url: str, timeout: float = 15.0) -> PIL_Image:
    import io

    import requests

    headers = {"User-Agent": "live_demo/1.0"}
    r = requests.get(url, headers=headers, timeout=timeout, stream=True)
    r.raise_for_status()
    ctype = r.headers.get("Content-Type", "")
    if "image" not in ctype.lower():
        logger.warning(f"URL Content-Type is not image/* ({ctype}). Attempting to load anyway...")
    try:
        img = PIL_Image.open(io.BytesIO(r.content)).convert("RGB").resize((224, 224))
    except Exception as e:
        raise ValueError(f"Failed to load image from URL: {e}")
    return img


def load_image_from_path(path: str | Path) -> PIL_Image:
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Image path does not exist or is not a file: {p}")
    try:
        with open(p, "rb") as f:
            img = PIL_Image.open(f).convert("RGB").resize((224, 224))
    except Exception as e:
        raise ValueError(f"Failed to open image at {p}: {e}")
    return img


def get_validated_image(source: str) -> PIL_Image:
    trace_img = None
    while trace_img is None:
        try:
            if is_url(source):
                trace_img = fetch_image_from_url(source)
            else:
                trace_img = load_image_from_path(source)
            break
        except Exception as e:
            logger.error(f"Image loading failed: {e}")
            source = input("Please enter a valid image path or URL (or /bye to exit): ").strip()
            if source == "/bye":
                exit(0)
            continue
    return trace_img


import multiprocessing as mp

## image display function
import cv2


def viewer_proc(win_name: str = "Input Image", stop_event=mp.Event, frame=None):
    while not stop_event.is_set():
        if frame is not None:
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.imshow(win_name, frame)
            key = cv2.waitKey(1)
            if key == ord("q"):
                stop_event.set()
                break
    cv2.destroyAllWindows()


def create_multimodal_model(
    mesh_device,
    max_batch_size,
    max_seq_len,
    dtype=ttnn.bfloat16,
    use_paged_kv_cache=False,
    checkpoint=None,
):
    from models.bos_model.qwen3_vl.tt.model_config import ModelArgs
    from models.bos_model.qwen3_vl.tt.vision.qwen_e2e_model import TtQwen_Model
    from models.tt_transformers.tt.multimodal.llama_vision_model import CrossAttentionTransformer

    tt_model_args = ModelArgs(mesh_device, max_batch_size=max_batch_size)
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
        )
        model_args.append(model_args_i)
        model.append(model_i)

    return model_args, model


def warmup_model(generator, model_args, xattn_caches, HF_MODEL=False, processor=None):
    """Run a short warmup forward pass to stabilize the model before inference."""
    print("🔄 Running warmup pass...")

    # Minimal fake text and image input
    dummy_text = "Hello Qwen!"
    dummy_img = PIL_Image.new("RGB", (224, 224), color=(128, 128, 128))

    trace_dialogs = [
        [
            UserMessage(
                content=[
                    ImageMedia(image=dummy_img),
                    dummy_text,
                ]
            )
        ],
    ]

    prompt_encoder = hf_multimodal_encode if HF_MODEL else ChatFormat(model_args[0].tokenizer).encode_dialog_prompt
    batch_model_input = [
        prompt_encoder(dialog, processor) if HF_MODEL else prompt_encoder(dialog, tool_prompt_format=False)
        for dialog in trace_dialogs
    ]

    if HF_MODEL:
        tokenizer = processor.tokenizer
        image_grid_thw = [model_input.image_grid_thw for model_input in batch_model_input]
    else:
        tokenizer = model_args[0].tokenizer
        image_grid_thw = None

    vision_images = [model_input.vision.images if model_input.vision else None for model_input in batch_model_input]
    vision_mask = [model_input.vision.mask if model_input.vision else None for model_input in batch_model_input]
    prompt_tokens = [model_input.tokens for model_input in batch_model_input]
    prefill_lens = torch.tensor([len(tokens) for tokens in prompt_tokens], dtype=torch.long)
    total_lens = prefill_lens + 8  # small generation length for warmup

    pad_id = tokenizer.pad_token_id if HF_MODEL else tokenizer.pad_id
    bsz = len(prompt_tokens)
    tokens = torch.full((bsz, max(total_lens)), pad_id, dtype=torch.long)
    for i, seq in enumerate(prompt_tokens):
        tokens[i, : len(seq)] = seq.detach().clone().to(dtype=torch.long)

    # Forward passes (prefill + decode)
    generator.prefill_forward(
        vision_images, vision_mask, tokens, xattn_caches, total_lens, prefill_lens, image_grid_thw=image_grid_thw
    )

    next_token = torch.tensor([[tokenizer.eos_token_id if HF_MODEL else tokenizer.eos_id]])
    generator.decode_forward(prefill_lens, next_token, None, None, None, None, xattn_caches)

    print("✅ Warmup complete.")


def qwen_inference(
    mesh_device,
    warmup_iters=0,
    enable_trace=True,
    max_batch_size=1,
    data_parallel=1,
    max_seq_len=2048,
    temperature: float = 0,
    top_p: float = 0.9,
    max_gen_len: Optional[int] = 128,
    print_to_file: bool = True,  # <-- NEW ARG
    output_dir: Optional[str] = "models/bos_model/qwen3_vl/demo/outputs",  # <-- Optional save path
    image_path=None,
    display: bool = True,
):
    """
    Simple multimodal demo with limited dependence on reference code.
    """

    num_devices = mesh_device.get_num_devices() if isinstance(mesh_device, ttnn.MeshDevice) else 1
    max_batch_size *= data_parallel  # input batch_size is interpreted as size per DP group

    model_args, model = prepare_generator_args(
        num_devices=num_devices,
        data_parallel=data_parallel,
        mesh_device=mesh_device,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
    )

    # Prepare output directory and file (only once)
    output_path = None
    if print_to_file:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"qwen3_vl_outputs.txt")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(
                f"Model Outputs (Model Name = {model_args[0].base_model_name}, Batch size = {max_batch_size}, Max gen len = {max_gen_len})\n"
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

        processor = AutoProcessor.from_pretrained(model_args[0].CKPT_DIR)

    generator = Generator(model, model_args, mesh_device)

    xattn_caches = [
        model.setup_cache(model_args[i].max_batch_size) if not HF_MODEL else None
        for i, model in enumerate(generator.model)
    ]

    # --- Warmup run before interactive inference ---
    with suppress_loguru():
        warmup_model(generator, model_args, xattn_caches, HF_MODEL, processor if HF_MODEL else None)
    # -----------------------------------------------

    # Input image
    trace_img_0 = get_validated_image(image_path)
    if display:
        s = mp.Event()
        p = mp.Process(target=viewer_proc, args=("Input Image", s, np.array(trace_img_0)[:, :, ::-1]), daemon=False)
        p.start()

    input_prompt = "Hi Qwen!"
    print('Qwen: Hello! How can I assist you today? Enter "/bye" to exit.', end="")
    while True:
        input_prompt = input("\n\nMe: ")
        if input_prompt == "/bye":
            print("\nQwen: Thank you for using the service! Goodbye!")
            break

        trace_dialogs = [
            [
                UserMessage(
                    content=[
                        ImageMedia(image=trace_img_0),
                        input_prompt,
                    ]
                )
            ],
        ]

        if len(trace_dialogs) < max_batch_size:
            trace_dialogs *= max_batch_size // len(trace_dialogs)

        num_trace_batches = len(trace_dialogs) // max_batch_size
        sampler = get_batch_sampler(temperature, top_p, model_args[0].tokenizer)
        _num_prefill_tokens = 0
        _num_decode_tokens = 0

        prompt_encoder = hf_multimodal_encode if HF_MODEL else formatter.encode_dialog_prompt

        for iter_num in range(warmup_iters + 1):
            current_dialogs = trace_dialogs

            for batch_idx in range(num_trace_batches):
                batch_dialogs = current_dialogs[batch_idx * max_batch_size : (batch_idx + 1) * max_batch_size]

                for dialog in batch_dialogs:
                    for msg in dialog:
                        logger.info(f"{msg.role.capitalize()}: {msg.content}")

                batch_model_input = [
                    prompt_encoder(dialog, processor) if HF_MODEL else prompt_encoder(dialog, tool_prompt_format=False)
                    for dialog in batch_dialogs
                ]

                if HF_MODEL:
                    tokenizer = processor.tokenizer
                    image_grid_thw = [model_input.image_grid_thw for model_input in batch_model_input]
                else:
                    image_grid_thw = None

                # Prefill inputs
                vision_images = [
                    model_input.vision.images if model_input.vision else None for model_input in batch_model_input
                ]
                vision_mask = [
                    model_input.vision.mask if model_input.vision else None for model_input in batch_model_input
                ]
                prompt_tokens = [model_input.tokens for model_input in batch_model_input]
                prefill_lens = torch.tensor([len(tokens) for tokens in prompt_tokens], dtype=torch.long)
                _num_prefill_tokens += prefill_lens.sum().item()
                total_lens = prefill_lens + max_gen_len

                pad_id = tokenizer.pad_token_id if HF_MODEL else tokenizer.pad_id
                bsz = len(prompt_tokens)
                tokens = torch.full((bsz, max(total_lens)), pad_id, dtype=torch.long)
                # for i, seq in enumerate(prompt_tokens):
                #     tokens[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
                for i, seq in enumerate(prompt_tokens):
                    tokens[i, : len(seq)] = seq.detach().clone().to(dtype=torch.long)

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

                next_tokens, next_texts = sampler(batch_logits)
                prefilled_token = next_tokens.clone()

                for i, (next_token, next_text) in enumerate(zip(next_tokens, next_texts)):
                    tokens[i, prefill_lens[i]] = next_token

                all_outputs = [[*t[: prefill_lens[i]].tolist(), next_tokens[i].item()] for i, t in enumerate(tokens)]
                users_decoding = [True] * bsz
                gen_idx = 0
                print_prefill_token = True

                while any(users_decoding) and gen_idx < max_gen_len - 1:
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

                    # Append tokens and decode partial text for preview
                    skip_tokens = {"<|eot_id|>", "<|eom_id|>", "<|im_end|>", "<|im_start|>", "addCriterion"}
                    for i in range(bsz):
                        if not users_decoding[i]:
                            continue
                        user_tok = next_tokens[i].item()
                        all_outputs[i].append(user_tok)
                        text = tokenizer.decode(all_outputs[i])

                        if print_prefill_token:
                            token_text = tokenizer.decode([int(prefilled_token[0].item())])
                            print(f"Qwen: {token_text}", end="", flush=True)
                            print_prefill_token = False

                        # Decode the current token
                        token_text = tokenizer.decode([user_tok])

                        # Only print if it's not in skip_tokens
                        if token_text not in skip_tokens:
                            print(token_text, end="", flush=True)

                        # Stop decoding if the token signals end-of-message
                        if token_text in skip_tokens:
                            users_decoding[i] = False

                    gen_idx += 1

                _num_decode_tokens += gen_idx * bsz

                for i in range(bsz):
                    tokens_out = [t for t in all_outputs[i] if t not in [tokenizer.pad_token_id]]
                    text = tokenizer.decode(tokens_out)
                    # logger.info(f"[User {i}] Output: {text.strip()}\n")

                    # ✅ Append each user's output directly to the single file
                    if print_to_file and output_path:
                        with open(output_path, "a", encoding="utf-8") as f:
                            f.write(f"[User {i}] Output:\n{text}\n\n")
    if display:
        s.set()
        p.join()


# Main inference function with context
def qwen_inference_with_context(
    mesh_device,
    warmup_iters=0,
    enable_trace=True,
    max_batch_size=1,
    data_parallel=1,
    max_seq_len=4096,
    temperature: float = 0,
    top_p: float = 0.9,
    max_gen_len: Optional[int] = 128,
    print_to_file: bool = True,  # <-- NEW ARG
    output_dir: Optional[str] = "models/bos_model/qwen3_vl/demo/outputs",  # <-- Optional save path
    image_path=None,
    display: bool = True,
):
    """
    Simple multimodal demo with limited dependence on reference code.
    """

    num_devices = mesh_device.get_num_devices() if isinstance(mesh_device, ttnn.MeshDevice) else 1
    max_batch_size *= data_parallel  # input batch_size is interpreted as size per DP group

    model_args, model = prepare_generator_args(
        num_devices=num_devices,
        data_parallel=data_parallel,
        mesh_device=mesh_device,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
    )

    # Prepare output directory and file (only once)
    output_path = None
    if print_to_file:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"qwen3_vl_outputs.txt")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(
                f"Model Outputs (Model Name = {model_args[0].base_model_name}, Batch size = {max_batch_size}, Max gen len = {max_gen_len})\n"
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

        processor = AutoProcessor.from_pretrained(model_args[0].CKPT_DIR)

    generator = Generator(model, model_args, mesh_device)

    xattn_caches = [
        model.setup_cache(model_args[i].max_batch_size) if not HF_MODEL else None
        for i, model in enumerate(generator.model)
    ]

    # --- Warmup run before interactive inference ---
    warmup_model(generator, model_args, xattn_caches, HF_MODEL, processor if HF_MODEL else None)
    # -----------------------------------------------

    # Input image
    trace_img_0 = get_validated_image(image_path)
    if display:
        s = mp.Event()
        p = mp.Process(target=viewer_proc, args=("Input Image", s, np.array(trace_img_0)[:, :, ::-1]), daemon=False)
        p.start()

    max_pairs = 2
    num_entries = max_pairs * 2 + 1  # Each pair has user and assistant, plus the initial user message

    context = []
    input_prompt = "Hi Qwen!"
    print('Qwen: Hello! How can I assist you today? Enter "/bye" to exit.', end="")
    while True:
        input_prompt = input("\n\nMe: ")
        if input_prompt == "/bye":
            print("\nQwen: Thank you for using the service! Goodbye!")
            break

        context.append(f"<|User|> {input_prompt}")
        context = (
            context[-num_entries:] if len(context) >= num_entries else context
        )  # Keep only the last max_pairs of exchanges

        input_text = "\n".join(context)

        trace_dialogs = [
            [
                UserMessage(
                    content=[
                        ImageMedia(image=trace_img_0),
                        input_text,
                    ]
                )
            ],
        ]

        if len(trace_dialogs) < max_batch_size:
            trace_dialogs *= max_batch_size // len(trace_dialogs)

        num_trace_batches = len(trace_dialogs) // max_batch_size
        sampler = get_batch_sampler(temperature, top_p, model_args[0].tokenizer)
        _num_prefill_tokens = 0
        _num_decode_tokens = 0

        prompt_encoder = hf_multimodal_encode if HF_MODEL else formatter.encode_dialog_prompt

        for iter_num in range(warmup_iters + 1):
            current_dialogs = trace_dialogs

            for batch_idx in range(num_trace_batches):
                batch_dialogs = current_dialogs[batch_idx * max_batch_size : (batch_idx + 1) * max_batch_size]

                for dialog in batch_dialogs:
                    for msg in dialog:
                        logger.info(f"{msg.role.capitalize()}: {msg.content}")

                batch_model_input = [
                    prompt_encoder(dialog, processor) if HF_MODEL else prompt_encoder(dialog, tool_prompt_format=False)
                    for dialog in batch_dialogs
                ]

                if HF_MODEL:
                    tokenizer = processor.tokenizer
                    image_grid_thw = [model_input.image_grid_thw for model_input in batch_model_input]
                else:
                    image_grid_thw = None

                # Prefill inputs
                vision_images = [
                    model_input.vision.images if model_input.vision else None for model_input in batch_model_input
                ]
                vision_mask = [
                    model_input.vision.mask if model_input.vision else None for model_input in batch_model_input
                ]
                prompt_tokens = [model_input.tokens for model_input in batch_model_input]
                prefill_lens = torch.tensor([len(tokens) for tokens in prompt_tokens], dtype=torch.long)
                _num_prefill_tokens += prefill_lens.sum().item()
                total_lens = prefill_lens + max_gen_len

                pad_id = tokenizer.pad_token_id if HF_MODEL else tokenizer.pad_id
                bsz = len(prompt_tokens)
                tokens = torch.full((bsz, max(total_lens)), pad_id, dtype=torch.long)
                # for i, seq in enumerate(prompt_tokens):
                #     tokens[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
                for i, seq in enumerate(prompt_tokens):
                    tokens[i, : len(seq)] = seq.detach().clone().to(dtype=torch.long)

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

                next_tokens, next_texts = sampler(batch_logits)
                prefilled_token = next_tokens.clone()

                for i, (next_token, next_text) in enumerate(zip(next_tokens, next_texts)):
                    tokens[i, prefill_lens[i]] = next_token

                all_outputs = [[*t[: prefill_lens[i]].tolist(), next_tokens[i].item()] for i, t in enumerate(tokens)]
                users_decoding = [True] * bsz
                gen_idx = 0
                print_prefill_token = True

                while any(users_decoding) and gen_idx < max_gen_len - 1:
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

                    # Append tokens and decode partial text for preview
                    skip_tokens = {"<|eot_id|>", "<|eom_id|>", "<|im_end|>", "<|im_start|>", "addCriterion"}
                    for i in range(bsz):
                        if not users_decoding[i]:
                            continue
                        user_tok = next_tokens[i].item()
                        all_outputs[i].append(user_tok)
                        text = tokenizer.decode(all_outputs[i])

                        if print_prefill_token:
                            token_text = tokenizer.decode([int(prefilled_token[0].item())])
                            print(f"Qwen: {token_text}", end="", flush=True)
                            print_prefill_token = False

                        # Decode the current token
                        token_text = tokenizer.decode([user_tok])

                        # Only print if it's not in skip_tokens
                        if token_text not in skip_tokens:
                            print(token_text, end="", flush=True)

                        # Stop decoding if the token signals end-of-message
                        if token_text in skip_tokens:
                            users_decoding[i] = False

                    gen_idx += 1

                _num_decode_tokens += gen_idx * bsz

                for i in range(bsz):
                    tokens_out = [t for t in all_outputs[i] if t not in [tokenizer.pad_token_id]]
                    text = tokenizer.decode(tokens_out)

                    prompt_including_assistant_tags = tokenizer.decode(
                        prompt_encoder(trace_dialogs[0], processor).tokens
                    )
                    text_after_prompt = text.replace(prompt_including_assistant_tags, "", 1)
                    context.append(f"<|Qwen|> {text_after_prompt}")

                    # logger.info(f"[User {i}] Output: {text.strip()}\n")

                    # ✅ Append each user's output directly to the single file
                    if print_to_file and output_path:
                        with open(output_path, "a", encoding="utf-8") as f:
                            f.write(f"[User {i}] Output:\n{text}\n\n")
    if display:
        s.set()
        p.join()


def get_mesh_device(param=1, trace_region_size=10419200):
    # Grab all available device IDs
    device_ids = ttnn.get_device_ids()

    if param is None:
        param = len(device_ids)

    # Handle tuple (grid shape) or integer (1×N mesh)
    if isinstance(param, tuple):
        grid_dims = param
        num_devices_requested = grid_dims[0] * grid_dims[1]
        mesh_shape = ttnn.MeshShape(*grid_dims)
    else:
        num_devices_requested = min(param, len(device_ids))
        mesh_shape = ttnn.MeshShape(1, num_devices_requested)

    # Assign trace region size (can add more params if needed)
    device_params = {"trace_region_size": trace_region_size}

    # Open the mesh device
    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape, **device_params)
    return mesh_device


if __name__ == "__main__":
    token_accuracy = False
    mesh_device = {"N150": (1, 1), "N300": (1, 2), "N150x4": (1, 4), "T3K": (1, 8), "TG": (8, 4)}.get(
        os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
    )
    mesh_device = get_mesh_device(mesh_device)

    args = parse_args()

    if args.context:
        qwen_inference_with_context(
            mesh_device, max_gen_len=args.max_generated_tokens, image_path=args.image_path, display=args.display
        )
    else:
        qwen_inference(
            mesh_device, max_gen_len=args.max_generated_tokens, image_path=args.image_path, display=args.display
        )
