import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.bos_model.llama32.tt.common import (
    PagedAttentionConfig,
    create_tt_model,
    preprocess_inputs_prefill,
    sample_host,
)
from models.bos_model.llama32.tt.generator import Generator, SamplingParams, create_submeshes
from models.bos_model.llama32.tt.model_config import DecodersPrecision

LLAMA_HOME_PATH = os.path.dirname(os.path.realpath(__file__))
QUERIES_PATH = os.path.join(LLAMA_HOME_PATH, "tests", "queries.txt")


def parse_args(argv=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--device_id", type=int, default=None, help="device id to use for inference")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("-n", "--num_iters", type=int, default=-1, help="number of images to process")

    parser.add_argument("-g", "--max_generated_tokens", default=256, help="Maximum number of generated tokens")
    parser.add_argument("--max_seq_len", default=1024, help="Maximum output sequence length")
    parser.add_argument("--post_process", action="store_true", help="Post process the output to end at sentence end")
    parser.add_argument(
        "--memory_mode",
        type=str,
        default="LOW_CONTEXT",
        choices=["NO_MEMORY", "LOW_CONTEXT", "HIGH_CONTEXT"],
        help="Memory modes: NO_MEMORY has no memory, LOW_CONTEXT (default) stores only last 5 questions, or HIGH_CONTEXT stores all previous questions, and last answer",
    )

    parser.add_argument("--no_trace", action="store_false", help="Disables Trace mode")
    parser.add_argument("--enable_logger", action="store_true", help="Enable Logging")

    parser.add_argument("--queries", nargs="?", default=QUERIES_PATH, type=str, help="Questions to process")
    parser.add_argument(
        "--output_path",
        nargs="?",
        default=None,
        type=str,
        help="Path to save output file with queries, responses, and metrics",
    )
    parser.add_argument("--live", action="store_true", help="Enables live mode for user input")

    args, _ = parser.parse_known_args(argv)
    return args


def create_tt_page_table(global_batch_size, data_parallel, paged_attention_config: PagedAttentionConfig):
    page_table = None

    if paged_attention_config:
        # Implied shuffling of blocks
        permutation = torch.randperm(paged_attention_config.max_num_blocks)
        # Page table which maps virtual blocks to physical
        reverse_permutation = torch.argsort(permutation).repeat(data_parallel)
        page_table = reverse_permutation.reshape(
            global_batch_size, paged_attention_config.max_num_blocks // (global_batch_size // data_parallel)
        )
    return page_table


def load_model(
    mesh_device,
    model_id="meta-llama/Llama-3.2-1B-Instruct",
    instruct=True,
    max_seq_len=1024,
    global_batch_size=1,
    data_parallel=1,
):
    page_params = {"page_block_size": 32, "page_max_num_blocks_per_dp": 1024}
    paged_attention_config = PagedAttentionConfig(
        block_size=page_params["page_block_size"],
        max_num_blocks=page_params["page_max_num_blocks_per_dp"],
    )

    submesh_devices = create_submeshes(mesh_device, data_parallel)

    model_args_list = []
    model_list = []
    tt_kv_cache_list = []
    state_dict = None

    for submesh in submesh_devices:
        model_args_i, model_i, tt_kv_cache_i, state_dict = create_tt_model(
            mesh_device=submesh,
            instruct=instruct,
            max_batch_size=global_batch_size // data_parallel,
            optimizations=lambda margs: DecodersPrecision.performance(margs.n_layers, margs.model_name),
            max_seq_len=max_seq_len,
            paged_attention_config=paged_attention_config,
            dtype=ttnn.bfloat8_b,
            state_dict=state_dict,
        )
        model_args_list.append(model_args_i)
        model_list.append(model_i)
        tt_kv_cache_list.append(tt_kv_cache_i)
        torch.save(state_dict, "model_weights.pth")

    page_table = create_tt_page_table(
        global_batch_size=global_batch_size,
        data_parallel=data_parallel,
        paged_attention_config=paged_attention_config,
    )

    tokenizer = model_args_list[0].tokenizer

    return model_args_list, model_list, page_table, tt_kv_cache_list, tokenizer


def prepare_generator_args(
    num_devices,
    data_parallel,
    mesh_device,
    instruct,
    global_batch_size,
    optimizations,
    max_seq_len,
    page_params,
    paged_attention,
):
    submesh_devices = create_submeshes(mesh_device, data_parallel)
    state_dict = None

    # Hybrid requires a model per submesh
    model_args = []
    model = []
    tt_kv_cache = []

    paged_attention_config = (
        PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks_per_dp"],
        )
        if paged_attention
        else None
    )

    for submesh in submesh_devices:
        model_args_i, model_i, tt_kv_cache_i, state_dict = create_tt_model(
            submesh,
            instruct=instruct,
            max_batch_size=global_batch_size // data_parallel,
            optimizations=optimizations,
            max_seq_len=max_seq_len,
            paged_attention_config=paged_attention_config,
            dtype=ttnn.bfloat8_b,
            state_dict=state_dict,
        )
        model_args.append(model_args_i)
        model.append(model_i)
        tt_kv_cache.append(tt_kv_cache_i)

    page_table = create_tt_page_table(
        global_batch_size=global_batch_size,
        data_parallel=data_parallel,
        paged_attention_config=paged_attention_config,
    )
    # Host code, safe to reuse tokenizer from the 1st model
    tokenizer = model_args[
        0
    ].tokenizer  # TODO Should we support Data Parallel different models? If so, we need to support multiple tokenizers
    return model_args, model, page_table, tt_kv_cache, tokenizer


def get_mesh_device(param=1, device_ids=ttnn.get_device_ids(), trace_region_size=10419200):
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
    device_params = {"trace_region_size": trace_region_size, "num_command_queues": 2}

    # Open the mesh device
    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape, **device_params)
    return mesh_device


def special_postprocess(text, count=0):
    new_text = re.sub(r"(?s)(.*(?<!\d)[.!?]).*", r"\1", text)
    # removed_length = len(text) - len(new_text)

    import shutil

    row_length, window_height = shutil.get_terminal_size()
    # print(f"Height (rows):   {window_height}")
    # print(f"Width (columns): {row_length}")
    num_lines_in_terminal = 0
    for line in text.split("\n"):
        num_lines_in_terminal += 1
        if len(line) > row_length:
            num_lines_in_terminal += len(line) // row_length

    new_text = new_text.replace("<|Instructor|>", "").replace("<|User|>", "").replace("<|Llama|>", "").strip()

    # for _ in range(row_length):
    #     sys.stdout.write("\b")  # move cursor back one space
    #     sys.stdout.flush()
    # for _ in range(num_lines_in_terminal):
    #     sys.stdout.write("\b")  # move cursor back one space
    #     sys.stdout.flush()

    return new_text


def create_context_prompt(queries, responses, num_queries=-1, num_responses=0):
    instruction = "<|Instruction|> You are a helpful assistant that provides informative answers based on the conversation history.\n"
    if len(responses) > 0:
        if num_queries == -1:
            instruction += "\nHere are all the previous questions you were asked, and have been answered.\n\n"
            for i in range(len(queries) - 1):
                instruction += f"<|User|> {queries[i]}\n"
        else:
            start_index = (len(queries) - num_queries - 1) if (len(queries) - num_queries - 1) >= 0 else 0
            instruction += f"\nHere are the last {len(queries)-start_index-1} questions you were asked, and have been answered.\n\n"
            for i in range(start_index, len(queries) - 1):
                instruction += f"<|User|> {queries[i]}\n"
        if num_responses == 1:
            instruction += "\n<|Instruction|> This was your answer to the last question:\n"
            instruction += f"<|Llama|> {responses[-1]}\n"
            instruction += "\n<|Instruction|> Since you've already answered this question, now "
        elif num_responses > 1:
            instruction += f"\n<|Instruction|> Here are your previous answers to the last {num_responses} questions:\n\n"
            for i in range(num_responses - 1, 0, -1):
                instruction += f"<|Llama|> {responses[-i]}\n"
            instruction += "\n<|Instruction|> Since you've already answered those questions, now "
        else:
            instruction += "\n<|Instruction|> Since you've already answered those questions, now "
            start_index = (len(queries) - num_queries) if (len(queries) - num_queries) >= 0 else 0
    else:
        instruction += (
            "\n<|Instruction|> You have not answered any questions yet. This is the first question.\nSo now, "
        )
    instruction += "provide a brief answer to the user's latest question:"
    instruction += f"\n<|User|> {queries[-1]}\n"

    return instruction


def llama_runner(device_id=None, batch_size=1, num_iters=5, **kwargs):
    assert batch_size in [1], "Only batch size 1 is supported in Llama32"
    if not kwargs.get("enable_logger", True):
        logger.remove()

    mesh_device = {"N150": (1, 1), "N300": (1, 2), "N150x4": (1, 4), "T3K": (1, 8), "TG": (8, 4)}.get(
        os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
    )
    if device_id is not None:
        mesh_device = get_mesh_device(mesh_device, device_ids=[device_id])
    else:
        mesh_device = get_mesh_device(mesh_device)

    model_args, model, page_table, tt_kv_cache, tokenizer = load_model(mesh_device)

    generator = Generator(model, model_args, mesh_device, tokenizer=tokenizer)

    queries = []
    responses = []
    input_prompt = "Hi Llama!"
    count = 0
    # print(f"🚀 Running model: {os.environ.get('HF_MODEL', 'HF_MODEL not set')}")
    # print(f"🧠 Maximum Generated tokens: {max_generated_tokens}\n")
    print('\n\U0001F999 Llama: Hello! How can I assist you today? Enter "/bye" to exit.', end="")
    token_count = 0
    token_count_tracker = []
    total_time = 0
    prefill_time_list = []
    effective_ttft_list = []
    response_times_list = []

    # output_times = []

    tps = 0.0
    ttft = 0
    while True:
        token_count_tracker.append(token_count)
        if not kwargs.get("live_mode"):
            input_prompt, count = kwargs.get("queries")[count], count + 1
            print("\n\n\U0001F464 Me: ", input_prompt)
        else:
            if len(queries) == num_iters - 1:
                print(
                    f"\n\n\U0001F999 Llama: Thank you for using the service! You've asked {len(queries)} questions. Goodbye!"
                )
                break
            input_prompt, count = input("\n\n\U0001F464 Me: "), count + 1

        effective_start_time = time.time()

        if input_prompt == "/bye":
            print("\n\U0001F999 Llama: Thank you for using the service! Goodbye!")
            break

        queries.append(input_prompt)
        if kwargs.get("memory_mode") == "LOW_CONTEXT":
            input_prompt = create_context_prompt(queries, responses, num_queries=5)
        elif kwargs.get("memory_mode") == "HIGH_CONTEXT":
            input_prompt = create_context_prompt(queries, responses, num_responses=1)

        # Preprocess initial prompt inputs
        max_generated_tokens = kwargs.get("max_generated_tokens", 256)
        max_seq_len = kwargs.get("max_seq_len", 1024)
        paged_attention = kwargs.get("paged_attention", True)
        page_params = {"page_block_size": 32, "page_max_num_blocks_per_dp": 1024}
        sampling_params = kwargs.get("sampling_params", {"temperature": 0, "top_p": 0.08})
        stop_at_eos = kwargs.get("stop_at_eos", True)
        (
            input_tokens_prefill_pt,
            encoded_prompts,
            decoding_pos,
            prefill_lens,
        ) = preprocess_inputs_prefill(
            [input_prompt],
            tokenizer,
            model_args,
            kwargs.get("instruct"),
            max_generated_tokens,
            max_prefill_len=max_seq_len,
        )
        max_encoded_prompt_len = max(len(p) for p in encoded_prompts)
        assert (
            max_generated_tokens + max_encoded_prompt_len <= max_seq_len
        ), f"Prompt prefill tokens ({max_encoded_prompt_len}) + maximum number of decoded iterations ({max_generated_tokens}) needs to be <= than max_seq_len ({max_seq_len})"

        if paged_attention:
            paged_cache_max_seq_len = page_params["page_block_size"] * page_params["page_max_num_blocks_per_dp"]
            assert (
                max_generated_tokens + max_encoded_prompt_len <= paged_cache_max_seq_len
            ), f"max_generated_tokens ({max_generated_tokens}) needs to be <= than paged_cache_max_seq_len ({paged_cache_max_seq_len})"

        # Setting KV-cache to zero for each new prompt
        for i in range(len(model)):
            for layer in model[i].layers:
                k_cache, v_cache = layer.attention.layer_past
                k_cache = ttnn.mul(k_cache, 0, output_tensor=k_cache)
                v_cache = ttnn.mul(v_cache, 0, output_tensor=v_cache)

        input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(1, -1)

        # Compile Pre-fill
        logits = generator.prefill_forward_text(
            input_tokens_prefill_pt,  # Prefill warmup for all users, in case some users have different seqlens than others
            page_table=page_table,
            kv_cache=tt_kv_cache,
            prompt_lens=decoding_pos,
        )

        # Inference Pre-fill
        prefill_start_time = time.time()
        logits = generator.prefill_forward_text(
            input_tokens_prefill_pt,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            prompt_lens=decoding_pos,
        )
        prefilled_token = torch.argmax(logits, dim=-1)
        prefill_end_time = time.time()
        prefill_time_list.append(prefill_end_time - prefill_start_time)

        all_outputs = [encoded_prompts[0][: prefill_lens[0]]]
        user_tok = int(prefilled_token[0].item())
        all_outputs[0].append(user_tok)
        user_done = [False]

        argmax_on_device = sampling_params["temperature"] == 0
        if argmax_on_device:
            device_sampling_params = SamplingParams(temperature=0.0, top_k=-1, top_p=1.0)
        else:
            device_sampling_params = None

        # Start decoding
        current_pos = torch.tensor([decoding_pos[0]])
        iteration = 0
        users_decoding = True
        out_tok = prefilled_token
        print_prefill_token = True

        response_start_time = time.time()
        final_output = ""
        op_event = ttnn.record_event(mesh_device, 0)
        print(f"\U0001F999 Llama: ", end="", flush=True)
        while users_decoding:

            def create_out_token(
                generator=generator,
                out_tok=out_tok,
                current_pos=current_pos,
                enable_trace=kwargs.get("trace"),
                page_table=page_table,
                tt_kv_cache=tt_kv_cache,
                device_sampling_params=device_sampling_params,
                sampling_params=sampling_params,
            ):
                logits = generator.decode_forward_text(
                    out_tok,
                    current_pos,
                    enable_trace=kwargs.get("trace"),
                    page_table=page_table,
                    kv_cache=tt_kv_cache,
                    sampling_params=device_sampling_params,
                )
                if device_sampling_params is not None:
                    out_tok = logits.unsqueeze(1)
                else:
                    _, out_tok = sample_host(
                        logits,
                        temperature=sampling_params["temperature"],
                        top_p=sampling_params["top_p"],
                        on_host=True,
                    )
                return out_tok, current_pos + 1

            def decode_out_token(
                out_tok,
                all_outputs=all_outputs,
                user_done=user_done,
                iteration=iteration,
                max_generated_tokens=max_generated_tokens,
                stop_at_eos=stop_at_eos,
                print_prefill_token=print_prefill_token,
                tokenizer=tokenizer,
                final_output=final_output,
                post_process=kwargs.get("post_process"),
                users_decoding=users_decoding,
                token_text="",
            ):
                # Save output token to print out later
                user_tok = out_tok[0].item()
                if (
                    user_tok not in tokenizer.stop_tokens and user_done[0] == False
                ):  # Read until an eos token (e.g. <|eot_id|>); create_tokenizer adds stop_tokens to HF tokenizers
                    all_outputs[0].append(user_tok)

                    if print_prefill_token:
                        token_text = tokenizer.decode([int(prefilled_token[0].item())])
                        # print(f"\U0001F999 Llama: {token_text}", end="", flush=True) if not post_process else None
                        # output_end_time = time.time()
                        # print(output_end_time - output_start_time)
                        # output_times.append(output_end_time - output_start_time)
                        print_prefill_token = False
                        effective_ttft_list.append(time.time() - effective_start_time)

                    token_text += tokenizer.decode([user_tok])
                    # print(token_text, end="", flush=True) if not post_process else None

                    return token_text, print_prefill_token

                else:
                    if (
                        stop_at_eos
                    ):  # For performance gathering in CI, we want to sometimes force decoding for a fixed number of iterations
                        user_done[0] = True
                        if all(user_done):
                            return "\end", print_prefill_token

            ttnn.wait_for_event(1, op_event)
            out_tok, current_pos = create_out_token()
            write_event = ttnn.record_event(mesh_device, 1)
            ttnn.wait_for_event(0, write_event)
            token_text, print_prefill_token = decode_out_token(out_tok, final_output=final_output)
            if token_text == "\end":
                if (token_count < max_generated_tokens) and (not final_output.endswith(".")):
                    final_output += "."
                final_output = special_postprocess(final_output, count=count)
                users_decoding = False
                break
            else:
                print(token_text, end="", flush=True)
                final_output += token_text
            if iteration > max_generated_tokens:
                users_decoding = False

            token_count += 1

            iteration += 1
            op_event = ttnn.record_event(mesh_device, 0)

        responses.append(final_output)
        ttnn.synchronize_device(mesh_device)

        response_end_time = time.time()
        total_time += response_end_time - response_start_time
        response_times_list.append(response_end_time - response_start_time)
        # post_processed_output = special_postprocess(final_output)
        # print(post_processed_output, flush=True)
        # print(f"\nTook {total_time:.2f}s for {token_count} tokens")

    total_time = sum(response_times_list)
    tps = token_count / total_time if total_time > 0 else 0
    print(f"Speed: {tps:.2f} tok/sec" if tps > 0 else "")
    token_counts = [token_count_tracker[i] - token_count_tracker[i - 1] for i in range(1, len(token_count_tracker))]
    tps_list = [token_counts[i] / response_times_list[i] for i in range(len(token_counts))]

    final_output = f"Model Configuration: {os.environ.get('HF_MODEL')}\n"
    # model_name = os.environ.get("HF_MODEL").split("/")[-1]
    final_output += f"Model Name: {os.environ.get('HF_MODEL')}\n"
    final_output += f"Maximum Sequence length: {kwargs.get('max_seq_len')}\n"
    final_output += f"Maximum Generated tokens: {kwargs.get('max_generated_tokens')}\n"
    final_output += f"Memory Mode: {kwargs.get('memory_mode')}\n\n\n"
    for i in range(len(queries)):
        final_output += f"<|User|>: {queries[i]}\n<|Llama|>: {responses[i]}\n\n"
    final_output += f"\n\n--- Performance Metrics ---\n"
    final_output += f"Overall Speed: {tps:.2f} tok/sec\n"
    final_output += f"Prefill Times (s): [{', '.join([f'{x:.4f}' for x in prefill_time_list])}]\n"
    final_output += f"Effective TTFT Times (s): [{', '.join([f'{x:.4f}' for x in effective_ttft_list])}]\n"
    final_output += f"Token counts per query: [{', '.join([str(x) for x in token_counts])}]\n"
    final_output += f"Response Times (s): [{', '.join([f'{x:.4f}' for x in response_times_list])}]\n"

    if kwargs.get("output_filename") is not None:
        with open(kwargs.get("output_filename"), "w") as f:
            f.write(final_output)
        print(f"Output written to {kwargs.get('output_filename')}")

    ttnn.close_mesh_device(mesh_device)

    return {
        "tps": tps_list,
        "ttft": prefill_time_list,
        "output": final_output,
    }


if __name__ == "__main__":
    args = parse_args()

    with open(args.queries, "r", encoding="utf-8") as f:
        queries = [line for line in f.read().splitlines() if not line.startswith("#") and line != ""]
    if args.num_iters == -1:
        num_iters = len(queries)
    else:
        num_iters = min(args.num_iters, len(queries))
    queries = queries[:num_iters] + ["/bye"]
    num_iters += 1

    cfg = {
        "instruct": True,
        "max_seq_len": args.max_seq_len,
        "max_generated_tokens": args.max_generated_tokens,
        "paged_attention": True,
        "page_params": {"page_block_size": 32, "page_max_num_blocks_per_dp": 1024},
        "sampling_params": {"temperature": 0, "top_p": 0.08},
        "trace": not args.no_trace,
        # "2cq": args.cq2,
        "stop_at_eos": True,
        "memory_mode": args.memory_mode,
        "enable_logger": args.enable_logger,
        "queries": queries,
        "post_process": args.post_process,
        "output_filename": args.output_path,
        "live_mode": args.live,
    }

    llama_runner(device_id=args.device_id, batch_size=args.batch_size, num_iters=num_iters, **cfg)
