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


def create_context_prompt(questions, answers):
    instruction = "<|Instruction|> You are a helpful assistant that provides informative answers based on the conversation history.\n"
    if len(answers) > 0:
        instruction += "Here are the previous questions you were asked, and have been answered.\n\n"
        for i in range(len(questions) - 1):
            instruction += f"<|User|> {questions[i]}\n"
        instruction + "\n<|Instruction|> This was your answer to the last question:\n"
        instruction += f"<|Llama|> {answers[-1]}\n"
        instruction += "\n<|Instruction|> Since you've already answered those questions, now "
    else:
        instruction + "\n<|Instruction|> You have not answered any questions yet. This is the first question.\nSo now, "
    instruction += "provide a brief answer to this latest question:"
    instruction += f"\n<|User|> {questions[-1]}\n"

    return instruction


@pytest.mark.parametrize(
    "instruct, max_seq_len, max_generated_tokens, paged_attention, enable_trace, stop_at_eos, output_filename",
    [
        (True, 1024, 256, True, False, True, None),
    ],
)
@pytest.mark.parametrize(
    "page_params, sampling_params",
    [
        ({"page_block_size": 32, "page_max_num_blocks_per_dp": 1024}, {"temperature": 0, "top_p": 0.08}),
    ],
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "N150x4": (1, 4), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 30000000, "num_command_queues": 1}], indirect=True)
@pytest.mark.parametrize(
    "context_memory, post_process, test_mode",
    [
        (False, False, True),
    ],
)
@pytest.mark.parametrize(
    "input_prompts",
    [
        ["Write numbers from 1 to 5, and nothing else", "/bye"],
        # ["What do you know about Pokemon?",
        #  "Tell me about the different types of Pokemon.",
        #  "List out all the pokemon you know.",
        #  "What was my first question?",
        #  "/bye",
        # ],
    ],
)
def test_llama_inference(
    mesh_device,
    instruct,
    max_seq_len,
    max_generated_tokens,
    paged_attention,
    page_params,
    sampling_params,
    enable_trace,
    stop_at_eos,
    output_filename,
    context_memory,
    post_process,
    input_prompts,
    test_mode,
):
    model_args, model, page_table, tt_kv_cache, tokenizer = load_model(mesh_device)

    generator = Generator(model, model_args, mesh_device, tokenizer=tokenizer)

    questions = []
    answers = []
    input_prompt = "Hi Llama!"
    count = 0
    # print(f"🚀 Running model: {os.environ.get('HF_MODEL', 'HF_MODEL not set')}")
    # print(f"🧠 Maximum Generated tokens: {max_generated_tokens}\n")
    print('\n\U0001F999 Llama: Hello! How can I assist you today? Enter "/bye" to exit.', end="")
    token_count = 0
    total_time = 0
    while True:
        if test_mode:
            input_prompt, count = input_prompts[count], count + 1
            print("\n\n\U0001F464 Me: ", input_prompt)
        else:
            input_prompt, count = input("\n\n\U0001F464 Me: "), count + 1

        if input_prompt == "/bye":
            print("\n\U0001F999 Llama: Thank you for using the service! Goodbye!")
            print(f"Speed: {token_count/total_time:.2f} tok/sec" if total_time > 0 else "")
            break

        questions.append(input_prompt)
        if context_memory:
            input_prompt = create_context_prompt(questions, answers)

        # Preprocess initial prompt inputs
        (
            input_tokens_prefill_pt,
            encoded_prompts,
            decoding_pos,
            prefill_lens,
        ) = preprocess_inputs_prefill(
            [input_prompt], tokenizer, model_args, instruct, max_generated_tokens, max_prefill_len=max_seq_len
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
        logits = generator.prefill_forward_text(
            input_tokens_prefill_pt,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            prompt_lens=decoding_pos,
        )
        prefilled_token = torch.argmax(logits, dim=-1)

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

        start_time = time.time()
        final_output = ""
        print(f"\U0001F999 Llama: ", end="", flush=True)
        while users_decoding:

            def create_out_token(
                generator=generator,
                out_tok=out_tok,
                current_pos=current_pos,
                enable_trace=enable_trace,
                page_table=page_table,
                tt_kv_cache=tt_kv_cache,
                device_sampling_params=device_sampling_params,
                sampling_params=sampling_params,
            ):
                logits = generator.decode_forward_text(
                    out_tok,
                    current_pos,
                    enable_trace=enable_trace,
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
                post_process=post_process,
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
                        print_prefill_token = False

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

            out_tok, current_pos = create_out_token()
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

        answers.append(final_output)
        ttnn.synchronize_device(mesh_device)

        end_time = time.time()
        total_time += end_time - start_time
        # post_processed_output = special_postprocess(final_output)
        # print(post_processed_output, flush=True)
        # print(f"\nTook {total_time:.2f}s for {token_count} tokens")

    if output_filename is not None:
        with open(output_filename, "w") as f:
            for i in range(len(questions)):
                f.write(f"<|User|>: {questions[i]}\n<|Llama|>: {answers[i]}\n\n")
    ttnn.close_mesh_device(mesh_device)
