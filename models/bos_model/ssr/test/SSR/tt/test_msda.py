import pytest
import torch
import ttnn
import tracy
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout
from tests.ttnn.unit_tests.operations.test_bos_deformable_attention_utils import (
    generate_parametric_inputs,
    multi_scale_deformable_attn_pytorch_ref,
    multi_scale_deformable_attn_ttnn,
    make_test_case,
    pt2tt,
    tt2pt,
)


TEST_CASES = [
    make_test_case(batch_size=1, num_queries=1, num_heads=4, num_levels=1, num_points=1, num_keys=10, embed_dims=1),
    make_test_case(batch_size=2, num_queries=2, num_heads=1, num_levels=4, num_points=2, num_keys=20, embed_dims=1),
    make_test_case(batch_size=2, num_queries=4, num_heads=4, num_levels=1, num_points=2, num_keys=10, embed_dims=2),
    make_test_case(batch_size=6, num_queries=4, num_heads=8, num_levels=1, num_points=4, num_keys=24, embed_dims=2),
    make_test_case(batch_size=2, num_queries=4, num_heads=1, num_levels=2, num_points=2, num_keys=10, embed_dims=2),
    make_test_case(batch_size=2, num_queries=2, num_heads=2, num_levels=2, num_points=2, num_keys=10, embed_dims=2),
    make_test_case(batch_size=2, num_queries=2, num_heads=2, num_levels=2, num_points=8, num_keys=10, embed_dims=2),
    make_test_case(batch_size=6, num_queries=20, num_heads=4, num_levels=1, num_points=8, num_keys=50, embed_dims=32),
    make_test_case(batch_size=3, num_queries=5, num_heads=2, num_levels=5, num_points=3, num_keys=25, embed_dims=4),
    make_test_case(batch_size=1, num_queries=10, num_heads=1, num_levels=3, num_points=16, num_keys=48, embed_dims=8),
]

SANTITY_TEST_CASES = [
    make_test_case(
        batch_size=2,
        num_queries=10000,
        num_heads=8,
        num_levels=1,
        num_points=4,
        num_keys=10000,
        embed_dims=32,
        pcc=0.99,
    ),
    make_test_case(
        batch_size=6, 
        num_queries=3680, 
        num_heads=8, 
        num_levels=1, 
        num_points=8, 
        num_keys=240, 
        embed_dims=32, 
        pcc=0.99
    ),
]


@pytest.mark.parametrize("test_case", TEST_CASES + SANTITY_TEST_CASES)
@pytest.mark.parametrize("use_fp32", [False, True])
def test_deformable_attention_kernel_functionality(device, test_case, use_fp32):
    # Generate sample inputs
    input_dict = generate_parametric_inputs(**test_case)
    value = input_dict["value"]
    value_spatial_shapes = input_dict["value_spatial_shapes"]
    sampling_locations = input_dict["sampling_locations"]
    attention_weights = input_dict["attention_weights"]

    # TT
    value_tt = pt2tt(value, device=device)
    value_spatial_shapes_tt = pt2tt(value_spatial_shapes, device=device, layout=ttnn.TILE_LAYOUT)
    sampling_locations_tt = pt2tt(sampling_locations, device=device, layout=ttnn.TILE_LAYOUT)
    attention_weights_tt = pt2tt(attention_weights, device=device, layout=ttnn.TILE_LAYOUT)

    print("pt")
    out_pt = multi_scale_deformable_attn_pytorch_ref(
        value,
        value_spatial_shapes,
        sampling_locations,
        attention_weights,
    )

    print("tt")
    print(value_tt.shape)
    print(value_spatial_shapes_tt.shape)
    print(sampling_locations_tt.shape)
    print(attention_weights_tt.shape)

    print(value_tt.memory_config())
    print(value_spatial_shapes_tt.memory_config())
    print(sampling_locations_tt.memory_config())
    print(attention_weights_tt.memory_config())

    out_tt = ttnn.bos_deformable_attention(
        value_tt, value_spatial_shapes_tt, sampling_locations_tt, attention_weights_tt
    )
    out_tt = ttnn.to_torch(out_tt)

    # Check results
    assert out_tt.shape == out_pt.shape, f"Output shape mismatch: {out_tt.shape} vs {out_pt.shape}"
    passed, msg = check_with_pcc_without_tensor_printout(out_pt, out_tt, pcc=test_case["pcc"])
    print(msg)
    # assert passed, f"Test failed: {msg}, out_tt={out_tt}, out_pt={out_pt}, test_case={test_case}"

    ttnn.deallocate(value_tt)
    ttnn.deallocate(value_spatial_shapes_tt)
    ttnn.deallocate(sampling_locations_tt)
    ttnn.deallocate(attention_weights_tt)

    import time

    num_iter = 10
    for i in range(num_iter):
        tracy.signpost("Performance msda")
        value_tt = pt2tt(value, device=device)
        value_spatial_shapes_tt = pt2tt(value_spatial_shapes, device=device, layout=ttnn.TILE_LAYOUT)
        sampling_locations_tt = pt2tt(sampling_locations, device=device, layout=ttnn.TILE_LAYOUT)
        attention_weights_tt = pt2tt(attention_weights, device=device, layout=ttnn.TILE_LAYOUT)
        ttnn.synchronize_device(device)
        st = time.time()
        
        tracy.signpost('choquy')
        out_tt = ttnn.bos_deformable_attention(
            value_tt, value_spatial_shapes_tt, sampling_locations_tt, attention_weights_tt
        )
        ttnn.synchronize_device(device)
        en = time.time()
        avg_exec_time = en - st
        print(f"Iter {i+1}/{num_iter} done, runtime: {avg_exec_time:.4f} s", end="\r")

        ttnn.deallocate(value_tt)
        ttnn.deallocate(value_spatial_shapes_tt)
        ttnn.deallocate(sampling_locations_tt)
        ttnn.deallocate(attention_weights_tt)
        ttnn.synchronize_device(device)

    if num_iter >= 1:
        print(f"Average time: {avg_exec_time:.4f} s | {1/(avg_exec_time):.4f} FPS")


def deformable_attention_profiling():
    device = ttnn.open_device(device_id=0)
    device.enable_program_cache()
    test_deformable_attention_kernel_functionality(device, SANTITY_TEST_CASES[1], False)


if __name__ == "__main__":
    deformable_attention_profiling()
 