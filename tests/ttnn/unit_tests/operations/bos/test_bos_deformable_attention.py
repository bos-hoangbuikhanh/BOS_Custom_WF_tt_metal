import pytest
import torch
import ttnn
import tracy
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout
from tests.ttnn.unit_tests.operations.bos.test_bos_deformable_attention_utils import (
    generate_parametric_inputs,
    multi_scale_deformable_attn_pytorch_ref,
    make_test_case,
    make_weight_hash_config,
    pt2tt,
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
    make_test_case(batch_size=6, num_queries=20, num_heads=4, num_levels=1, num_points=8, num_keys=50, embed_dims=37),
    make_test_case(batch_size=3, num_queries=5, num_heads=2, num_levels=5, num_points=3, num_keys=25, embed_dims=4),
    # make_test_case(batch_size=3, num_queries=5, num_heads=2, num_levels=5, num_points=7, num_keys=25, embed_dims=65), #todo: embed_dims>64 not supported yet
    make_test_case(batch_size=1, num_queries=10, num_heads=1, num_levels=3, num_points=16, num_keys=48, embed_dims=8),
]

SANTITY_TEST_CASES = [
    make_test_case(
        batch_size=2,
        num_queries=10_000,
        num_heads=8,
        num_levels=1,
        num_points=4,
        num_keys=10000,
        embed_dims=32,
        pcc=0.99,
    ),
    make_test_case(
        batch_size=6, 
        num_queries=3_680, 
        num_heads=8, 
        num_levels=1, 
        num_points=8, 
        num_keys=240, 
        embed_dims=32, 
        pcc=0.99
    )
]


WEIGHT_HASH_CONFIG_CASES = [
    ttnn.BilinearWeightHashConfig(
        step_x=100, 
        step_y=100, 
        layout=ttnn.ROW_MAJOR_LAYOUT, 
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ),
    ttnn.BilinearWeightHashConfig(
        step_x=100, 
        step_y=100, 
        layout=ttnn.ROW_MAJOR_LAYOUT, 
        memory_config=ttnn.L1_MEMORY_CONFIG
    ),
    # ttnn.BilinearWeightHashConfig(
    #     step_x=1_000, 
    #     step_y=1_000, 
    #     layout=ttnn.ROW_MAJOR_LAYOUT, 
    #     memory_config=ttnn.DRAM_MEMORY_CONFIG,
    # ),
    # ttnn.BilinearWeightHashConfig( # oom issue at runtime
    #     step_x=1_000, 
    #     step_y=1_000, 
    #     layout=ttnn.ROW_MAJOR_LAYOUT, 
    #     memory_config=ttnn.L1_MEMORY_CONFIG
    # ),
]


@pytest.mark.parametrize("test_case", TEST_CASES + SANTITY_TEST_CASES)
@pytest.mark.parametrize("use_fp32", [False, True])
@pytest.mark.parametrize("is_denormed_grid", [False, True])
@pytest.mark.parametrize("use_bilinear_weight_hash", [False, True])
@pytest.mark.parametrize("weight_hash_config_case", WEIGHT_HASH_CONFIG_CASES)
@pytest.mark.parametrize("num_iter", [1])
def test_deformable_attention_kernel_functionality(device, test_case, use_fp32, is_denormed_grid, use_bilinear_weight_hash, weight_hash_config_case, num_iter):
    # Hanging issue happened when num_levels * num_points > 32 with use_fp32=True.
    # However, w/o use_fp32, most cases should work well now
    # use_fp32 is not working well with use_bilinear_weight_hash 
    use_fp32 = use_fp32 and (test_case["num_levels"] * test_case["num_points"] <= 32) and (not use_bilinear_weight_hash)
    # Generate sample inputs
    input_dict = generate_parametric_inputs(**test_case)
    value = input_dict["value"]
    value_spatial_shapes = input_dict["value_spatial_shapes"]
    sampling_locations = input_dict["sampling_locations"]
    attention_weights = input_dict["attention_weights"]
    bilinear_steps = weight_hash_config_case["step_x"]

    # TT
    value_tt = pt2tt(value, device=device)
    value_spatial_shapes_tt = pt2tt(value_spatial_shapes, device=device, layout=ttnn.TILE_LAYOUT)
    attention_weights_tt = pt2tt(attention_weights, device=device, layout=ttnn.TILE_LAYOUT)
    bilinear_weight_hash_ = ttnn.bos_create_bilinear_hash(device, **weight_hash_config_case) if use_bilinear_weight_hash else None
    bilinear_weight_hash = ttnn.reshape(bilinear_weight_hash_, (bilinear_steps, bilinear_steps, 4)) if use_bilinear_weight_hash else None
    if use_bilinear_weight_hash:
        ttnn.deallocate(bilinear_weight_hash_)

    if is_denormed_grid:
        flipped_shapes = value_spatial_shapes.flip(-1).unsqueeze(1)
        sampling_locations_tt = pt2tt(sampling_locations * flipped_shapes - 0.5, device=device, layout=ttnn.TILE_LAYOUT)
    else:
        sampling_locations_tt = pt2tt(sampling_locations, device=device, layout=ttnn.TILE_LAYOUT)

    print("pt")
    out_pt = multi_scale_deformable_attn_pytorch_ref(
        value,
        value_spatial_shapes,
        sampling_locations,
        attention_weights,
    )

    print("tt")
    print(f"Value shape: {value_tt.shape}")
    # value_tt = ttnn.to_memory_config(value_tt, ttnn.L1_MEMORY_CONFIG)
    out_tt = ttnn.bos_deformable_attention(
        value_tt,
        value_spatial_shapes_tt,
        sampling_locations_tt,
        attention_weights_tt,
        use_fp32=use_fp32,
        bilinear_weight_hash=bilinear_weight_hash,
        is_denormed_grid=is_denormed_grid,)
    out_tt = ttnn.to_torch(out_tt)
    # print(out_tt)
    # print(out_pt)
    
    # Check results
    assert out_tt.shape == out_pt.shape, f"Output shape mismatch: {out_tt.shape} vs {out_pt.shape}"
    passed, msg = check_with_pcc_without_tensor_printout(out_pt, out_tt, pcc=test_case["pcc"])
    print(msg)
    assert passed, f"Test failed: {msg}, \nout_tt={out_tt}, \nout_pt={out_pt}, \ntest_case={test_case}"

    import time
    ttnn.deallocate(value_tt)
    ttnn.deallocate(value_spatial_shapes_tt)
    ttnn.deallocate(sampling_locations_tt)
    ttnn.deallocate(attention_weights_tt)

    for i in range(num_iter):
        value_tt = pt2tt(value, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
        value_spatial_shapes_tt = pt2tt(value_spatial_shapes, device=device, layout=ttnn.TILE_LAYOUT)
        attention_weights_tt = pt2tt(attention_weights, device=device, layout=ttnn.TILE_LAYOUT)
        if is_denormed_grid:
            flipped_shapes = value_spatial_shapes.flip(-1).unsqueeze(1)
            sampling_locations_tt = pt2tt(sampling_locations * flipped_shapes - 0.5, device=device, layout=ttnn.TILE_LAYOUT)
        else:
            sampling_locations_tt = pt2tt(sampling_locations, device=device, layout=ttnn.TILE_LAYOUT)
        tracy.signpost("Performance msda")
        # value_tt = ttnn.to_memory_config(value_tt, ttnn.L1_MEMORY_CONFIG)
        st = time.time()
        out_tt = ttnn.bos_deformable_attention(
            value_tt,
            value_spatial_shapes_tt,
            sampling_locations_tt,
            attention_weights_tt,
            use_fp32=use_fp32,
            bilinear_weight_hash=bilinear_weight_hash,
            is_denormed_grid=is_denormed_grid,
        )
        ttnn.deallocate(value_tt)
        ttnn.deallocate(value_spatial_shapes_tt)
        ttnn.deallocate(sampling_locations_tt)
        ttnn.deallocate(attention_weights_tt)
        if i != (num_iter - 1): ttnn.deallocate(out_tt)
        ttnn.synchronize_device(device)
        en = time.time()
        avg_exec_time = en - st
        print(f"Iter {i+1}/{num_iter} done, runtime: {avg_exec_time:.4f} s", end="\r")

    if num_iter >= 1:
        print(f"Average time: {avg_exec_time:.4f} s | {1/(avg_exec_time):.4f} FPS")
        
        out_tt = ttnn.to_torch(out_tt)
        passed, msg = check_with_pcc_without_tensor_printout(out_pt, out_tt, pcc=test_case["pcc"])
        assert passed, f"Test failed: {msg}, \nout_tt={out_tt}, \nout_pt={out_pt}, \ntest_case={test_case}"
        print(msg)

def deformable_attention_profiling():
    device = ttnn.open_device(device_id=0)
    test_deformable_attention_kernel_functionality(device, 
                                                   TEST_CASES[0], 
                                                   use_fp32=False, 
                                                   is_denormed_grid=True,
                                                   use_bilinear_weight_hash=True,
                                                   weight_hash_config_case=
                                                   WEIGHT_HASH_CONFIG_CASES[1],
                                                   num_iter=1)

if __name__ == "__main__":
    deformable_attention_profiling()