import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc, assert_equal
import math

# torch.set_printoptions(threshold=float('inf'), linewidth=200)
# 매핑: ttnn dtype → torch dtype
TTNN_TO_TORCH_DTYPE = {
    ttnn.bfloat16: torch.bfloat16,
    ttnn.bfloat8_b: torch.float32
    # 추가적으로 다른 dtype들도 여기 넣을 수 있음
}


# @pytest.mark.parametrize("shape", [(1, 3, 256, 256), (1, 3, 512, 512), (1, 3, 320, 320), (1, 3, 384, 640) ]) #
@pytest.mark.parametrize("shape", [[1, 1, 100, 512], [1, 1, 2, 2100]])  #
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_to_layout(device, shape, dtype):
    print("This is pytest for bos")

    torch.manual_seed(1231)

    torch_dtype = TTNN_TO_TORCH_DTYPE[dtype]  # convert
    source_tensor = torch.randn(*shape, dtype=torch_dtype)  # shape unpack

    tt_tensor = ttnn.from_torch(
        source_tensor, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    output_tensor = ttnn.to_layout(tt_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.to_torch(output_tensor)
    result, msg = assert_with_pcc(source_tensor, output_tensor)
    print(result)
    print(msg)


# @pytest.mark.parametrize("shape", [(1, 3, 256, 256), (1, 3, 512, 512), (1, 3, 320, 320), (1, 3, 384, 640) ]) #
@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 100, 512],
    ],
)  #  [1,1, 2, 2100]
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b])
def test_shard_to_layout(device, shape, dtype):
    print("This is pytest for bos")

    torch.manual_seed(512123)

    torch_dtype = TTNN_TO_TORCH_DTYPE[dtype]  # convert
    source_tensor = torch.randn(*shape, dtype=torch_dtype)  # shape unpack

    tt_tensor = ttnn.from_torch(
        source_tensor, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    input_mem_config = ttnn.create_sharded_memory_config(
        [1, 1, 32, 128],
        ttnn.CoreGrid(y=4, x=4),
        ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    output_tensor = ttnn.to_memory_config(tt_tensor, memory_config=input_mem_config)
    output_tensor1 = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
    output_tensor1 = ttnn.to_layout(output_tensor1, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor1 = ttnn.to_torch(output_tensor1)
    result, msg = assert_with_pcc(source_tensor, output_tensor1)
    print(result)
    print(msg)
