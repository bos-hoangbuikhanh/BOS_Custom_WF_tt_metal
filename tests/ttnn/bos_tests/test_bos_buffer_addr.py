import ttnn
import torch
import pytest
from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc, assert_equal


# 루프를 돌면서, buffer addr가 같은 tensor에 대하여 반복문에 공유하는 지 체크
@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.bfloat16,
    ],
    ids=["bfloat16"],
)
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
    ],
)
def test_div(shape, dtype, layout, device):
    torch.manual_seed(3)

    if dtype == torch.int32:
        tt_dtype = ttnn.int32
    if dtype == torch.bfloat16:
        tt_dtype = ttnn.bfloat16

    x = torch.randint(low=0, high=10, size=shape).to(dtype)
    dev_x = ttnn.to_layout(ttnn.Tensor(x, tt_dtype).to(device), layout)
    a = hex(dev_x.buffer_address())
    ttnn.deallocate(dev_x)
    dev_x = ttnn.to_layout(ttnn.Tensor(x, tt_dtype).to(device), layout)
    b = hex(dev_x.buffer_address())
    assert a == b
