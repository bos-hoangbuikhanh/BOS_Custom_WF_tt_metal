import ttnn
import torch
import pytest
from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc, assert_equal


@pytest.mark.parametrize(
    "shape",
    [
        [2, 32, 32, 32],
        [2, 32, 64, 64],
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

    tmp_x = x[1:2] / 12
    dev_x1 = ttnn.div(dev_x[1:2], 12)
    dev_x2 = ttnn.div(dev_x[1:2], 12)

    tmp = dev_x2.cpu().to_torch()

    # print(x)
    dev_x = dev_x.cpu().to_torch()
    result, msg = assert_with_pcc(dev_x, x)
    result2, msg2 = assert_with_pcc(tmp_x, tmp)
