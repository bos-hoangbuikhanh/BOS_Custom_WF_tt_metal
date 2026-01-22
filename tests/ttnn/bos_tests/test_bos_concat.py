import ttnn
import torch
import pytest
from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc, assert_equal


@pytest.mark.parametrize(
    "shape",
    [
        # [2, 128, 128, 128],
        [2, 480000, 1, 8],
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.bfloat16,
    ],
    ids=["bfloat16"],
)
def test(shape, dtype, device):
    torch.manual_seed(5)

    if dtype == torch.int32:
        tt_dtype = ttnn.int32
    if dtype == torch.bfloat16:
        tt_dtype = ttnn.bfloat16

    x = torch.randint(low=0, high=10, size=shape).to(dtype)
    dev_x = ttnn.to_layout(ttnn.Tensor(x, tt_dtype).to(device), ttnn.ROW_MAJOR_LAYOUT)

    # x = torch.concat(
    #     [
    #         x[0:1],
    #         x[1:2],
    #     ],
    #     dim=0,
    # )
    # dev_x = ttnn.concat(
    #     [
    #         dev_x[0:1],
    #         dev_x[1:2],
    #     ],
    #     dim=0,
    # )

    # ori
    x = torch.concat(
        [
            torch.div(x[0:1], 20),
            torch.div(x[1:2], 12),
        ],
        dim=0,
    )
    dev_x = ttnn.concat(
        [
            ttnn.div(dev_x[0:1], 20),
            ttnn.div(dev_x[1:2], 12),
        ],
        dim=0,
    )

    dev_x = dev_x.cpu().to_torch()
    # print(dev_x)
    # print(x)
    result, msg = assert_with_pcc(dev_x, x)
