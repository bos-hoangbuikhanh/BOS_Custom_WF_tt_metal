import pytest

import ttnn
from tests.ttnn.nightly.unit_tests.operations.pool.test_avgpool2d import run_avg_pool2d


@pytest.fixture(scope="module")
def tensor_map():
    tensor_map = {}

    return tensor_map


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "input_shape",  # NCHW
    ([1, 2048, 16, 32],),
)
@pytest.mark.parametrize(
    "kernel_size",
    ((16, 32),),
)
@pytest.mark.parametrize(
    "stride",
    ((1, 1),),
)
@pytest.mark.parametrize(
    "padding",
    ((0, 0),),
)
@pytest.mark.parametrize(
    "ceil_mode",
    [False],
)
@pytest.mark.parametrize(
    "count_include_pad",
    [False],
)
@pytest.mark.parametrize(
    "divisor_override",
    [
        None,
    ],
)
@pytest.mark.parametrize(
    "shard_scheme",
    [ttnn.TensorMemoryLayout.BLOCK_SHARDED],
)
@pytest.mark.parametrize(
    "use_program_cache",
    [True],
)
def test_avg_pool2d_pdl(
    device,
    use_program_cache,
    tensor_map,
    input_shape,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    divisor_override,
    count_include_pad,
    shard_scheme,
):
    run_avg_pool2d(
        device=device,
        use_program_cache=use_program_cache,
        tensor_map=tensor_map,
        input_shape=input_shape,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        divisor_override=divisor_override,
        count_include_pad=count_include_pad,
        shard_scheme=shard_scheme,
    )
