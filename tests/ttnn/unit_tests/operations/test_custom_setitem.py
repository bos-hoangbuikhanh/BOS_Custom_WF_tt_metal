import torch
import pytest
import ttnn
from loguru import logger
from models.common.utility_functions import comp_pcc
from bos_metal import op, device_box

SHAPE_QUERIES = [
    [16, [0.65, 0.5, 0.25], 10],
    [256, [0.65, 0.5, 0.25], 128],
    [10_000, [0.65, 0.5, 0.25], 3680],
    [10_000, [0.3], 10_000],
]
POS = ["start", "middle", "end"]


def _to_ttnn(inp, device, is_dtype=None):
    if is_dtype is None:
        is_dtype = ttnn.bfloat16
    return ttnn.from_torch(inp, dtype=is_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG, device=device)


def make_inp(shapes):
    len_non_zero, p_list, len_query, dim = shapes
    inp_queries = []
    zero_bev_map = torch.zeros(len_non_zero, dtype=torch.bfloat16)
    for i in range(len(p_list)):
        torch.random.manual_seed(i)
        mask = torch.rand(len_query) > p_list[i]
        inp_query = torch.ones(len_query, dtype=torch.bfloat16)
        inp_query *= mask
        inp_queries.append(inp_query)
    return zero_bev_map, inp_queries


# def make_inp_padded(shapes):
#     len_non_zero, p_list, len_query, dim = shapes
#     inp_queries = []
#     zero_bev_map = torch.zeros((dim, len_query), dtype=torch.bfloat16)  # [N, dim]

#     for i in range(len(p_list)):
#         torch.random.manual_seed(i)

#         # calculate number of zeros first
#         num_zeros = int(len_query * p_list[i])
#         num_ones = len_query - num_zeros

#         # build tensor: [1, ..., 1, 0, ..., 0] for each dimension
#         ones_tensor = torch.ones((dim, num_ones), dtype=torch.bfloat16)
#         zeros_tensor = torch.zeros((dim, num_zeros), dtype=torch.bfloat16)
#         inp_query = torch.cat([ones_tensor, zeros_tensor], dim=1)  # Concatenate along the last dimension
#         inp_query[:, -1] = 1  # Set the last row to 1
#         inp_queries.append(inp_query)

#     return zero_bev_map, inp_queries


def make_inp_padded(shapes):
    len_non_zero, p_list, len_query = shapes
    inp_queries = []
    zero_bev_map = torch.zeros(len_query, dtype=torch.bfloat16)  # [N, dim]

    for i in range(len(p_list)):
        torch.random.manual_seed(i)

        # calculate number of zeros first
        num_zeros = int(len_query * p_list[i])
        num_ones = len_query - num_zeros

        # build tensor: [1, ..., 1, 0, ..., 0] for each dimension
        ones_tensor = torch.ones(num_ones, dtype=torch.bfloat16)
        zeros_tensor = torch.zeros(num_zeros, dtype=torch.bfloat16)
        inp_query = torch.cat([ones_tensor, zeros_tensor], dim=0)  # Concatenate along the last dimension
        inp_query[-1] = 1  # Set the last element to 1
        inp_queries.append(inp_query)

    return zero_bev_map, inp_queries


@pytest.mark.parametrize("shapes", SHAPE_QUERIES)
@pytest.mark.parametrize("device", [device_box.open(enable_program_cache=True)])
def test_bev_mapping(shapes, device):
    ttnn.device.DisablePersistentKernelCache()
    zero_bev_map, inp_queries = make_inp_padded(shapes)
    # print(f'inp queries: {[q for q in inp_queries]}')
    zero_bev_map_tt = _to_ttnn(zero_bev_map, device)
    inp_queries_lst = [_to_ttnn(q, device) for q in inp_queries]
    # print(zero_bev_map.shape, inp_queries[0].shape)
    non_zero_lst = []
    for i in range(len(inp_queries_lst)):
        cnt, non_zero_inp = ttnn.nonzero(inp_queries_lst[i])
        non_zero_lst.append(non_zero_inp)

    for i in range(len(inp_queries_lst)):
        tmp = ttnn.operations.moreh.getitem(zero_bev_map_tt, [non_zero_lst[i]], [0])
        tmp_t = ttnn.to_torch(tmp)
        # print(f'tmp before {i}: {tmp_t}')
        tmp = tmp + inp_queries_lst[i]
        lst = ttnn.to_torch(non_zero_lst[i])
        # print(f'non_zero_lst {i}: {lst}')
        tmp_t = ttnn.to_torch(tmp)
        # print(f'tmp after {i}: {tmp_t}')
        zero_bev_map_tt[non_zero_lst[i]] = tmp
    output = ttnn.to_torch(zero_bev_map_tt)
    # print(f'output: {output}')
    assert output[-1] == 0, f"-1 computed, test failed"
    # if output[-1] == 0:
    #     print('no -1 computed, test passed')
    # else:
    #     print('error: -1 computed, test failed')


@pytest.mark.parametrize("device", [device_box.open(enable_program_cache=True)])
def test_simple(device):
    ttnn.device.DisablePersistentKernelCache()
    lst = torch.zeros(10, dtype=torch.bfloat16)
    tmp = torch.randint(0, 10, (5,), dtype=torch.int32)
    indices = [3, -1, -2, -1, 2]
    t_lst = _to_ttnn(lst, device)
    t_tmp = _to_ttnn(tmp, device)
    t_lst[1] = t_tmp[1]
    out = ttnn.to_torch(t_lst)
    print(f"out: {out}")


if __name__ == "__main__":
    test_bev_mapping(SHAPE_QUERIES[0], device_box.open(enable_program_cache=True))
    # test_simple(device_box.open(enable_program_cache=True))
