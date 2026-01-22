import ttnn

CORE_SET_14 = ttnn.CoreRangeSet(
    {
        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 1)),
        ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(3, 2)),
    }
)
CORE_SET_16 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))})
CORE_GRID_16 = ttnn.CoreGrid(x=4, y=4)

# module_conf: ttnn operation's parameters,
# [reshard_if_not_optimal, height_sharding, packer_l1_acc, enable_act_double_buffer, force_split_reader, ops_parallel_config]
LAYER_CONFIG = {
    "layer1": {
        "module1": [False, True, True, True, True, None],
        "module2": [False, None, True, False, True, None],
        "module3": [False, None, True, False, True, None],
    },
    "layer2": {
        "module1": [False, True, True, True, True, None],
        "module2": [False, None, True, True, True, None],
        "module3": [False, None, True, True, True, None],
        "module4": [False, None, True, True, True, None],
    },
    "layer3": {
        "module1": [True, False, True, True, False, None],
        "module2": [False, None, True, True, False, None],
        "module3": [False, None, True, True, False, None],
        "module4": [False, None, True, True, False, None],
        "module5": [False, None, True, True, False, None],
        "module6": [False, None, True, True, False, None],
    },
    "layer4": {
        "module1": [False, False, True, True, False, None],
        "module2": [False, None, True, True, False, None],
        "module3": [False, None, True, True, False, None],
    },
}
