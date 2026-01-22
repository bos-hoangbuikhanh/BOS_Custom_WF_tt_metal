import ttnn

module_config = {
    "conv1": {
        "config": {
            "dtype": ttnn.bfloat16,
            "weights_dtype": ttnn.bfloat8_b,
            "activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            "act_block_h_override": 32*12,
            "deallocate_activation": True,
            "reallocate_halo_output": True,
            #"in_place": False,
            "reshard_if_not_optimal": False,
            "override_sharding_config": True,
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "core_grid": ttnn.CoreRangeSet([
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(4, 3)
                ),
            ]),
            "transpose_shards": False,
            "output_layout": ttnn.Layout.TILE,
            "enable_act_double_buffer": True,
            "force_split_reader": True,
            # "enable_subblock_padding": False,
        },
    },
    "layer1": {
        "0": {
            "conv1": {
                "config": {
                    "dtype": ttnn.bfloat8_b,
                    "weights_dtype": ttnn.bfloat8_b,
                    "activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                    "act_block_h_override": 0,
                    "deallocate_activation": True,
                    "reallocate_halo_output": False,
                    #"in_place": False,
                    "reshard_if_not_optimal": False,
                    "override_sharding_config": True,
                    "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(4, 3)
                        ),
                    ]),
                    "transpose_shards": False,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": True,
                    # "enable_subblock_padding": False,
                },
            },
            "conv2": {
                "config": {
                    "dtype": ttnn.bfloat8_b,
                    "weights_dtype": ttnn.bfloat8_b,
                    "activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                    "act_block_h_override": 32*12,
                    "deallocate_activation": True,
                    "reallocate_halo_output": False,
                    #"in_place": False,
                    "reshard_if_not_optimal": False,
                    "override_sharding_config": True,
                    "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(4, 3)
                        ),
                    ]),
                    "transpose_shards": False,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": True,
                    # "enable_subblock_padding": False,
                },
            },
            "conv3": {
                "config": {
                    "dtype": ttnn.bfloat8_b,
                    "weights_dtype": ttnn.bfloat8_b,
                    "activation": None,
                    "act_block_h_override": 0,
                    "deallocate_activation": True,
                    "reallocate_halo_output": False,
                    #"in_place": False,
                    "reshard_if_not_optimal": False,
                    "override_sharding_config": True,
                    "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(4, 3)
                        ),
                    ]),
                    "transpose_shards": False,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": True,
                    # "enable_subblock_padding": False,
                },
            },
            "downsample": {
                "0": {
                    "init_config": {
                        "output_config": [
                            {
                                "reallocate": True,
                            }
                        ]
                    },
                    "config": {
                        "dtype": ttnn.bfloat16,
                        "weights_dtype": ttnn.bfloat8_b,
                        "activation": None,
                        "act_block_h_override": 0,
                        "deallocate_activation": False,
                        "reallocate_halo_output": False,
                        #"in_place": False,
                        "reshard_if_not_optimal": False,
                        "override_sharding_config": True,
                        "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                        "core_grid": ttnn.CoreRangeSet([
                            ttnn.CoreRange(
                                ttnn.CoreCoord(0, 0),
                                ttnn.CoreCoord(4, 3)
                            ),
                        ]),
                        "transpose_shards": False,
                        "output_layout": ttnn.Layout.TILE,
                        "enable_act_double_buffer": True,
                        "force_split_reader": True,
                        # "enable_subblock_padding": False,
                    },
                }
            }
        },
        "1": {
            "conv1": {
                "config": {
                    "dtype": ttnn.bfloat8_b,
                    "weights_dtype": ttnn.bfloat8_b,
                    "activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                    "act_block_h_override": 0,
                    "deallocate_activation": False,
                    "reallocate_halo_output": False,
                    #"in_place": False,
                    "reshard_if_not_optimal": False,
                    "override_sharding_config": True,
                    "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(4, 3)
                        ),
                    ]),
                    "transpose_shards": False,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": True,
                    # "enable_subblock_padding": False,
                }
            },
            "conv2": {
                "config": {
                    "dtype": ttnn.bfloat8_b,
                    "weights_dtype": ttnn.bfloat8_b,
                    "activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                    "act_block_h_override": 32*12,
                    "deallocate_activation": True,
                    "reallocate_halo_output": False,
                    #"in_place": False,
                    "reshard_if_not_optimal": False,
                    "override_sharding_config": True,
                    "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(4, 3)
                        ),
                    ]),
                    "transpose_shards": False,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": True,
                    # "enable_subblock_padding": False,
                },
            },
            "conv3": {
                "config": {
                    "dtype": ttnn.bfloat8_b,
                    "weights_dtype": ttnn.bfloat8_b,
                    "activation": None,
                    "act_block_h_override": 0,
                    "deallocate_activation": True,
                    "reallocate_halo_output": False,
                    #"in_place": False,
                    "reshard_if_not_optimal": False,
                    "override_sharding_config": True,
                    "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(4, 3)
                        ),
                    ]),
                    "transpose_shards": False,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": True,
                    # "enable_subblock_padding": False,
                },
            }
        },
        "2": {
            "conv1": {
                "config": {
                    "dtype": ttnn.bfloat8_b,
                    "weights_dtype": ttnn.bfloat8_b,
                    "activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                    "act_block_h_override": 0,
                    "deallocate_activation": False,
                    "reallocate_halo_output": False,
                    #"in_place": False,
                    "reshard_if_not_optimal": False,
                    "override_sharding_config": True,
                    "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(4, 3)
                        ),
                    ]),
                    "transpose_shards": False,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": True,
                    # "enable_subblock_padding": False,
                }
            },
            "conv2": {
                "config": {
                    "dtype": ttnn.bfloat8_b,
                    "weights_dtype": ttnn.bfloat8_b,
                    "activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                    "act_block_h_override": 32*12,
                    "deallocate_activation": True,
                    "reallocate_halo_output": False,
                    #"in_place": False,
                    "reshard_if_not_optimal": False,
                    "override_sharding_config": True,
                    "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(4, 3)
                        ),
                    ]),
                    "transpose_shards": False,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": True,
                    # "enable_subblock_padding": False,
                },
            },
            "conv3": {
                "config": {
                    "dtype": ttnn.bfloat8_b,
                    "weights_dtype": ttnn.bfloat8_b,
                    "activation": None,
                    "act_block_h_override": 0,
                    "deallocate_activation": True,
                    "reallocate_halo_output": False,
                    #"in_place": False,
                    "reshard_if_not_optimal": False,
                    "override_sharding_config": True,
                    "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(4, 3)
                        ),
                    ]),
                    "transpose_shards": False,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": True,
                    # "enable_subblock_padding": False,
                },
            }
        },
    },
    "layer2": {
        "0": {
            "conv1": {
                "config": {
                    "dtype": ttnn.bfloat8_b,
                    "weights_dtype": ttnn.bfloat8_b,
                    "activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                    "act_block_h_override": 32*6,
                    "deallocate_activation": True,
                    "reallocate_halo_output": False,
                    #"in_place": False,
                    "reshard_if_not_optimal": False,
                    "override_sharding_config": True,
                    "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(4, 3)
                        ),
                    ]),
                    "transpose_shards": False,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": True,
                    # "enable_subblock_padding": False,
                },
            },
            "conv2": {
                "config": {
                    "dtype": ttnn.bfloat8_b,
                    "weights_dtype": ttnn.bfloat8_b,
                    "activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                    "act_block_h_override": 0,
                    "deallocate_activation": True,
                    "reallocate_halo_output": False,
                    #"in_place": False,
                    "reshard_if_not_optimal": False,
                    "override_sharding_config": True,
                    "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(4, 3)
                        ),
                    ]),
                    "transpose_shards": False,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": True,
                    # "enable_subblock_padding": False,
                },
            },
            "conv3": {
                "config": {
                    "dtype": ttnn.bfloat8_b,
                    "weights_dtype": ttnn.bfloat8_b,
                    "activation": None,
                    "act_block_h_override": 0,
                    "deallocate_activation": True,
                    "reallocate_halo_output": False,
                    #"in_place": False,
                    "reshard_if_not_optimal": False,
                    "override_sharding_config": True,
                    "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(4, 3)
                        ),
                    ]),
                    "transpose_shards": False,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": True,
                    # "enable_subblock_padding": False,
                },
            },
            "downsample": {
                "0": {
                    "init_config": {
                        "output_config": [
                            {
                                "reallocate": True,
                            }
                        ]
                    },
                    "config": {
                        "dtype": ttnn.bfloat8_b,
                        "weights_dtype": ttnn.bfloat8_b,
                        "activation": None,
                        "act_block_h_override": 32*3,
                        "deallocate_activation": False,
                        "reallocate_halo_output": False,
                        #"in_place": False,
                        "reshard_if_not_optimal": False,
                        "override_sharding_config": True,
                        "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                        "core_grid": ttnn.CoreRangeSet([
                            ttnn.CoreRange(
                                ttnn.CoreCoord(0, 0),
                                ttnn.CoreCoord(4, 3)
                            ),
                        ]),
                        "transpose_shards": False,
                        "output_layout": ttnn.Layout.TILE,
                        "enable_act_double_buffer": True,
                        "force_split_reader": True,
                        # "enable_subblock_padding": False,
                    },
                }
            }
        },
        "1": {
            "conv1": {
                "config": {
                    "dtype": ttnn.bfloat8_b,
                    "weights_dtype": ttnn.bfloat8_b,
                    "activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                    "act_block_h_override": 0,
                    "deallocate_activation": False,
                    "reallocate_halo_output": False,
                    #"in_place": False,
                    "reshard_if_not_optimal": False,
                    "override_sharding_config": True,
                    "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(4, 3)
                        ),
                    ]),
                    "transpose_shards": False,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": True,
                    # "enable_subblock_padding": False,
                },
            },
            "conv2": {
                "config": {
                    "dtype": ttnn.bfloat8_b,
                    "weights_dtype": ttnn.bfloat8_b,
                    "activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                    "act_block_h_override": 0,
                    "deallocate_activation": True,
                    "reallocate_halo_output": False,
                    #"in_place": False,
                    "reshard_if_not_optimal": False,
                    "override_sharding_config": True,
                    "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(4, 3)
                        ),
                    ]),
                    "transpose_shards": False,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": True,
                    # "enable_subblock_padding": False,
                },
            },
            "conv3": {
                "config": {
                    "dtype": ttnn.bfloat8_b,
                    "weights_dtype": ttnn.bfloat8_b,
                    "activation": None,
                    "act_block_h_override": 0,
                    "deallocate_activation": True,
                    "reallocate_halo_output": False,
                    #"in_place": False,
                    "reshard_if_not_optimal": False,
                    "override_sharding_config": True,
                    "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(4, 3)
                        ),
                    ]),
                    "transpose_shards": False,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": True,
                    # "enable_subblock_padding": False,
                },
            }
        },
        "2": {
            "conv1": {
                "config": {
                    "dtype": ttnn.bfloat8_b,
                    "weights_dtype": ttnn.bfloat8_b,
                    "activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                    "act_block_h_override": 0,
                    "deallocate_activation": False,
                    "reallocate_halo_output": False,
                    #"in_place": False,
                    "reshard_if_not_optimal": False,
                    "override_sharding_config": True,
                    "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(4, 3)
                        ),
                    ]),
                    "transpose_shards": False,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": True,
                    # "enable_subblock_padding": False,
                },
            },
            "conv2": {
                "config": {
                    "dtype": ttnn.bfloat8_b,
                    "weights_dtype": ttnn.bfloat8_b,
                    "activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                    "act_block_h_override": 0,
                    "deallocate_activation": True,
                    "reallocate_halo_output": False,
                    #"in_place": False,
                    "reshard_if_not_optimal": False,
                    "override_sharding_config": True,
                    "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(4, 3)
                        ),
                    ]),
                    "transpose_shards": False,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": True,
                    # "enable_subblock_padding": False,
                },
            },
            "conv3": {
                "config": {
                    "dtype": ttnn.bfloat8_b,
                    "weights_dtype": ttnn.bfloat8_b,
                    "activation": None,
                    "act_block_h_override": 0,
                    "deallocate_activation": True,
                    "reallocate_halo_output": False,
                    #"in_place": False,
                    "reshard_if_not_optimal": False,
                    "override_sharding_config": True,
                    "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(4, 3)
                        ),
                    ]),
                    "transpose_shards": False,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": True,
                    # "enable_subblock_padding": False,
                },
            }
        },
        "3": {
            "conv1": {
                "config": {
                    "dtype": ttnn.bfloat8_b,
                    "weights_dtype": ttnn.bfloat8_b,
                    "activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                    "act_block_h_override": 0,
                    "deallocate_activation": False,
                    "reallocate_halo_output": False,
                    #"in_place": False,
                    "reshard_if_not_optimal": False,
                    "override_sharding_config": True,
                    "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(4, 3)
                        ),
                    ]),
                    "transpose_shards": False,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": True,
                    # "enable_subblock_padding": False,
                },
            },
            "conv2": {
                "config": {
                    "dtype": ttnn.bfloat8_b,
                    "weights_dtype": ttnn.bfloat8_b,
                    "activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                    "act_block_h_override": 0,
                    "deallocate_activation": True,
                    "reallocate_halo_output": False,
                    #"in_place": False,
                    "reshard_if_not_optimal": False,
                    "override_sharding_config": True,
                    "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(4, 3)
                        ),
                    ]),
                    "transpose_shards": False,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": True,
                    # "enable_subblock_padding": False,
                },
            },
            "conv3": {
                "config": {
                    "dtype": ttnn.bfloat8_b,
                    "weights_dtype": ttnn.bfloat8_b,
                    "activation": None,
                    "act_block_h_override": 0,
                    "deallocate_activation": True,
                    "reallocate_halo_output": False,
                    #"in_place": False,
                    "reshard_if_not_optimal": False,
                    "override_sharding_config": True,
                    "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(4, 3)
                        ),
                    ]),
                    "transpose_shards": False,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": True,
                    # "enable_subblock_padding": False,
                },
            }
        },
    },
    "layer3": {
        "0": {
            "conv1": {
                "init_config": {
                    "output_config": [
                        {
                            "reallocate": True,
                        }
                    ]
                },
                "config": {
                    "dtype": ttnn.bfloat8_b,
                    "weights_dtype": ttnn.bfloat8_b,
                    "activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                    "act_block_h_override": 0,
                    "deallocate_activation": True,
                    "reallocate_halo_output": False,
                    #"in_place": False,
                    "reshard_if_not_optimal": False,
                    "override_sharding_config": True,
                    "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(4, 3)
                        ),
                    ]),
                    "transpose_shards": False,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": True,
                    # "enable_subblock_padding": False,
                },
            },
            "conv2": {
                "config": {
                    "dtype": ttnn.bfloat8_b,
                    "weights_dtype": ttnn.bfloat8_b,
                    "activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                    "act_block_h_override": 0,
                    "deallocate_activation": True,
                    "reallocate_halo_output": False,
                    #"in_place": False,
                    "reshard_if_not_optimal": False,
                    "override_sharding_config": True,
                    "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(4, 3)
                        ),
                    ]),
                    "transpose_shards": True,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": False,
                    # "enable_subblock_padding": False,
                },
            },
            "conv3": {
                "config": {
                    "dtype": ttnn.bfloat8_b,
                    "weights_dtype": ttnn.bfloat8_b,
                    "activation": None,
                    "act_block_h_override": 0,
                    "deallocate_activation": True,
                    "reallocate_halo_output": False,
                    #"in_place": False,
                    "reshard_if_not_optimal": False,
                    "override_sharding_config": True,
                    "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(4, 3)
                        ),
                    ]),
                    "transpose_shards": True,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": False,
                    # "enable_subblock_padding": False,
                },
            },
            "downsample": {
                "0": {
                    "init_config": {
                        "output_config": [
                            {
                                "reallocate": True,
                            }
                        ]
                    },
                    "config": {
                        "dtype": ttnn.bfloat8_b,
                        "weights_dtype": ttnn.bfloat8_b,
                        "activation": None,
                        "act_block_h_override": 0,
                        "deallocate_activation": False,
                        "reallocate_halo_output": False,
                        #"in_place": False,
                        "reshard_if_not_optimal": False,
                        "override_sharding_config": True,
                        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                        "core_grid": ttnn.CoreRangeSet([
                            ttnn.CoreRange(
                                ttnn.CoreCoord(0, 0),
                                ttnn.CoreCoord(4, 3)
                            ),
                        ]),
                        "transpose_shards": True,
                        "output_layout": ttnn.Layout.TILE,
                        "enable_act_double_buffer": True,
                        "force_split_reader": False,
                        # "enable_subblock_padding": False,
                    },
                }
            }
        },
        "1": {
            "conv1": {
                "config": {
                    "dtype": ttnn.bfloat8_b,
                    "weights_dtype": ttnn.bfloat8_b,
                    "activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                    "act_block_h_override": 0,
                    "deallocate_activation": False,
                    "reallocate_halo_output": False,
                    #"in_place": False,
                    "reshard_if_not_optimal": False,
                    "override_sharding_config": True,
                    "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(4, 3)
                        ),
                    ]),
                    "transpose_shards": False,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": True,
                    # "enable_subblock_padding": False,
                },
            },
            "conv2": {
                "config": {
                    "dtype": ttnn.bfloat8_b,
                    "weights_dtype": ttnn.bfloat8_b,
                    "activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                    "act_block_h_override": 0,
                    "deallocate_activation": True,
                    "reallocate_halo_output": False,
                    #"in_place": False,
                    "reshard_if_not_optimal": False,
                    "override_sharding_config": True,
                    "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(4, 3)
                        ),
                    ]),
                    "transpose_shards": False,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": True,
                    # "enable_subblock_padding": False,
                },
            },
            "conv3": {
                "config": {
                    "dtype": ttnn.bfloat8_b,
                    "weights_dtype": ttnn.bfloat8_b,
                    "activation": None,
                    "act_block_h_override": 0,
                    "deallocate_activation": True,
                    "reallocate_halo_output": False,
                    #"in_place": False,
                    "reshard_if_not_optimal": False,
                    "override_sharding_config": True,
                    "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(4, 3)
                        ),
                    ]),
                    "transpose_shards": False,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": True,
                    # "enable_subblock_padding": False,
                },
            }
        },
        "2": {
            "conv1": {
                "config": {
                    "dtype": ttnn.bfloat8_b,
                    "weights_dtype": ttnn.bfloat8_b,
                    "activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                    "act_block_h_override": 0,
                    "deallocate_activation": False,
                    "reallocate_halo_output": False,
                    #"in_place": False,
                    "reshard_if_not_optimal": False,
                    "override_sharding_config": True,
                    "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(4, 3)
                        ),
                    ]),
                    "transpose_shards": False,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": True,
                    # "enable_subblock_padding": False,
                },
            },
            "conv2": {
                "config": {
                    "dtype": ttnn.bfloat8_b,
                    "weights_dtype": ttnn.bfloat8_b,
                    "activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                    "act_block_h_override": 0,
                    "deallocate_activation": True,
                    "reallocate_halo_output": False,
                    #"in_place": False,
                    "reshard_if_not_optimal": False,
                    "override_sharding_config": True,
                    "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(4, 3)
                        ),
                    ]),
                    "transpose_shards": False,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": True,
                    # "enable_subblock_padding": False,
                },
            },
            "conv3": {
                "config": {
                    "dtype": ttnn.bfloat8_b,
                    "weights_dtype": ttnn.bfloat8_b,
                    "activation": None,
                    "act_block_h_override": 0,
                    "deallocate_activation": True,
                    "reallocate_halo_output": False,
                    #"in_place": False,
                    "reshard_if_not_optimal": False,
                    "override_sharding_config": True,
                    "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(4, 3)
                        ),
                    ]),
                    "transpose_shards": False,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": True,
                    # "enable_subblock_padding": False,
                },
            }
        },
        "3": {
            "conv1": {
                "config": {
                    "dtype": ttnn.bfloat8_b,
                    "weights_dtype": ttnn.bfloat8_b,
                    "activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                    "act_block_h_override": 0,
                    "deallocate_activation": False,
                    "reallocate_halo_output": False,
                    #"in_place": False,
                    "reshard_if_not_optimal": False,
                    "override_sharding_config": True,
                    "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(4, 3)
                        ),
                    ]),
                    "transpose_shards": False,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": True,
                    # "enable_subblock_padding": False,
                },
            },
            "conv2": {
                "config": {
                    "dtype": ttnn.bfloat8_b,
                    "weights_dtype": ttnn.bfloat8_b,
                    "activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                    "act_block_h_override": 0,
                    "deallocate_activation": True,
                    "reallocate_halo_output": False,
                    #"in_place": False,
                    "reshard_if_not_optimal": False,
                    "override_sharding_config": True,
                    "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(4, 3)
                        ),
                    ]),
                    "transpose_shards": False,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": True,
                    # "enable_subblock_padding": False,
                },
            },
            "conv3": {
                "config": {
                    "dtype": ttnn.bfloat8_b,
                    "weights_dtype": ttnn.bfloat8_b,
                    "activation": None,
                    "act_block_h_override": 0,
                    "deallocate_activation": True,
                    "reallocate_halo_output": False,
                    #"in_place": False,
                    "reshard_if_not_optimal": False,
                    "override_sharding_config": True,
                    "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(4, 3)
                        ),
                    ]),
                    "transpose_shards": False,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": True,
                    # "enable_subblock_padding": False,
                },
            }
        },
        "4": {
            "conv1": {
                "config": {
                    "dtype": ttnn.bfloat8_b,
                    "weights_dtype": ttnn.bfloat8_b,
                    "activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                    "act_block_h_override": 0,
                    "deallocate_activation": False,
                    "reallocate_halo_output": False,
                    #"in_place": False,
                    "reshard_if_not_optimal": False,
                    "override_sharding_config": True,
                    "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(4, 3)
                        ),
                    ]),
                    "transpose_shards": False,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": True,
                    # "enable_subblock_padding": False,
                },
            },
            "conv2": {
                "config": {
                    "dtype": ttnn.bfloat8_b,
                    "weights_dtype": ttnn.bfloat8_b,
                    "activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                    "act_block_h_override": 0,
                    "deallocate_activation": True,
                    "reallocate_halo_output": False,
                    #"in_place": False,
                    "reshard_if_not_optimal": False,
                    "override_sharding_config": True,
                    "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(4, 3)
                        ),
                    ]),
                    "transpose_shards": False,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": True,
                    # "enable_subblock_padding": False,
                },
            },
            "conv3": {
                "config": {
                    "dtype": ttnn.bfloat8_b,
                    "weights_dtype": ttnn.bfloat8_b,
                    "activation": None,
                    "act_block_h_override": 0,
                    "deallocate_activation": True,
                    "reallocate_halo_output": False,
                    #"in_place": False,
                    "reshard_if_not_optimal": False,
                    "override_sharding_config": True,
                    "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(4, 3)
                        ),
                    ]),
                    "transpose_shards": False,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": True,
                    # "enable_subblock_padding": False,
                },
            }
        },
        "5": {
            "conv1": {
                "config": {
                    "dtype": ttnn.bfloat8_b,
                    "weights_dtype": ttnn.bfloat8_b,
                    "activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                    "act_block_h_override": 0,
                    "deallocate_activation": False,
                    "reallocate_halo_output": False,
                    #"in_place": False,
                    "reshard_if_not_optimal": False,
                    "override_sharding_config": True,
                    "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(4, 3)
                        ),
                    ]),
                    "transpose_shards": False,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": True,
                    # "enable_subblock_padding": False,
                },
            },
            "conv2": {
                "config": {
                    "dtype": ttnn.bfloat8_b,
                    "weights_dtype": ttnn.bfloat8_b,
                    "activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                    "act_block_h_override": 0,
                    "deallocate_activation": True,
                    "reallocate_halo_output": False,
                    #"in_place": False,
                    "reshard_if_not_optimal": False,
                    "override_sharding_config": True,
                    "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(4, 3)
                        ),
                    ]),
                    "transpose_shards": False,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": True,
                    # "enable_subblock_padding": False,
                },
            },
            "conv3": {
                "config": {
                    "dtype": ttnn.bfloat8_b,
                    "weights_dtype": ttnn.bfloat8_b,
                    "activation": None,
                    "act_block_h_override": 0,
                    "deallocate_activation": True,
                    "reallocate_halo_output": False,
                    #"in_place": False,
                    "reshard_if_not_optimal": False,
                    "override_sharding_config": True,
                    "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(4, 3)
                        ),
                    ]),
                    "transpose_shards": False,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": True,
                    # "enable_subblock_padding": False,
                },
            }
        },
    },
    "layer4": {
        "0": {
            "conv1": {
                "config": {
                    "dtype": ttnn.bfloat16,
                    "weights_dtype": ttnn.bfloat8_b,
                    "activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                    "act_block_h_override": 0,
                    "deallocate_activation": True,
                    "reallocate_halo_output": False,
                    #"in_place": False,
                    "reshard_if_not_optimal": False,
                    "override_sharding_config": True,
                    "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(4, 3)
                        ),
                    ]),
                    "transpose_shards": True,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": True,
                    # "enable_subblock_padding": False,
                },
            },
            "conv2": {
                "config": {
                    "dtype": ttnn.bfloat16,
                    "weights_dtype": ttnn.bfloat8_b,
                    "activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                    "act_block_h_override": 0,
                    "deallocate_activation": True,
                    "reallocate_halo_output": False,
                    #"in_place": False,
                    "reshard_if_not_optimal": False,
                    "override_sharding_config": True,
                    "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(4, 3)
                        ),
                    ]),
                    "transpose_shards": True,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": False,
                    "force_split_reader": False,
                    # "enable_subblock_padding": False,
                },
            },
            "conv3": {
                "config": {
                    "dtype": ttnn.bfloat16,
                    "weights_dtype": ttnn.bfloat8_b,
                    "activation": None,
                    "act_block_h_override": 0,
                    "deallocate_activation": True,
                    "reallocate_halo_output": False,
                    #"in_place": False,
                    "reshard_if_not_optimal": False,
                    "override_sharding_config": True,
                    "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(4, 3)
                        ),
                    ]),
                    "transpose_shards": True,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": False,
                    # "enable_subblock_padding": False,
                },
            },
            "downsample": {
                "0": {
                    "config": {
                        "dtype": ttnn.bfloat16,
                        "weights_dtype": ttnn.bfloat8_b,
                        "activation": None,
                        "act_block_h_override": 0,
                        "deallocate_activation": False,
                        "reallocate_halo_output": False,
                        #"in_place": False,
                        "reshard_if_not_optimal": False,
                        "override_sharding_config": True,
                        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                        "core_grid": ttnn.CoreRangeSet([
                            ttnn.CoreRange(
                                ttnn.CoreCoord(0, 0),
                                ttnn.CoreCoord(4, 3)
                            ),
                        ]),
                        "transpose_shards": True,
                        "output_layout": ttnn.Layout.TILE,
                        "enable_act_double_buffer": True,
                        "force_split_reader": False,
                        # "enable_subblock_padding": False,
                    },
                }
            }
        },
        "1": {
            "conv1": {
                "config": {
                    "dtype": ttnn.bfloat16,
                    "weights_dtype": ttnn.bfloat8_b,
                    "activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                    "act_block_h_override": 0,
                    "deallocate_activation": False,
                    "reallocate_halo_output": False,
                    #"in_place": False,
                    "reshard_if_not_optimal": False,
                    "override_sharding_config": True,
                    "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(4, 3)
                        ),
                    ]),
                    "transpose_shards": True,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": False,
                    # "enable_subblock_padding": False,
                },
            },
            "conv2": {
                "config": {
                    "dtype": ttnn.bfloat16,
                    "weights_dtype": ttnn.bfloat8_b,
                    "activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                    "act_block_h_override": 0,
                    "deallocate_activation": True,
                    "reallocate_halo_output": False,
                    #"in_place": False,
                    "reshard_if_not_optimal": False,
                    "override_sharding_config": True,
                    "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(4, 3)
                        ),
                    ]),
                    "transpose_shards": True,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": False,
                    # "enable_subblock_padding": False,
                },
            },
            "conv3": {
                "config": {
                    "dtype": ttnn.bfloat16,
                    "weights_dtype": ttnn.bfloat8_b,
                    "activation": None,
                    "act_block_h_override": 0,
                    "deallocate_activation": True,
                    "reallocate_halo_output": False,
                    #"in_place": False,
                    "reshard_if_not_optimal": False,
                    "override_sharding_config": True,
                    "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(4, 3)
                        ),
                    ]),
                    "transpose_shards": True,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": False,
                    # "enable_subblock_padding": False,
                },
            },
        },
        "2": {
            "conv1": {
                "config": {
                    "dtype": ttnn.bfloat16,
                    "weights_dtype": ttnn.bfloat8_b,
                    "activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                    "act_block_h_override": 0,
                    "deallocate_activation": False,
                    "reallocate_halo_output": False,
                    #"in_place": False,
                    "reshard_if_not_optimal": False,
                    "override_sharding_config": True,
                    "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(4, 3)
                        ),
                    ]),
                    "transpose_shards": True,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": False,
                    # "enable_subblock_padding": False,
                },
            },
            "conv2": {
                "config": {
                    "dtype": ttnn.bfloat16,
                    "weights_dtype": ttnn.bfloat8_b,
                    "activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                    "act_block_h_override": 0,
                    "deallocate_activation": True,
                    "reallocate_halo_output": False,
                    #"in_place": False,
                    "reshard_if_not_optimal": False,
                    "override_sharding_config": True,
                    "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(4, 3)
                        ),
                    ]),
                    "transpose_shards": True,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": False,
                    # "enable_subblock_padding": False,
                },
            },
            "conv3": {
                "config": {
                    "dtype": ttnn.bfloat16,
                    "weights_dtype": ttnn.bfloat8_b,
                    "activation": None,
                    "act_block_h_override": 0,
                    "deallocate_activation": True,
                    "reallocate_halo_output": False,
                    #"in_place": False,
                    "reshard_if_not_optimal": False,
                    "override_sharding_config": True,
                    "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                    "core_grid": ttnn.CoreRangeSet([
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(4, 3)
                        ),
                    ]),
                    "transpose_shards": True,
                    "output_layout": ttnn.Layout.TILE,
                    "enable_act_double_buffer": True,
                    "force_split_reader": False,
                    # "enable_subblock_padding": False,
                },
            },
        },
    }
}