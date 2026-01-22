// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "bos_reshard.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/core_coord.hpp>

namespace ttnn::operations::bos::bos_reshard {

void bind_bos_reshard_operation(pybind11::module& module) {
    bind_registered_operation(
        module,
        ttnn::bos_reshard,
        R"doc(reshard(input_tensor: ttnn.Tensor,  output_memory_config: MemoryConfig *, output_tensor: Optional[ttnn.Tensor] = None) -> ttnn.Tensor

        Converts a tensor from one sharded layout to another sharded layout

        Args:
            * :attr:`input_tensor` (ttnn.Tensor): input tensor
            * :attr:`output_memory_config` (MemoryConfig): Memory config with shard spec of output tensor

        Keyword Args:
            * :attr:`queue_id`: command queue id

        Example:
            >>> sharded_memory_config_dict = dict(
                core_grid=ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1)
                        ),
                    }
                ),
                strategy=ttnn.ShardStrategy.BLOCK,
            ),
            >>> shard_memory_config = ttnn.create_sharded_memory_config(input_shape, **input_sharded_memory_config_args)
            >>> sharded_tensor = ttnn.bos_reshard(tensor, shard_memory_config)

        )doc",
        ttnn::pybind_overload_t{
            [](decltype(ttnn::bos_reshard)& self,
               const ttnn::Tensor& input_tensor,
               const MemoryConfig& output_memory_config,
               const std::optional<Tensor>& output_tensor,
               QueueId queue_id) -> ttnn::Tensor {
                return self(queue_id, input_tensor, output_memory_config, output_tensor);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("output_memory_config"),
            py::arg("output_tensor").noconvert() = std::nullopt,
            py::kw_only(),
            py::arg("queue_id") = 0,

        });
}

}  // namespace ttnn::operations::bos::bos_reshard
