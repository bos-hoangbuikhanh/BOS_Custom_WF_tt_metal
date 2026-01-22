// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bos_non_zero_indices_pybind.hpp"
#include "bos_non_zero_indices.hpp"
#include "ttnn-pybind/decorators.hpp"

namespace ttnn::operations::bos::bos_non_zero_indices {
    void bind_bos_non_zero_indices_operation(py::module& module) {
        bind_registered_operation(
            module,
            ttnn::bos_nonzero,
            R"doc(

            Returns the number of elements (N) that are non-zero as well as a tensor of the same shape as input where the first N elements are the indices of non-zero elements.

            Args:
                input_tensor (ttnn.Tensor): Input Tensor should be 1D and in row major layout.

            Keyword Args:
                max_length (int, optional): Preallocated size for output indices. When `None`, matches input length. Defaults to `None`.
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
                queue_id (int, optional): command queue id. Defaults to `0`.

            Returns:
                List of ttnn.Tensor: the output tensors.

            Example:

                >>> tensor = ttnn.to_device(ttnn.from_torch(torch.zeros((1, 1, 1, 32), dtype=torch.bfloat16)), device)
                >>> output = ttnn.bos_nonzero(tensor)
            )doc",
            ttnn::pybind_overload_t{
                [](decltype(ttnn::bos_nonzero)& self,
                   const ttnn::Tensor& input_tensor,
                   const std::optional<std::uint32_t>& max_length,
                   const std::optional<ttnn::MemoryConfig>& memory_config,
                   QueueId queue_id) -> std::vector<ttnn::Tensor> {
                    return self(queue_id, input_tensor, max_length, memory_config);
                },
                py::arg("input_tensor").noconvert(),
                py::kw_only(),
                py::arg("max_length") = std::nullopt,
                py::arg("memory_config") = std::nullopt,
                py::arg("queue_id") = 0});
    }
}  // namespace ttnn::operations::bos
