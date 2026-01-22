// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"

#include "bos_setitem_pybind.hpp"
#include "bos_setitem.hpp"

namespace ttnn::operations::bos_setitem {
namespace py = pybind11;

void bind_bos_setitem(py::module& module) {
    auto doc =
        R"doc(
            Setitem operation.
            )doc";

    // a vector with a fixed size default value
    using OperationType = decltype(ttnn::bos_setitem);
    ttnn::bind_registered_operation(
        module,
        ttnn::bos_setitem,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input,
               const ttnn::Tensor& value,
               const ttnn::SmallVector<uint32_t>& begins,
               const ttnn::SmallVector<uint32_t>& ends,
               const std::optional<ttnn::SmallVector<uint32_t>>& steps,
               const std::optional<std::vector<ttnn::Tensor>>& index_tensors,
               const std::optional<ttnn::SmallVector<uint32_t>>& index_dims,
               QueueId queue_id) {
                return self(queue_id, input, value, begins, ends, steps, index_tensors, index_dims);
            },
            py::arg("input"),
            py::arg("value"),
            py::arg("begins"),
            py::arg("ends"),
            py::arg("steps") = std::nullopt,
            py::arg("index_tensors") = std::nullopt,
            py::arg("index_dims") = std::nullopt,
            py::kw_only(),
            py::arg("queue_id") = 0,
        },
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input,
               const ttnn::Tensor& value,
               const ttnn::SmallVector<int>& begins,
               const ttnn::SmallVector<int>& ends,
               const std::optional<ttnn::SmallVector<int>>& steps,
               const std::optional<std::vector<ttnn::Tensor>>& index_tensors,
               const std::optional<ttnn::SmallVector<int>>& index_dims,
               QueueId queue_id) {
                return self(queue_id, input, value, begins, ends, steps, index_tensors, index_dims);
            },
            py::arg("input"),
            py::arg("value"),
            py::arg("begins"),
            py::arg("ends"),
            py::arg("steps") = std::nullopt,
            py::arg("index_tensors") = std::nullopt,
            py::arg("index_dims") = std::nullopt,
            py::kw_only(),
            py::arg("queue_id") = 0,
        });
}
}  // namespace ttnn::operations::bos_setitem
