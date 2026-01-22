// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "bos_getitem_pybind.hpp"

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/bos/bos_getitem/bos_getitem.hpp"

namespace ttnn::operations::bos::bos_getitem {
void bind_bos_getitem_operation(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::bos_getitem,
        "bos Getitem operation",
        ttnn::pybind_arguments_t{
            py::arg("input"),
            py::arg("index_tensors"),
            py::arg("index_dims"),
            py::kw_only(),
            py::arg("output") = std::nullopt,
            py::arg("memory_config") = std::nullopt});
}
}  // namespace ttnn::operations::bos::bos_getitem

