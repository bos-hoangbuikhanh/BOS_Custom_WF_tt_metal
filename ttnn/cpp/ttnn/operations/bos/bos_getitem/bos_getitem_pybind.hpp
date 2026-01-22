// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace py = pybind11;

namespace ttnn::operations::bos::bos_getitem {
void bind_bos_getitem_operation(py::module& module);
}  // namespace ttnn::operations::bos::bos_getitem