// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::operations::bos_setitem {
namespace py = pybind11;

void bind_bos_setitem(py::module& module);
}  // namespace ttnn::operations::bos_setitem
