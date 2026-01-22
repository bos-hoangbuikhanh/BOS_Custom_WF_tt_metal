// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::operations::bos::bos_non_zero_indices {

void bind_bos_non_zero_indices_operation(pybind11::module& module);

}  // namespace ttnn::operations::bos
