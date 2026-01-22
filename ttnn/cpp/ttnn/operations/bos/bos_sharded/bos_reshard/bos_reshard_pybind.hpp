// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::operations::bos::bos_reshard {

void bind_bos_reshard_operation(pybind11::module& module);

}  // namespace ttnn::operations::bos::bos_reshard
