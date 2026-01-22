// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "bos_pybind.hpp"

#include "ttnn/operations/bos/bos_grid_sample/bos_grid_sample_pybind.hpp"
#include "ttnn/operations/bos/bos_deformable_attention/bos_deformable_attention_pybind.hpp"
#include "ttnn/operations/bos/bos_non_zero_indices/bos_non_zero_indices_pybind.hpp"
#include "ttnn/operations/bos/bos_sharded/bos_reshard/bos_reshard_pybind.hpp"
#include "ttnn/operations/bos/bos_setitem/bos_setitem_pybind.hpp"
#include "ttnn/operations/bos/bos_getitem/bos_getitem_pybind.hpp"
#include "ttnn/operations/bos/bos_ssr_deformable_attention/bos_ssr_deformable_attention_pybind.hpp"

namespace py = pybind11;

namespace ttnn::operations::bos {
void bind_bos_operations(py::module& module) {
    bos_grid_sample::bind_bos_grid_sample_operation(module);
    bos_deformable_attention::bind_bos_deformable_attention_operation(module);
    bos_non_zero_indices::bind_bos_non_zero_indices_operation(module);
    bos_reshard::bind_bos_reshard_operation(module);
    bos_setitem::bind_bos_setitem(module);
    bos_getitem::bind_bos_getitem_operation(module);
    bos_ssr_deformable_attention::bind_bos_ssr_deformable_attention_operation(module);

}
}  // namespace ttnn::operations::bos
