// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/bos/bos_getitem/device/bos_getitem_device_operation.hpp"

namespace ttnn::operations::bos::bos_getitem {
struct BosGetItem {
    static Tensor invoke(
        const std::optional<const Tensor>& input,
        const std::vector<Tensor>& index_tensors,
        const ttnn::SmallVector<uint32_t>& index_dims,
        const std::optional<Tensor>& output,
        // const CoreRange core_range,
        const std::optional<MemoryConfig>& memory_config);
};
}  // namespace ttnn::operations::bos::bos_getitem

namespace ttnn {
constexpr auto bos_getitem =
    ttnn::register_operation<"ttnn::bos_getitem", ttnn::operations::bos::bos_getitem::BosGetItem>();
}
