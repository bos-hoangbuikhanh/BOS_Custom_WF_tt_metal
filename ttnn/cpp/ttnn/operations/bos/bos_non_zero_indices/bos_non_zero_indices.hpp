// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <optional>

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::bos::bos_non_zero_indices {

struct BosNonZeroIndicesOperation {
    static std::vector<ttnn::Tensor> invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        const std::optional<std::uint32_t>& max_length,
        const std::optional<MemoryConfig>& memory_config);
};

}  // namespace operations::bos

constexpr auto bos_nonzero =
    ttnn::register_operation<"ttnn::bos_nonzero", ttnn::operations::bos::bos_non_zero_indices::BosNonZeroIndicesOperation>();

}  // namespace ttnn
