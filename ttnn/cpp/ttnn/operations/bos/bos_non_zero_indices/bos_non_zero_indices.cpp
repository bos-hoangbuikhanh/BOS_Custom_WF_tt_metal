// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "bos_non_zero_indices.hpp"
#include "device/bos_non_zero_indices_op.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/common/queue_id.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::bos::bos_non_zero_indices {

std::vector<ttnn::Tensor> BosNonZeroIndicesOperation::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const std::optional<std::uint32_t>& max_length,
    const std::optional<MemoryConfig>& memory_config_arg) {
    auto memory_config = memory_config_arg.value_or(input_tensor.memory_config());
    return operation::run(BosNonZeroIndices{memory_config, max_length}, {input_tensor}, {}, {});
}

}  // namespace ttnn::operations::bos
