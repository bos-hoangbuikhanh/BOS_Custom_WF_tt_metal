// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/run_operation.hpp"
#include "device/bos_reshard_op.hpp"
#include "bos_reshard.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::bos::bos_reshard {

ttnn::Tensor BosReshardOperation::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const MemoryConfig& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return operation::run(
               BosReshardDeviceOperation{.output_mem_config = memory_config}, {input_tensor}, {}, {optional_output_tensor})
        .at(0);
}

}  // namespace ttnn::operations::bos::bos_reshard
