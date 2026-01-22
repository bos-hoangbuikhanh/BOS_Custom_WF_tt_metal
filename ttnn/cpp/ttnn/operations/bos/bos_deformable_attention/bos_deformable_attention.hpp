#pragma once

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/command_queue.hpp>
#include "ttnn/operations/data_movement/bcast/bcast.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::bos::bos_deformable_attention {
    struct BosDeformableAttentionOperation {
        static Tensor invoke(
            const Tensor& input,
            const Tensor& spatial_shapes,
            const Tensor& sampling_locations,
            const Tensor& attention_weights,
            const bool use_fp32,
            const bool is_denormed_grid,
            const std::optional<Tensor> bilinear_weight_hash = std::nullopt, // [steps, steps, 4]
            const std::optional<const MemoryConfig>& memory_config = std::nullopt,
            const std::optional<Tensor>& optional_output_tensor = std::nullopt,
            const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt
        );
    };
}

namespace ttnn{
    constexpr auto bos_deformable_attention = ttnn::register_operation<"ttnn::bos_deformable_attention", operations::bos::bos_deformable_attention::BosDeformableAttentionOperation>();
}