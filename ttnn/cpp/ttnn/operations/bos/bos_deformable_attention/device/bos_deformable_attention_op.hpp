#pragma once
#include <optional>

#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"

namespace ttnn::operations::bos::bos_deformable_attention {
    struct BosDeformableAttention{

        // Configuration attributes
        const bool use_fp32;
        const bool use_bilinear_weight_hash;
        const bool is_padded_input;
        const bool is_denormed_grid;
        const MemoryConfig output_mem_config;
        const DeviceComputeKernelConfig compute_kernel_config;
        const uint32_t batch_size;
        const uint32_t num_keys;
        const uint32_t num_heads;
        const uint32_t num_queries;
        const uint32_t num_levels;
        const uint32_t num_points;

        void validate_with_output_tensors(
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& output_tensors
        ) const;

        std::vector<TensorSpec> compute_output_specs(
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<Tensor>>& output_tensors
        ) const;

        std::vector<Tensor> create_output_tensors(
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<Tensor>>& output_tensors
        ) const;

        tt::tt_metal::operation::ProgramWithCallbacks create_program(
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            std::vector<Tensor>& output_tensors
        ) const;
    };
}
