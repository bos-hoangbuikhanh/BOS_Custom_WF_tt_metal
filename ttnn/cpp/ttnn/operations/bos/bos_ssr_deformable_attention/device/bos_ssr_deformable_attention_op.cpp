#include <algorithm>
#include <cmath>
#include <numeric>
#include <optional>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/run_operation.hpp"
#include "ttnn/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

#include "ttnn/operations/bos/bos_ssr_deformable_attention/device/bos_ssr_deformable_attention_op.hpp"
#include "bos_ssr_deformable_attention_program_factory.hpp"

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::bos::bos_ssr_deformable_attention{
    void BosSSRDeformableAttention::validate_with_output_tensors(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<std::optional<Tensor>>& output_tensors
    ) const {
        const auto& value_tensor = input_tensors[0];
        const auto& flipped_spatial_shapes = input_tensors[1];
        const auto& sampling_locations = input_tensors[2];
        const auto& attention_weights = input_tensors[3];
        const auto& bilinear_weight_hash = optional_input_tensors[0];
        const auto& optional_output_tensor = output_tensors.at(0);

        // Validate tensor layout
        TT_FATAL(value_tensor.layout() == Layout::ROW_MAJOR, "Only ROW_MAJOR layout is supported for input value!");
        TT_FATAL(flipped_spatial_shapes.layout() == Layout::TILE, "Only TILE layout is supported for spatial_shapes!");
        TT_FATAL(attention_weights.layout() == Layout::TILE, "Only TILE layout is supported for attention_weights!");
        TT_FATAL(sampling_locations.layout() == Layout::TILE, "Only TILE layout is supported for sampling_locations!");

        // Validate dtype
        if (use_fp32){
            TT_FATAL(attention_weights.dtype() == DataType::FLOAT32, "Only FLOAT32 is supported for attention_weights when set use_fp32!");
            TT_FATAL(sampling_locations.dtype() == DataType::FLOAT32, "Only FLOAT32 is supported for sampling_locations when set use_fp32!");
        }
        else{
            TT_FATAL(attention_weights.dtype() == DataType::BFLOAT16, "Only BFLOAT16 is supported for attention_weights when not set use_fp32!");
            TT_FATAL(sampling_locations.dtype() == DataType::BFLOAT16, "Only BFLOAT16 is supported for sampling_locations when not set use_fp32!");
        }


        // Validate memory layout
        TT_FATAL(value_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED, "Only INTERLEAVED is supported for input value!");
        TT_FATAL(attention_weights.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED, "Only INTERLEAVED is supported for attention_weights!");
        TT_FATAL(sampling_locations.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED, "Only INTERLEAVED is supported for sampling_locations!");

        // Use bilinear weight hash validation
        if (use_bilinear_weight_hash) {
            auto bilinear_weight_hash_ = bilinear_weight_hash.value();
            TT_FATAL(bilinear_weight_hash_.layout() == Layout::ROW_MAJOR, "Only ROW_MAJOR layout is supported for bilinear_weight_hash!");
            if (use_fp32) {
                TT_FATAL(bilinear_weight_hash_.dtype() == DataType::FLOAT32, "Only FLOAT32 is supported for bilinear_weight_hash, if use_fp32 = True!");
            } else {
                TT_FATAL(bilinear_weight_hash_.dtype() == DataType::BFLOAT16, "Only BFLOAT16 is supported for bilinear_weight_hash, if use_fp32 = False!");
            }
        }

        // Validate output tensor shape
        if (optional_output_tensor.has_value()) {
            TT_FATAL(
                optional_output_tensor.value().memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
                "Only INTERLEAVED memory layout is supported for outputs in deformable attention!"
            );
        }
    }

    std::vector<TensorSpec> BosSSRDeformableAttention::compute_output_specs(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<Tensor>>& output_tensors
    ) const {
        if (output_tensors.at(0).has_value()) {
            return {output_tensors.at(0)->tensor_spec()};
        }

        // Input shapes and validate
        const auto& value_tensor = input_tensors[0];
        auto value_shape = value_tensor.logical_shape();

        // Allocate output tensor
        ttnn::Shape output_shape;
        uint32_t output_batch_size = is_QHB ? 1 : batch_size;
        if (is_padded_input)
            output_shape = ttnn::Shape({
                output_batch_size,
                num_queries,
                num_heads,
                value_shape[3],
            });
        else
            output_shape = ttnn::Shape({
                output_batch_size,
                num_queries,
                num_heads * value_shape[3],
            });

        return {TensorSpec(
            output_shape,
            TensorLayout(value_tensor.dtype(), PageConfig(value_tensor.layout()), output_mem_config)
        )};
    }

    std::vector<Tensor> BosSSRDeformableAttention::create_output_tensors(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<Tensor>>& output_tensors
    ) const {
        // already preallocate
        if (output_tensors.at(0).has_value()) {
            return {output_tensors.at(0).value()};
        }

        return {create_device_tensor(
            compute_output_specs(input_tensors, output_tensors)[0], input_tensors[0].device()
        )};
    }

    operation::ProgramWithCallbacks BosSSRDeformableAttention::create_program(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor>& output_tensors
    ) const {
        const auto& value = input_tensors.at(0);
        const auto& flipped_spatial_shapes = input_tensors.at(1);
        const auto& sampling_locations = input_tensors.at(2);
        const auto& attention_weights = input_tensors.at(3);
        const auto& bilinear_weight_hash = optional_input_tensors.at(0);
        const auto& output_tensor = output_tensors.at(0);
        return bos_ssr_deformable_attention_multi_core_interleaved(
            value,
            flipped_spatial_shapes,
            sampling_locations,
            attention_weights,
            output_tensor,
            bilinear_weight_hash,
            batch_size,
            num_keys,
            num_heads,
            num_queries,
            num_levels,
            num_points,
            original_embed_dims,
            compute_kernel_config,
            use_fp32,
            is_QHB,
            use_bilinear_weight_hash,
            is_padded_input,
            is_denormed_grid
        );
    }
}
