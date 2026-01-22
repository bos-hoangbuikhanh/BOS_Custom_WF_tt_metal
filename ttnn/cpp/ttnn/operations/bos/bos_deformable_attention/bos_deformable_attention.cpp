#include "bos_deformable_attention.hpp"
#include "device/bos_deformable_attention_op.hpp"
#include <cmath>

#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/repeat_interleave/repeat_interleave.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"

namespace ttnn::operations::bos::bos_deformable_attention {
    Tensor BosDeformableAttentionOperation::invoke(
        const Tensor& input,
        const Tensor& spatial_shapes,
        const Tensor& sampling_locations,
        const Tensor& attention_weights,
        const bool use_fp32,
        const bool is_denormed_grid,
        const std::optional<Tensor> bilinear_weight_hash, // [steps, steps, 4]
        const std::optional<const MemoryConfig>& memory_config,
        const std::optional<Tensor>& optional_output_tensor,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config
    ) {
        auto input_shape = input.logical_shape();
        auto spatial_shape = spatial_shapes.logical_shape();
        auto sampling_locations_shape = sampling_locations.logical_shape();
        auto attention_weights_shape = attention_weights.logical_shape();

        // very first validation to ensure later pre-processing
        TT_FATAL(
            input_shape.rank() == 4,
            "Input should be in format (B, num_keys, num_heads, embed_dims). Got: {}",
            input_shape);
        TT_FATAL(
            sampling_locations_shape.rank() == 6,
            "Sampling locations should be in format of (B, num_queries, num_heads, num_levels, num_points, 2). "
            "Got {}",
            sampling_locations_shape);
        TT_FATAL(
            attention_weights_shape.rank() == 5,
            "Sampling locations should be in format of (B, num_queries, num_heads, num_levels, num_points). "
            "Got {}",
            attention_weights_shape);
        TT_FATAL(spatial_shape.rank() == 2,
            "spatial_shape should be in format of (num_level, 2). Got {}",
            spatial_shape);
        TT_FATAL(sampling_locations.layout() == Layout::TILE, "sampling_locations must be in TILE_LAYOUT");
        TT_FATAL(attention_weights.layout() == Layout::TILE, "attention_weights must be in TILE_LAYOUT");
        TT_FATAL(spatial_shapes.layout() == Layout::TILE, "spatial_shapes must be in TILE_LAYOUT");

        uint32_t steps_x = 0, steps_y = 0;
        bool use_bilinear_weight_hash = bilinear_weight_hash.has_value();
        if (use_bilinear_weight_hash) {
            auto bilinear_weight_hash_shape = bilinear_weight_hash.value().logical_shape();
            TT_FATAL(bilinear_weight_hash_shape.rank() == 3,
                "bilinear_weight_hash should be in format of (steps, steps, 4). Got {}",
                bilinear_weight_hash_shape);
            TT_FATAL(bilinear_weight_hash_shape[0] > 1 && bilinear_weight_hash_shape[1] > 1 && bilinear_weight_hash_shape[2] == 4,
                "bilinear_weight_hash should be in format of (steps, steps, 4). Got {}",
                bilinear_weight_hash_shape);
            TT_FATAL(bilinear_weight_hash_shape.rank() == 3,
                "bilinear_weight_hash should be in format of (steps, steps, 4). Got {}",
                bilinear_weight_hash_shape);
            TT_FATAL(bilinear_weight_hash_shape[0] > 1 && bilinear_weight_hash_shape[1] > 1 && bilinear_weight_hash_shape[2] == 4,
                "bilinear_weight_hash should be in format of (steps, steps, 4). Got {}",
                bilinear_weight_hash_shape);
            // Check step support
            steps_x = bilinear_weight_hash_shape[0];
            steps_y = bilinear_weight_hash_shape[1];
            TT_FATAL(steps_x == steps_y, "Currently support bilinear hash step_x = step_y, but got: step_x={}, step_y={}", steps_x, steps_y);
        }

        // =====================================================Quy=====================================================
        // "To maximize performance:
        // - Keep the input tensor unchanged.
        // - Reshape the sampling locations tensor from (B, num_queries, num_heads, num_levels, num_points, 2)
        //   to (B*num_queries*num_heads, num_levels*num_points, 2).
        // - Reshape the attention weights tensor from (B, num_queries, num_heads, num_levels, num_points)
        //   to (B*num_queries*num_heads, 1, num_levels*num_points).
        // - Expand the spatial tensor from (num_levels, 2) to (num_levels, num_points, 2).
        //   Then, reshape to (1, num_levels*num_points, 2) and broadcast for denormalization.
        // - Apply bilinear weight hash

        auto device = input.device();

        const uint32_t batch_size = input_shape[0];
        const uint32_t num_keys = input_shape[1];
        const uint32_t num_heads = input_shape[2];
        const uint32_t num_queries = sampling_locations_shape[1];
        const uint32_t num_levels = sampling_locations_shape[3];
        const uint32_t num_points = sampling_locations_shape[4];
        [[maybe_unused]] const uint32_t original_embed_dims = input_shape[3];

        ttnn::Shape logical_shape(
            {batch_size * num_queries * num_heads, num_levels * num_points, sampling_locations_shape[5]});
        ttnn::Shape padded_shape(
            {batch_size * num_queries * num_heads,
             round_up_to_mul32(num_levels * num_points),
             round_up_to_mul32(sampling_locations_shape[5])});
        Tensor sampling_locations_ = ttnn::reshape(sampling_locations, logical_shape, padded_shape);

        ttnn::SmallVector<uint32_t> begins = {0, 0};
        ttnn::SmallVector<uint32_t> ends = {num_levels, 1};
        ttnn::SmallVector<uint32_t> steps = {1, 1};
        Tensor in_height = ttnn::slice(spatial_shapes, begins, ends, steps);

        begins = {0, 1};
        ends = {num_levels, 2};
        steps = {1, 1};
        Tensor in_width = ttnn::slice(spatial_shapes, begins, ends, steps);

        Tensor flipped_spatial_shapes = ttnn::concat(std::vector<ttnn::Tensor>({in_width, in_height}), 1);
        in_height.deallocate();
        in_width.deallocate();

        //reshape to (spatial[0], 1, spatial[1]) but in tile, must hardcode
        logical_shape = ttnn::Shape({num_levels, 1, spatial_shape[1]});
        padded_shape = ttnn::Shape({num_levels, 32, round_up_to_mul32(spatial_shape[1])});
        flipped_spatial_shapes = ttnn::reshape(flipped_spatial_shapes, logical_shape, padded_shape);
        flipped_spatial_shapes = ttnn::repeat_interleave(flipped_spatial_shapes, num_points, 1);
        logical_shape = ttnn::Shape({1, num_levels*num_points, spatial_shape[1]});
        padded_shape = ttnn::Shape({1, round_up_to_mul32(num_levels*num_points), round_up_to_mul32(spatial_shape[1])});
        flipped_spatial_shapes = ttnn::reshape(flipped_spatial_shapes, logical_shape, padded_shape);

        // B*num_queries*num_heads, 1, num_levels*num_points
        logical_shape = ttnn::Shape({batch_size*num_queries*num_heads, 1, num_levels*num_points});
        padded_shape = ttnn::Shape({batch_size*num_queries*num_heads, 32, round_up_to_mul32(num_levels*num_points)});
        Tensor attention_weights_ = ttnn::reshape(attention_weights, logical_shape, padded_shape);

        // typecast to preserve accuracy
        if (use_fp32){
            sampling_locations_ = ttnn::typecast(sampling_locations_, DataType::FLOAT32);
            attention_weights_ = ttnn::typecast(attention_weights_, DataType::FLOAT32);
        }

        ttnn::DeviceComputeKernelConfig config = compute_kernel_config.value_or(
            ttnn::init_device_compute_kernel_config(
                device->arch(),
                std::nullopt,
                MathFidelity::HiFi2,
                /*default_approx_mode=*/false,
                /*default_fp32_acc=*/true
            )
        );
        ttnn::Tensor input_;
        const uint32_t TILE_WIDTH = 32;
        const uint32_t embed_dims = input_shape[-1];
        uint32_t num_padded_channels = embed_dims;
        num_padded_channels = embed_dims + (TILE_WIDTH - embed_dims%TILE_WIDTH);

        bool is_padded_input = (embed_dims % 32 != 0);

        if (is_padded_input){
            input_ = ttnn::pad(
                            input,
                            tt::tt_metal::Array4D(
                                    {input_shape[0],
                                    input_shape[1],
                                    input_shape[2],
                                    num_padded_channels}),
                            tt::tt_metal::Array4D({0, 0, 0, 0}),
                            0);
        } else {
            input_ = input;
        }
        auto output = tt::tt_metal::operation::run(
                        BosDeformableAttention{
                            use_fp32,
                            use_bilinear_weight_hash,
                            is_padded_input,
                            is_denormed_grid,
                            memory_config.value_or(input_.memory_config()),
                            config,
                            batch_size,
                            num_keys,
                            num_heads,
                            num_queries,
                            num_levels,
                            num_points
                        },
                        {input_, flipped_spatial_shapes, sampling_locations_, attention_weights_},
                        {bilinear_weight_hash},
                        {optional_output_tensor}
                    ).at(0);

        sampling_locations_.deallocate();
        flipped_spatial_shapes.deallocate();
        attention_weights_.deallocate();

        auto output_shape = output.logical_shape();
        if (is_padded_input){
            input_.deallocate();
            begins = {0, 0, 0, 0};
            ends = {output_shape[0], output_shape[1], output_shape[2], embed_dims};
            steps = {1, 1, 1, 1};
            output = ttnn::slice(output, begins, ends, steps);

            ttnn::Shape new_shape({output_shape[0], output_shape[1], output_shape[2]*embed_dims});
            output = ttnn::reshape(output, new_shape);
        }
        return output;
    }
    };  // namespace ttnn::operations::bos::bos_deformable_attention
