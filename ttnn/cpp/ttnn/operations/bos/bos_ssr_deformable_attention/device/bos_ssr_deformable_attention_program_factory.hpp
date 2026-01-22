#include "ttnn/run_operation.hpp"

namespace ttnn::operations::bos::bos_ssr_deformable_attention {

using namespace tt::constants;

    tt::tt_metal::operation::ProgramWithCallbacks bos_ssr_deformable_attention_multi_core_interleaved(
        const Tensor& value,
        const Tensor& flipped_spatial_shapes,
        const Tensor& sampling_locations,
        const Tensor& attention_weights,
        const Tensor& output,
        const std::optional<Tensor>& bilinear_weight_hash,
        const uint32_t batch_size,
        const uint32_t num_keys,
        const uint32_t num_heads,
        const uint32_t num_queries,
        const uint32_t num_levels,
        const uint32_t num_points,
        const uint32_t original_embed_dims,
        const DeviceComputeKernelConfig compute_kernel_config,
        const bool use_fp32,
        const bool is_QHB,
        const bool use_bilinear_weight_hash,
        const bool is_padded_input,
        const bool is_denormed_grid
    );
}
