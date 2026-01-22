#include <cmath>

#include "bos_grid_sample.hpp"
#include "device/bos_grid_sample_op.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"

namespace ttnn::operations::bos::bos_grid_sample {
Tensor SampleOperation::invoke(
    const Tensor& input,
    const Tensor& grid,
    const bool align_corners,
    const std::string mode,
    const std::optional<const MemoryConfig>& memory_config,
    const std::optional<Tensor> output_tensor,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    ttnn::DeviceComputeKernelConfig config = compute_kernel_config.value_or(
        ttnn::init_device_compute_kernel_config(input.device()->arch(), std::nullopt, MathFidelity::HiFi2));

    const uint32_t TILE_WIDTH = 32;
    auto input_shape = input.padded_shape();
    uint32_t num_channels = input_shape[-1];
    uint32_t num_padded_channels = num_channels;
    ttnn::Tensor input_;
    bool deallocate_input = false;
    if (mode == "bilinear" && num_channels % TILE_WIDTH != 0) {
        num_padded_channels = num_channels + (TILE_WIDTH - num_channels % TILE_WIDTH);
        input_ = ttnn::pad(
            input,
            tt::tt_metal::Array4D({input_shape[0], input_shape[1], input_shape[2], num_padded_channels}),
            tt::tt_metal::Array4D({0, 0, 0, 0}),
            0);
        deallocate_input = true;
    } else {
        input_ = input;  // reference copy (same address)
    }
    auto output = tt::tt_metal::operation::run(
                      Sample{memory_config.value_or(input.memory_config()), align_corners, mode, config},
                      {input_, grid},
                      {},
                      {output_tensor})
                      .at(0);

    if (deallocate_input) {
        ttnn::deallocate(input_);
    }

    auto output_shape = output.padded_shape();
    // std::cout << output_shape << std::endl;
    ttnn::SmallVector<uint32_t> begins = {0, 0, 0, 0};
    ttnn::SmallVector<uint32_t> ends = {output_shape[0], output_shape[1], output_shape[2], num_channels};
    ttnn::SmallVector<uint32_t> step = {1, 1, 1, 1};
    output = ttnn::slice(output, begins, ends, step);
    return output;
}
};  // namespace ttnn::operations::bos::bos_grid_sample
