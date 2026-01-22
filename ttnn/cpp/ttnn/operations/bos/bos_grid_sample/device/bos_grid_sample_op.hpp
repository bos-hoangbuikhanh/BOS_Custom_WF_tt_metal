#pragma once

#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::bos::bos_grid_sample {
struct Sample {
    const tt::tt_metal::MemoryConfig output_mem_config;
    const bool align_corners;
    const std::string mode;
    const DeviceComputeKernelConfig compute_kernel_config;

    void validate_with_output_tensors(
        const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
    std::vector<TensorSpec> compute_output_specs(
        const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
    std::vector<Tensor> create_output_tensors(
        const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
};

tt::tt_metal::operation::ProgramWithCallbacks sample_multi_core_bilinear(
    const Tensor& input,
    const Tensor& grid,
    const bool align_corners,
    const Tensor& output,
    const DeviceComputeKernelConfig compute_kernel_config);
tt::tt_metal::operation::ProgramWithCallbacks sample_multi_core_nearest(
    const Tensor& input, const Tensor& grid, const bool align_corners, const Tensor& output);

}  // namespace ttnn::operations::bos::bos_grid_sample
