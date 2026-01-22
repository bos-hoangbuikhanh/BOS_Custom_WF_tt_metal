#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn::operations::bos::bos_grid_sample {
struct SampleOperation {
    static Tensor invoke(
        const Tensor& input,
        const Tensor& grid,
        const bool align_corners = false,
        const std::string mode = "bilinear",
        const std::optional<const MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor> output_tensor = std::nullopt,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);
};
}  // namespace ttnn::operations::bos::bos_grid_sample

namespace ttnn {
constexpr auto bos_grid_sample =
    ttnn::register_operation<"ttnn::bos_grid_sample", ttnn::operations::bos::bos_grid_sample::SampleOperation>();
}
