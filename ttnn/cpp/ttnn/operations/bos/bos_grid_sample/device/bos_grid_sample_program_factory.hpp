#include "ttnn/run_operation.hpp"

namespace ttnn::operations::bos::bos_grid_sample {

using namespace tt::constants;

tt::tt_metal::operation::ProgramWithCallbacks sample_multi_core_nearest(
    const Tensor& input, const Tensor& grid, const bool align_corners, const Tensor& output);

tt::tt_metal::operation::ProgramWithCallbacks sample_multi_core_bilinear(
    const Tensor& input,
    const Tensor& grid,
    const bool align_corners,
    const Tensor& output,
    const DeviceComputeKernelConfig compute_kernel_config);
}  // namespace ttnn::operations::bos::bos_grid_sample
