#include "ttnn/operations/bos/bos_grid_sample/device/bos_grid_sample_op.hpp"
#include "bos_grid_sample_program_factory.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <optional>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/run_operation.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/tensor/tensor.hpp"

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::bos::bos_grid_sample {
// inline uint32_t get_max_l1_space(const Tensor& input_tensor_a) {
//     auto device = input_tensor_a.device();
//     auto lowest_address = device->lowest_occupied_compute_l1_address();
//     uint32_t max_l1_space = lowest_address.has_value() ? lowest_address.value() : device->l1_size_per_core();
//     max_l1_space = max_l1_space - device->get_base_allocator_addr(HalMemType::L1);
//     return max_l1_space;
// }

// inline void get_max_num_sticks(uint32_t& total_size, uint32_t& num_cores, uint32_t& num_sticks_per_core){
//     while (total_size % num_cores != 0) num_cores--;
//     num_sticks_per_core = total_size/num_cores;
// }

void Sample::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    TT_FATAL(input_tensors.size() == 2, "Must have 2 input tensors");
    TT_FATAL(output_tensors.size() == 1, "Must have 1 output tensors");

    const auto& input_tensor = input_tensors[0];
    const auto& grid_tensor = input_tensors[1];
    [[maybe_unused]] const auto& mask_tensor = input_tensors[2];
    const auto& optional_output_tensor = output_tensors.at(0);

    TT_FATAL(input_tensor.dtype() == DataType::BFLOAT16, "Only BFLOAT16 is supported for inputs!");
    TT_FATAL(input_tensor.layout() == Layout::ROW_MAJOR, "Only ROW_MAJOR layout is supported for inputs!");
    TT_FATAL(
        mode == "bilinear" || mode == "nearest",
        "Only bilinear and nearest modes are supported for grid sample operation!");
    if (mode == "nearest") {
        TT_FATAL(grid_tensor.dtype() == DataType::FLOAT32, "Grid tensor for nearest mode should be Float32!");
    }
    TT_FATAL(grid_tensor.layout() == Layout::ROW_MAJOR, "Only ROW_MAJOR layout is supported for grid!");
    TT_FATAL(
        mode == "bilinear" || mode == "nearest",
        "Only bilinear and nearest modes are supported for grid sample operation!");
    if (optional_output_tensor.has_value()) {
        TT_FATAL(
            optional_output_tensor.value().dtype() == DataType::BFLOAT16, "Only BFLOAT16 is supported for outputs!");
        TT_FATAL(
            output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "Only INTERLEAVED memory layout is supported for outputs!");
    }
}

std::vector<TensorSpec> Sample::compute_output_specs(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if (output_tensors.at(0).has_value()) {
        return {output_tensors.at(0)->tensor_spec()};
    }

    const auto& input_tensor = input_tensors[0];
    const auto& grid_tensor = input_tensors[1];

    auto grid_shape = grid_tensor.padded_shape();  // 1, 1, BHoWo, 2
    TT_FATAL(grid_shape[3] == 2, "The last dimension should contain only pixel coordinations (x, y)");

    auto input_shape = input_tensor.padded_shape();  // 1, 1, BHiWi, Ci
    TT_FATAL(input_shape[0] == grid_shape[0], "batch size of grid and input must be equal");

    tt::tt_metal::Shape output_shape({grid_shape[0], grid_shape[1], grid_shape[2], input_shape[-1]});
    return {TensorSpec(
        output_shape, TensorLayout(input_tensor.dtype(), PageConfig(input_tensor.layout()), output_mem_config))};
}

std::vector<Tensor> Sample::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if (output_tensors.at(0).has_value()) {
        return {output_tensors.at(0).value()};
    }

    return {create_device_tensor(compute_output_specs(input_tensors, output_tensors)[0], input_tensors[0].device())};
}

operation::ProgramWithCallbacks Sample::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input = input_tensors.at(0);
    const auto& grid = input_tensors.at(1);
    const auto& output_tensor = output_tensors.at(0);

    if (mode == "bilinear") {
        return sample_multi_core_bilinear(input, grid, align_corners, output_tensor, compute_kernel_config);
    } else {
        return sample_multi_core_nearest(input, grid, align_corners, output_tensor);
    }
}
}  // namespace ttnn::operations::bos::bos_grid_sample
