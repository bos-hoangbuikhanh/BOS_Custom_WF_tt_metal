// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "bos_non_zero_indices_op.hpp"

using namespace tt::tt_metal;

namespace ttnn {

namespace operations::bos::bos_non_zero_indices {

void BosNonZeroIndices::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto input_tensor_a_shape = input_tensor_a.padded_shape();
    // TT_FATAL(
    //     input_tensor_a_shape[0] == 1 and input_tensor_a_shape[1] == 1 and input_tensor_a_shape[2] == 1,
    //     "The input shape must be 4D with the following form: 1, 1, 1, X.");
    TT_FATAL(input_tensor_a.layout() == Layout::ROW_MAJOR, "Currently only supporting row major layout");
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to Non-zero need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands to Non-zero need to be allocated in buffers on device!");
    TT_FATAL(
        input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Non-zero does not currently support sharding");
}

std::vector<ttnn::TensorSpec> BosNonZeroIndices::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    ttnn::Shape num_non_zero_shape({1});
    // Determine output indices size based on optional max_length or input length
    const auto& input_tensor = input_tensors.at(0);
    const std::uint32_t input_length = input_tensor.padded_shape()[-1];
    const std::uint32_t output_length = max_length.has_value() ? max_length.value() : input_length;
    ttnn::Shape output_indices_shape({output_length});
    TensorLayout layout(DataType::INT32, PageConfig(Layout::ROW_MAJOR), output_mem_config);
    return {TensorSpec(num_non_zero_shape, layout), TensorSpec(output_indices_shape, layout)};
}

operation::ProgramWithCallbacks BosNonZeroIndices::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& out_num_indices = output_tensors.at(0);
    const auto& out_indices = output_tensors.at(1);

    return bos_non_zero_indices_single_core(input_tensor, out_num_indices, out_indices);
}

}  // namespace operations::bos

}  // namespace ttnn
