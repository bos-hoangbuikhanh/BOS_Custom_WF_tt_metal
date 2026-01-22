#include <algorithm>
#include <cmath>
#include <numeric>
#include <optional>

#include <tt-metalium/constants.hpp>
#include "bos_setitem_op.hpp"
#include "bos_setitem_program_factory.hpp"

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::bos_setitem {

void BosSetitem::validate(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    // Input tensors
    uint32_t num_input = 2;
    TT_FATAL(input_tensors.size() == num_input, "Setitem requires exactly {} input tensors: input, value", num_input);
    const Tensor& input = input_tensors[0];
    const Tensor& value = input_tensors[1];
    auto input_rank = input.logical_shape().rank();
    auto value_rank = value.logical_shape().rank();
    auto input_tensor_layout = input.layout();
    TT_FATAL(input_rank > 0, "Invalid input rank {}, rank must be positive", input_rank);
    TT_FATAL(value_rank > 0, "Invalid value rank {}, rank must be positive", value_rank);

    // 1) Storage validation
    // Validate input storage
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Operand input must be on device");
    TT_FATAL(input.buffer() != nullptr, "Operand input must have an allocated device buffer");

    // Validate value storage
    TT_FATAL(value.storage_type() == StorageType::DEVICE, "Operand value must be on device");
    TT_FATAL(value.buffer() != nullptr, "Operand value must have an allocated device buffer");

    // 2) Memory / layout / dtype validation
    // Validate input memory config
    TT_FATAL(
        input.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Tensor input memory layout is not supported: expected {}, got {}",
        TensorMemoryLayout::INTERLEAVED,
        input.memory_config().memory_layout());

    // Validate matching memory configuration between value and input
    TT_FATAL(
        value.padded_shape().rank() == input.padded_shape().rank(),
        "Rank mismatch between input and value tensors: input has rank {}, but value has rank {}",
        input.padded_shape().rank(),
        value.padded_shape().rank());

    TT_FATAL(
        value.dtype() == input.dtype(),
        "Data type mismatch: input tensor has dtype {}, but value tensor has dtype {}",
        input.dtype(),
        value.dtype());

    TT_FATAL(
        value.layout() == input.layout(),
        "Layout mismatch: input tensor has layout {}, but value tensor has layout {}",
        input.layout(),
        value.layout());

    TT_FATAL(
        value.memory_config().memory_layout() == input.memory_config().memory_layout(),
        "Memory layout mismatch: input tensor has memory layout {}, but value tensor has memory layout {}",
        input.memory_config().memory_layout(),
        value.memory_config().memory_layout());

    // TODO: 3) Validation of slices
    //  Should be normalized slices
    //  Not support for negative steps
    //  Compute sizes for index checks - Max index boundaries
    //  uint32_t inner_size = input_rank > 0 ? input.logical_shape()[-1] : 0;
    //  uint32_t outer_size = input_rank > 0 ? input.volume() / inner_size : 0;

    // 3) Optional index tensor
    if (optional_input_tensors.size() > 0) {
        // Tensor dims must match with number of index tensors
        TT_FATAL(
            this->index_dims.has_value(), "Index tensor dimensions must be specified when index tensors is provided!");
        std::vector<uint32_t> index_dims = this->index_dims.value();
        uint32_t num_index_dims = index_dims.size();

        // Validate each tensor meet requirements
        uint32_t num_valid_tensors = 0;
        for (uint32_t i = 0; i < num_index_dims; ++i) {
            // Index tensor must be provided
            const auto& opt_index_tensor = optional_input_tensors[i];

            // Validate storage and memory
            if (opt_index_tensor.has_value()) {
                const auto index_tensor = opt_index_tensor.value();
                TT_FATAL(
                    index_tensor.storage_type() == StorageType::DEVICE,
                    "Index tensor at dim {} must be on device",
                    index_dims[i]);
                TT_FATAL(
                    index_tensor.buffer() != nullptr,
                    "Index tensor at dim {} must have an allocated device buffer",
                    index_dims[i]);
                TT_FATAL(
                    index_tensor.layout() == Layout::ROW_MAJOR,
                    "Index tensor at dim {} layout is not supported: expected {}, got {}",
                    index_dims[i],
                    Layout::ROW_MAJOR,
                    index_tensor.layout());
                TT_FATAL(
                    index_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
                    "Index tensor at dim {} memory layout is not supported: expected {}, got {}",
                    index_dims[i],
                    TensorMemoryLayout::INTERLEAVED,
                    index_tensor.memory_config().memory_layout());
                TT_FATAL(
                    (index_tensor.dtype() == DataType::UINT32 || index_tensor.dtype() == DataType::INT32),
                    "Index tensor at dim {} dtype is not supported: expected {}, got {}",
                    index_dims[i],
                    DataType::UINT32,
                    DataType::INT32,
                    index_tensor.dtype());
                TT_FATAL(
                    index_tensor.logical_shape().rank() == 1,
                    "Index tensor at dim {} must be 1-dimensional, but has rank {}",
                    index_dims[i],
                    index_tensor.logical_shape().rank());
                ++num_valid_tensors;
            }
        }
        TT_FATAL(
            num_valid_tensors == num_index_dims,
            "Setitem requires {} valid index tensors, but got {}",
            num_index_dims,
            num_valid_tensors);
    }

    // 5) Validation of TILE lauout
    if (input_tensor_layout == ttnn::Layout::TILE) {
        // For TILE layout, we need to validate slice of the last 2 dimensions
        for (int dim = input_rank - 1; dim >= input_rank - 2; --dim) {
            if (dim < 0) {
                break;  // Prevent underflow for uint32_t
            }
            log_debug(tt::LogOp, "Validating Setitem operation at dim {}", dim);

            // No indexing tensors on the last 2 dimensions
            bool is_indexing_tensor_dim = false;
            if (this->index_dims.has_value()) {
                is_indexing_tensor_dim = std::find(index_dims->begin(), index_dims->end(), dim) != index_dims->end();
            }

            log_debug(tt::LogOp, "Is indexing tensor at dim {}: {}", dim, is_indexing_tensor_dim);
            TT_FATAL(
                !is_indexing_tensor_dim,
                "Setitem operation does not support indexing by tensors on the last 2 dimensions for TILE layout, but "
                "got index tensor at dim {}",
                dim);

            // Validate steps
            TT_FATAL(
                this->steps[dim] == 1,
                "Setitem operation does not support steps on the last 2 dimensions for TILE layout, but got step {} at "
                "dim {}",
                this->steps[dim],
                dim);

            // Validate tile size
            uint32_t size = input.logical_shape()[dim];
            uint32_t slice_size = (this->ends[dim] - this->begins[dim] + this->steps[dim] - 1) / this->steps[dim];
            bool is_slice_able = (slice_size == size) ||
                                 ((slice_size % ttnn::TILE_SIZE == 0) && (this->begins[dim] % ttnn::TILE_SIZE == 0));

            log_debug(tt::LogOp, "Size: {}, and slize size: {}, is slice-able: {}", size, slice_size, is_slice_able);
            TT_FATAL(
                is_slice_able,
                "Setitem operation does not support slice size {} on the last 2 dimensions for TILE layout, but got "
                "slice size {} at dim {}",
                size,
                slice_size,
                dim);
        }
    }
}

std::vector<TensorSpec> BosSetitem::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    return {input_tensors[0].tensor_spec()};
}

std::vector<Tensor> BosSetitem::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    return {input_tensors[0]};
}

tt::tt_metal::operation::ProgramWithCallbacks BosSetitem::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    const std::vector<Tensor>& output_tensors) const {
    const auto& input = input_tensors[0];
    const auto& value = input_tensors[1];

    switch (input.layout()) {
        case Layout::ROW_MAJOR:
            log_debug(tt::LogOp, "Running bos_setitem_multi_core_rm_interleaved");
            return bos_setitem_multi_core_rm_interleaved(
                input,
                value,
                this->begins,
                this->ends,
                this->steps,
                this->slice_shape,
                optional_input_tensors,
                this->index_dims);

        case ttnn::Layout::TILE:
            log_debug(tt::LogOp, "Running bos_setitem_multi_core_tile_interleaved");
            return bos_setitem_multi_core_tile_interleaved(
                input,
                value,
                this->begins,
                this->ends,
                this->steps,
                this->slice_shape,
                optional_input_tensors,
                this->index_dims);

        default: TT_THROW("Unsupported layout for Setitem operation: {}", input.layout());
    }
}

}  // namespace ttnn::operations::bos_setitem
