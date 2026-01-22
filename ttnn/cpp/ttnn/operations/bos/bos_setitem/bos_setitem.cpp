#include <cmath>

#include "ttnn/run_operation.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "cpp/ttnn/operations/data_movement/unsqueeze/unsqueeze.hpp"
#include "bos_setitem.hpp"
#include "device/bos_setitem_op.hpp"

namespace ttnn::operations::bos_setitem {

uint32_t wrap_index(int index, int size) { return index < 0 ? size + index : index; }

/**
 * @brief Executes a setitem operation with slicing and optional index tensors.
 *
 * This operation updates the `input` tensor using values from `value` tensor
 * based on provided slice ranges and/or index tensors. The operation is validated
 * against tensor ranks, shapes, and indexing constraints to ensure correctness.
 *
 * @tparam T Type of the indexing values (signed/unsigned integers).
 * @param queue_id ID of the queue where the operation will be dispatched.
 * @param input Input tensor to be updated. Must be device-resident.
 * @param value Tensor containing values to write into the input. Must be device-resident.
 * @param begins Slice start indices (per dimension).
 * @param ends Slice end indices (per dimension).
 * @param steps Slice steps (per dimension).
 * @param index_tensors Optional vector of index tensors for advanced indexing (1D tensors).
 * @param index_dims Optional span of dimensions where index tensors apply.
 *
 * @return Updated tensor (same buffer as input if updated in-place).
 */
template <typename T>
ttnn::Tensor BosSetitemOperation::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input,
    const ttnn::Tensor& value,
    tt::stl::Span<const T> begins,
    tt::stl::Span<const T> ends,
    tt::stl::Span<const T> steps,
    const std::optional<std::vector<ttnn::Tensor>>& index_tensors,
    const std::optional<tt::stl::Span<const T>> index_dims) {
    // Validate input tensor
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Input tensor must be on device");
    TT_FATAL(input.buffer() != nullptr, "Input tensor must have an allocated device buffer");

    // Validate value tensor
    TT_FATAL(value.storage_type() == StorageType::DEVICE, "Value tensor must be on device");
    TT_FATAL(value.buffer() != nullptr, "Value tensor must have an allocated device buffer");

    const auto& input_shape = input.logical_shape();
    const uint32_t input_rank = input_shape.rank();

    // Ensure value tensor has the same rank as input
    ttnn::Tensor aligned_value = value;
    while (aligned_value.logical_shape().rank() < input_rank) {
        aligned_value = ttnn::unsqueeze(aligned_value, 0);
    }
    const auto& value_shape = aligned_value.logical_shape();
    const uint32_t value_rank = value_shape.rank();

    std::vector<uint32_t> normalized_index_dims;

    // Validate index tensors
    if (index_tensors.has_value()) {
        TT_FATAL(index_dims.has_value(), "Index tensor dimensions must be specified");
        TT_FATAL(
            index_tensors->size() == index_dims->size(),
            "Expected {} index tensors but got {}",
            index_dims->size(),
            index_tensors->size());

        // log_debug(tt::LogOp, "BosSetitem index dims: {}", index_dims.value());
        std::optional<ttnn::Shape> index_shape;

        for (size_t i = 0; i < index_dims->size(); ++i) {
            T raw_dim = index_dims.value()[i];
            uint32_t dim = std::is_signed_v<T> ? wrap_index(raw_dim, input_rank) : raw_dim;
            TT_FATAL(dim < input_rank, "Index dimension {} out of bounds for rank {}", dim, input_rank);
            normalized_index_dims.push_back(dim);
            log_debug(tt::LogOp, "BosSetitem norm index dim: {}", dim);

            const auto& idx_tensor = index_tensors.value()[i];
            TT_FATAL(idx_tensor.storage_type() == StorageType::DEVICE, "Index tensor {} must be on device", dim);
            TT_FATAL(idx_tensor.buffer() != nullptr, "Index tensor {} must have a device buffer", dim);
            TT_FATAL(idx_tensor.logical_shape().rank() == 1, "Index tensor {} must be 1D", dim);

            if (!index_shape.has_value()) {
                index_shape = idx_tensor.logical_shape();
            } else {
                TT_FATAL(
                    index_shape.value() == idx_tensor.logical_shape(),
                    "Incompatible shape across index tensors, expected {}, got {}",
                    index_shape.value(),
                    idx_tensor.logical_shape());
            }
        }
    }

    // log_debug(tt::LogOp, "Begins {}, Ends {}, Steps {}", begins, ends, steps);
    log_debug(tt::LogOp, "Input shape {}, Value shape {}", input_shape, value_shape);

    // Slice/input compatibility
    TT_FATAL(
        input_rank == value_rank,
        "Input and value tensor rank mismatch: input rank = {}, value rank = {}",
        input_rank,
        value_rank);
    TT_FATAL(
        input_rank == begins.size(),
        "Expected total of {} slices for input rank = {}, but got {}",
        input_rank,
        input_rank,
        begins.size());
    TT_FATAL(begins.size() == ends.size(), "Mismatch between begins ({}), and ends ({})", begins.size(), ends.size());
    TT_FATAL(
        steps.size() == begins.size(), "Steps ({}), must match begins/ends size ({})", steps.size(), begins.size());

    // Normalize slices
    std::vector<uint32_t> norm_begins(input_rank, 0);
    std::vector<uint32_t> norm_ends(input_rank, 0);
    std::vector<uint32_t> norm_steps(input_rank, 1);

    for (size_t dim = 0; dim < begins.size(); ++dim) {
        if constexpr (std::is_signed_v<T>) {
            norm_begins[dim] = wrap_index(begins[dim], input_shape[dim]);
            norm_ends[dim] = wrap_index(ends[dim], input_shape[dim]);
            norm_steps[dim] = static_cast<uint32_t>(steps[dim]);
        } else {
            norm_begins[dim] = begins[dim];
            norm_ends[dim] = ends[dim];
            norm_steps[dim] = steps[dim];
        }

        if (std::find(normalized_index_dims.begin(), normalized_index_dims.end(), dim) != normalized_index_dims.end()) {
            TT_FATAL(
                (norm_begins[dim] == 0 && norm_ends[dim] == input_shape[dim] && norm_steps[dim] == 1),
                "Cannot apply slice at dim ({}) that indicated by index tensor, expected begins, ends, steps = 0, {}, "
                "1, but got begins = {}, ends = {}, steps = {}",
                dim,
                input_shape[dim],
                norm_begins[dim],
                norm_ends[dim],
                norm_steps[dim]);
            continue;  // Skip index tensor dims
        }
        TT_FATAL(
            norm_ends[dim] >= norm_begins[dim],
            "Negative step not supported. End ({}) < Begin ({}) at dimension {}",
            norm_ends[dim],
            norm_begins[dim],
            dim);
        TT_FATAL(
            norm_ends[dim] <= input_shape[dim],
            "End ({}) out of bounds for input shape at dimension {}: {}",
            norm_ends[dim],
            dim,
            input_shape[dim]);
        TT_FATAL(norm_steps[dim] > 0, "Step must be > 0. Got {} at dimension {}", norm_steps[dim], dim);
    }
    log_debug(tt::LogOp, "Norm Begins {}, Norm Ends {}, Norm Steps {}", norm_begins, norm_ends, norm_steps);
    log_debug(tt::LogOp, "Norm index dims at dim {}", normalized_index_dims);

    // Broadcast-check and compute shape of slice region
    std::vector<uint32_t> sliced_shape_vec;
    uint32_t tensor_id = 0;
    sliced_shape_vec.reserve(input_rank);
    for (size_t i = 0; i < input_rank; ++i) {
        uint32_t slice_size;
        if (std::find(normalized_index_dims.begin(), normalized_index_dims.end(), i) != normalized_index_dims.end()) {
            const auto& idx_tensor = index_tensors.value()[tensor_id];  // Get if is index tensor dim
            slice_size = idx_tensor.logical_shape().volume();
            ++tensor_id;
        } else {
            slice_size = (norm_ends[i] - norm_begins[i] + norm_steps[i] - 1) / norm_steps[i];
        }

        log_debug(tt::LogOp, "Slice size at dim {}: {}", i, slice_size);
        TT_FATAL(
            (slice_size == 1) || (value_shape[i] == 1) || (slice_size == value_shape[i]),
            "Value shape [{}] not broadcastable to slice shape [{}] at dim {}",
            value_shape[i],
            slice_size,
            i);
        sliced_shape_vec.push_back(slice_size);
    }
    log_debug(tt::LogOp, "sliced_shape_vec {}", sliced_shape_vec);

    // Full overwrite check
    if (value_shape == input_shape) {
        bool is_empty_slice =
            std::any_of(sliced_shape_vec.begin(), sliced_shape_vec.end(), [](auto d) { return d == 0; });
        log_debug(tt::LogOp, "Is empty Bossetitem op: {}", is_empty_slice);
        if (is_empty_slice) {
            return input;
        }
        // TODO: make this inplace
        //  input = ttnn::clone(value, memory_config=input.memory_config());
        //  return input;
    }

    // Prepare optional index tensors
    std::vector<std::optional<const ttnn::Tensor>> converted_index_tensors;
    if (index_tensors.has_value()) {
        for (const auto& tensor : *index_tensors) {
            converted_index_tensors.emplace_back(tensor);
        }
    } else {
        converted_index_tensors.emplace_back(std::nullopt);
    }

    log_debug(tt::LogOp, "Running BosSetitem device operation");

    // Launch kernel
    return tt::tt_metal::operation::run(
               BosSetitem{norm_begins, norm_ends, norm_steps, sliced_shape_vec, normalized_index_dims},
               {input, aligned_value},
               {converted_index_tensors},
               {input})
        .at(0);
}

template <typename T>
ttnn::Tensor BosSetitemOperation::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input,
    const ttnn::Tensor& value,
    const ttnn::SmallVector<T> begins,
    const ttnn::SmallVector<T> ends,
    const std::optional<ttnn::SmallVector<T>> steps,
    const std::optional<std::vector<ttnn::Tensor>>& index_tensors,
    const std::optional<ttnn::SmallVector<T>> index_dims) {
    const auto step_value = steps.value_or(ttnn::SmallVector<T>(ends.size(), 1));
    std::optional<tt::stl::Span<const T>> index_dims_;
    if (index_dims.has_value()) {
        index_dims_ = tt::stl::Span<const T>(*index_dims);
    }

    return BosSetitemOperation::invoke<T>(
        queue_id,
        input,
        value,
        tt::stl::Span<const T>(begins),
        tt::stl::Span<const T>(ends),
        tt::stl::Span<const T>(step_value),
        index_tensors,
        index_dims_);
};

// ###################
//     Templates
// ###################

template ttnn::Tensor BosSetitemOperation::invoke<int>(
    QueueId queue_id,
    const ttnn::Tensor& input,
    const ttnn::Tensor& value,
    tt::stl::Span<const int> begins,
    tt::stl::Span<const int> ends,
    tt::stl::Span<const int> steps,
    const std::optional<std::vector<ttnn::Tensor>>& index_tensors,
    const std::optional<tt::stl::Span<const int>> index_dims);

template ttnn::Tensor BosSetitemOperation::invoke<uint32_t>(
    QueueId queue_id,
    const ttnn::Tensor& input,
    const ttnn::Tensor& value,
    tt::stl::Span<const uint32_t> begins,
    tt::stl::Span<const uint32_t> ends,
    tt::stl::Span<const uint32_t> steps,
    const std::optional<std::vector<ttnn::Tensor>>& index_tensors,
    const std::optional<tt::stl::Span<const uint32_t>> index_dims);

template ttnn::Tensor BosSetitemOperation::invoke<uint32_t>(
    QueueId queue_id,
    const ttnn::Tensor& input,
    const ttnn::Tensor& value,
    const ttnn::SmallVector<uint32_t> begins,
    const ttnn::SmallVector<uint32_t> ends,
    const std::optional<ttnn::SmallVector<uint32_t>> steps,
    const std::optional<std::vector<ttnn::Tensor>>& index_tensors,
    const std::optional<ttnn::SmallVector<uint32_t>> index_dims);

template ttnn::Tensor BosSetitemOperation::invoke<int>(
    QueueId queue_id,
    const ttnn::Tensor& input,
    const ttnn::Tensor& value,
    const ttnn::SmallVector<int> begins,
    const ttnn::SmallVector<int> ends,
    const std::optional<ttnn::SmallVector<int>> steps,
    const std::optional<std::vector<ttnn::Tensor>>& index_tensors,
    const std::optional<ttnn::SmallVector<int>> index_dims);

}  // namespace ttnn::operations::bos_setitem
