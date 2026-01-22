#pragma once

#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::bos_setitem {
struct BosSetitem {
    std::vector<uint32_t> begins;
    std::vector<uint32_t> ends;
    std::vector<uint32_t> steps;
    std::vector<uint32_t> slice_shape;
    std::optional<std::vector<uint32_t>> index_dims = std::nullopt;

    void validate(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;
    std::vector<TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors) const;
};

// tt::tt_metal::operation::ProgramWithCallbacks bos_setitem_multi_core_rm_interleaved(const Tensor& input, const Tensor&
// value);

}  // namespace ttnn::operations::bos_setitem
