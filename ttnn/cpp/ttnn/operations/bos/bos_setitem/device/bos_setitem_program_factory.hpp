#pragma once

#include <tt-metalium/host_api.hpp>

namespace ttnn::operations::bos_setitem {

tt::tt_metal::operation::ProgramWithCallbacks bos_setitem_multi_core_rm_interleaved(
    const Tensor& input,
    const Tensor& value,
    const std::vector<uint32_t> begins,
    const std::vector<uint32_t> ends,
    const std::vector<uint32_t> steps,
    const std::vector<uint32_t> set_shape,
    const std::vector<std::optional<const Tensor>> index_tensors,
    const std::optional<std::vector<uint32_t>> index_dims = std::nullopt);

tt::tt_metal::operation::ProgramWithCallbacks bos_setitem_multi_core_tile_interleaved(
    const Tensor& input,
    const Tensor& value,
    const std::vector<uint32_t> begins,
    const std::vector<uint32_t> ends,
    const std::vector<uint32_t> steps,
    const std::vector<uint32_t> set_shape,
    const std::vector<std::optional<const Tensor>> index_tensors,
    const std::optional<std::vector<uint32_t>> index_dims = std::nullopt);

}  // namespace ttnn::operations::bos_setitem
