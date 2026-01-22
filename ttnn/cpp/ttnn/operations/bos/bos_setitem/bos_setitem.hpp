#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/core/core.hpp"

using SliceType = std::variant<std::vector<uint32_t>, std::vector<int>, ttnn::Tensor>;

namespace ttnn {
namespace operations {
namespace bos_setitem {

struct BosSetitemOperation {
    template <typename T>
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input,
        const ttnn::Tensor& value,
        tt::stl::Span<const T> begins,
        tt::stl::Span<const T> ends,
        tt::stl::Span<const T> steps,
        const std::optional<std::vector<ttnn::Tensor>>& index_tensors = std::nullopt,
        const std::optional<tt::stl::Span<const T>> index_dims = std::nullopt);

    template <typename T>
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input,
        const ttnn::Tensor& value,
        const ttnn::SmallVector<T> begins,
        const ttnn::SmallVector<T> ends,
        const std::optional<ttnn::SmallVector<T>> steps = std::nullopt,
        const std::optional<std::vector<ttnn::Tensor>>& index_tensors = std::nullopt,
        const std::optional<ttnn::SmallVector<T>> index_dims = std::nullopt);
};

}  // namespace bos_setitem
}  // namespace operations

constexpr auto bos_setitem = ttnn::register_operation<"ttnn::bos_setitem", ttnn::operations::bos_setitem::BosSetitemOperation>();

}  // namespace ttnn
