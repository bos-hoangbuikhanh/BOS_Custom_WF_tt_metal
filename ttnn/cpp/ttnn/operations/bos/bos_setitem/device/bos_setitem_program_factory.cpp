#include <algorithm>
#include <cmath>
#include <tt-metalium/math.hpp>

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>

#include "bos_setitem_op.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
using namespace tt::constants;
using namespace tt::tt_metal;
using namespace ttnn::operations;

namespace ttnn::operations::bos_setitem {
inline std::pair<std::vector<uint32_t>, std::vector<uint32_t>> get_bos_setitem_runtime_args(
    tt::tt_metal::Buffer* input_buffer,
    tt::tt_metal::Buffer* value_buffer,
    uint32_t n_ttnn_tensors,
    const tt::tt_metal::Shape& input_shape,
    const ttnn::Shape& set_shape_,
    const tt::tt_metal::Shape& value_shape,
    const std::vector<uint32_t>& begins,
    const std::vector<uint32_t>& steps,
    const std::vector<uint32_t>& is_ttnn_tensor,
    const std::vector<std::optional<const Tensor>>& index_tensors) {
    std::vector<uint32_t> reader_runtime_args = {input_buffer->address(), value_buffer->address(), 0, 0};
    reader_runtime_args.insert(reader_runtime_args.end(), input_shape.cbegin(), input_shape.cend());
    reader_runtime_args.insert(reader_runtime_args.end(), set_shape_.cbegin(), set_shape_.cend());
    reader_runtime_args.insert(reader_runtime_args.end(), value_shape.cbegin(), value_shape.cend());
    reader_runtime_args.insert(reader_runtime_args.end(), begins.begin(), begins.end());
    reader_runtime_args.insert(reader_runtime_args.end(), steps.begin(), steps.end());
    reader_runtime_args.insert(reader_runtime_args.end(), is_ttnn_tensor.begin(), is_ttnn_tensor.end());
    for (uint32_t id = 0; id < n_ttnn_tensors; ++id) {
        auto id_tensor = index_tensors[id].value();
        reader_runtime_args.insert(reader_runtime_args.end(), id_tensor.buffer()->address());
    }
    for (uint32_t id = 0; id < n_ttnn_tensors; ++id) {
        auto id_tensor = index_tensors[id].value();
        auto id_esize = id_tensor.element_size();
        auto num_units = id_tensor.logical_shape().volume();
        uint32_t page_size = num_units * id_esize;
        reader_runtime_args.insert(reader_runtime_args.end(), page_size);
    }
    for (uint32_t id = 0; id < n_ttnn_tensors; ++id) {
        auto id_tensor = index_tensors[id].value();
        uint32_t is_dram = id_tensor.buffer()->buffer_type() == BufferType::DRAM ? 1 : 0;
        reader_runtime_args.insert(reader_runtime_args.end(), is_dram);
    }

    std::vector<uint32_t> writer_runtime_args = {input_buffer->address(), 0, 0};
    writer_runtime_args.insert(writer_runtime_args.end(), input_shape.cbegin(), input_shape.cend());
    writer_runtime_args.insert(writer_runtime_args.end(), set_shape_.cbegin(), set_shape_.cend());
    writer_runtime_args.insert(writer_runtime_args.end(), begins.begin(), begins.end());
    writer_runtime_args.insert(writer_runtime_args.end(), steps.begin(), steps.end());
    writer_runtime_args.insert(writer_runtime_args.end(), is_ttnn_tensor.begin(), is_ttnn_tensor.end());
    for (uint32_t id = 0; id < n_ttnn_tensors; ++id) {
        auto id_tensor = index_tensors[id].value();
        writer_runtime_args.insert(writer_runtime_args.end(), id_tensor.buffer()->address());
    }
    for (uint32_t id = 0; id < n_ttnn_tensors; ++id) {
        auto id_tensor = index_tensors[id].value();
        auto id_esize = id_tensor.element_size();
        auto num_units = id_tensor.logical_shape().volume();
        uint32_t page_size = num_units * id_esize;
        writer_runtime_args.insert(writer_runtime_args.end(), page_size);
    }
    for (uint32_t id = 0; id < n_ttnn_tensors; ++id) {
        auto id_tensor = index_tensors[id].value();
        uint32_t is_dram = id_tensor.buffer()->buffer_type() == BufferType::DRAM ? 1 : 0;
        writer_runtime_args.insert(writer_runtime_args.end(), is_dram);
    }

    return {reader_runtime_args, writer_runtime_args};
}

operation::ProgramWithCallbacks bos_setitem_multi_core_rm_interleaved(
    const Tensor& input,
    const Tensor& value,
    const std::vector<uint32_t> begins,
    const std::vector<uint32_t> ends,
    const std::vector<uint32_t> steps,
    const std::vector<uint32_t> set_shape,
    const std::vector<std::optional<const Tensor>> index_tensors,
    const std::optional<std::vector<uint32_t>> index_dims) {
    //** 1. Initialize program and get device */
    Program program{};
    IDevice* device = input.device();

    //** 2. Input, output info */
    // format
    tt::DataFormat input_df = datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat value_df = datatype_to_dataformat_converter(value.dtype());

    // size
    uint32_t input_esize = input.element_size();
    uint32_t value_esize = value.element_size();

    const auto& input_shape = input.logical_shape();
    const auto& value_shape = value.logical_shape();
    uint32_t input_rank = input_shape.rank();
    std::vector<uint32_t> actual_set_shape = set_shape;

    log_debug(tt::LogOp, "Norm Begins: {}, Norm Ends: {}, Norm Steps: {}, ", begins, ends, steps);

    if (index_dims.has_value()) {
        auto id_dims = index_dims.value();
        uint32_t captured_broadcast_size = 0;  // as flag indicate broadcast size has been captured

        for (uint32_t dim = 0; dim < input_rank; ++dim) {
            bool is_index_dim = std::find(id_dims.begin(), id_dims.end(), dim) != id_dims.end();
            uint32_t size = set_shape[dim];

            if (is_index_dim && (captured_broadcast_size != 0 || size <= 1)) {
                actual_set_shape[dim] = 1;  // if is index and size <= 1
            } else {
                if (is_index_dim) {
                    captured_broadcast_size = size;
                }
                actual_set_shape[dim] = size;
            }
        }
    }
    ttnn::Shape set_shape_(set_shape);
    ttnn::Shape actual_set_shape_(actual_set_shape);

    uint32_t input_num_units = input_shape[input_rank - 1];
    uint32_t value_num_units = value_shape[input_rank - 1];
    uint32_t inner_slices_num_units = actual_set_shape_[input_rank - 1];
    [[maybe_unused]] uint32_t input_num_sticks = input_shape.volume() / input_num_units;
    [[maybe_unused]] uint32_t value_num_sticks = value_shape.volume() / value_num_units;
    uint32_t outer_slices_num_units = actual_set_shape_.volume() / inner_slices_num_units;
    uint32_t is_broadcast_last_dim = (uint32_t)(value_num_units == 1 & inner_slices_num_units != value_num_units);
    uint32_t n_ttnn_tensors = index_dims.has_value() ? index_dims.value().size() : 0;
    std::vector<uint32_t> is_ttnn_tensor(input_rank, 0);
    if (n_ttnn_tensors > 0) {
        auto id_dims = index_dims.value();
        for (auto dim : id_dims) {
            is_ttnn_tensor[dim] = 1;
        }
    }

    log_debug(
        tt::LogOp,
        "slice num units {} {}, inp num units {} {}, inp rank {}, is_broadcast_last_dim {}",
        inner_slices_num_units,
        outer_slices_num_units,
        input_num_units,
        value_num_units,
        input_rank,
        is_broadcast_last_dim);
    log_debug(
        tt::LogOp,
        "Slice shape {} (actual set shape {}) from input shape {} and value shape {}",
        set_shape,
        actual_set_shape_,
        input_shape,
        value_shape);

    // buffers
    auto input_buffer = input.buffer();
    auto value_buffer = value.buffer();

    bool input_is_dram = input_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
    bool value_is_dram = value_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;

    //** 3. Split work to cores */
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    [[maybe_unused]] uint32_t num_units = num_cores_x * num_cores_y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_units_per_core_group_1, num_units_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, outer_slices_num_units);
    auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, false);

    //** 4. Circular Buffers Allocation */
    uint32_t next_cb_index = tt::CBIndex::c_0;
    auto SetitemCB = [&program, &all_cores, &next_cb_index](
                         uint32_t page_size, uint32_t num_pages, tt::DataFormat df, Buffer* buffer) {
        return tt::tt_metal::create_cb(next_cb_index++, program, all_cores, page_size, num_pages, df, buffer);
    };
    auto [input_cbi, input_cb] = SetitemCB(input_esize * input_num_units, 1, input_df, nullptr);
    [[maybe_unused]] auto [input_cbi_, input_cb_] = SetitemCB(input_esize * input_num_units, 1, input_df, nullptr);
    auto [value_cbi, value_cb] = SetitemCB(value_esize * value_num_units, 1, value_df, nullptr);
    if (n_ttnn_tensors > 0) {
        auto id_dims = index_dims.value();
        for (uint32_t dim = 0; dim < id_dims.size(); ++dim) {
            auto id_esize = index_tensors[dim].value().element_size();
            auto num_units = index_tensors[dim].value().logical_shape().volume();
            tt::DataFormat id_df = datatype_to_dataformat_converter(index_tensors[dim].value().dtype());
            [[maybe_unused]] auto [id_tensor_cbi, id_tensor_cb] = SetitemCB(id_esize * num_units, 1, id_df, nullptr);
            log_debug(tt::LogOp, "Create cb {} for id tensor {}", id_tensor_cbi, dim);
        }
    }

    //** 5. Compile-time arguments */
    std::vector<uint32_t> reader_compile_time_args = {// circular buffer
                                                      input_cbi,
                                                      value_cbi,
                                                      input_is_dram,
                                                      value_is_dram,
                                                      // size
                                                      inner_slices_num_units,
                                                      input_num_units,
                                                      value_num_units,
                                                      input_rank,
                                                      input_esize,
                                                      is_broadcast_last_dim,
                                                      n_ttnn_tensors};

    std::vector<uint32_t> writer_compile_time_args = {// circular buffer
                                                      input_cbi,
                                                      input_is_dram,
                                                      // size
                                                      input_num_units,
                                                      input_esize,
                                                      input_rank,
                                                      n_ttnn_tensors};

    bool is_inner_strided = inner_slices_num_units > 1 ? (steps[input_rank - 1] > 1 ? true : false) : false;
    if (index_dims.has_value()) {
        auto id_dim = index_dims.value();
        is_inner_strided = is_inner_strided || std::find(id_dim.begin(), id_dim.end(), input_rank - 1) != id_dim.end();
    }
    std::string reader_kernel_path = is_inner_strided ? "ttnn/cpp/ttnn/operations/bos/bos_setitem/device/kernels/"
                                                        "dataflow/bos_setitem_strided_reader_rm_interleaved_nd.cpp"
                                                      : "ttnn/cpp/ttnn/operations/bos/bos_setitem/device/kernels/"
                                                        "dataflow/bos_setitem_reader_unpad_dims_rm_interleaved_start_id.cpp";
    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program, reader_kernel_path, all_cores, ReaderDataMovementConfig(reader_compile_time_args));

    std::string writer_kernel_path =
        "ttnn/cpp/ttnn/operations/bos/bos_setitem/device/kernels/dataflow/"
        "bos_setitem_writer_stick_layout_interleaved_start_id.cpp";
    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program, writer_kernel_path, all_cores, WriterDataMovementConfig(writer_compile_time_args));

    auto reader_writer_runtime_args = get_bos_setitem_runtime_args(
        input_buffer,
        value_buffer,
        n_ttnn_tensors,
        input_shape,
        set_shape_,
        value_shape,
        begins,
        steps,
        is_ttnn_tensor,
        index_tensors);
    auto reader_runtime_args = reader_writer_runtime_args.first;
    auto writer_runtime_args = reader_writer_runtime_args.second;

    //** 6. Runtime arguments */
    uint32_t start_id = 0;
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores.at(i);
        bool is_group_1 = i < core_group_1.num_cores();
        uint32_t num_sticks_per_core = is_group_1 ? num_units_per_core_group_1 : num_units_per_core_group_2;
        reader_runtime_args[2] = start_id;
        reader_runtime_args[3] = num_sticks_per_core;
        writer_runtime_args[1] = start_id;
        writer_runtime_args[2] = num_sticks_per_core;
        SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
        SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);
        start_id += num_sticks_per_core;
    }
    auto override_runtime_args_callback = [cores,
                                           core_group_1,
                                           core_group_2,
                                           num_units_per_core_group_1,
                                           num_units_per_core_group_2,
                                           reader_kernel_id,
                                           writer_kernel_id,
                                           input_shape,
                                           set_shape_,
                                           value_shape,
                                           begins,
                                           ends,
                                           steps,
                                           is_ttnn_tensor,
                                           n_ttnn_tensors](
                                              const void* operation,
                                              Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>& optional_input_tensors,
                                              const std::vector<Tensor>& output_tensors) {
        auto input_buffer = input_tensors.at(0).buffer();
        auto value_buffer = input_tensors.at(1).buffer();

        auto reader_writer_runtime_args = get_bos_setitem_runtime_args(
            input_buffer,
            value_buffer,
            n_ttnn_tensors,
            input_shape,
            set_shape_,
            value_shape,
            begins,
            steps,
            is_ttnn_tensor,
            optional_input_tensors);
        auto reader_runtime_args = reader_writer_runtime_args.first;
        auto writer_runtime_args = reader_writer_runtime_args.second;
        uint32_t start_id = 0;
        for (uint32_t i = 0; i < cores.size(); i++) {
            bool is_group_1 = i < core_group_1.num_cores();
            uint32_t num_sticks_per_core = is_group_1 ? num_units_per_core_group_1 : num_units_per_core_group_2;
            reader_runtime_args[2] = start_id;
            reader_runtime_args[3] = num_sticks_per_core;
            writer_runtime_args[1] = start_id;
            writer_runtime_args[2] = num_sticks_per_core;
            SetRuntimeArgs(program, reader_kernel_id, cores[i], reader_runtime_args);
            SetRuntimeArgs(program, writer_kernel_id, cores[i], writer_runtime_args);
            start_id += num_sticks_per_core;
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

operation::ProgramWithCallbacks bos_setitem_multi_core_tile_interleaved(
    const Tensor& input,
    const Tensor& value,
    const std::vector<uint32_t> begins,
    const std::vector<uint32_t> ends,
    const std::vector<uint32_t> steps,
    const std::vector<uint32_t> set_shape,
    const std::vector<std::optional<const Tensor>> index_tensors,
    const std::optional<std::vector<uint32_t>> index_dims) {
    //** 1. Initialize program and get device */
    Program program{};
    IDevice* device = input.device();

    //** 2. Input, output info */
    // format
    [[maybe_unused]] tt::DataFormat input_df = datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat value_df = datatype_to_dataformat_converter(value.dtype());

    // size
    [[maybe_unused]] uint32_t input_esize = input.element_size();
    uint32_t value_esize = value.element_size();

    const auto& input_shape = input.padded_shape();
    const auto& value_shape = value.padded_shape();
    uint32_t input_rank = input_shape.rank();
    std::vector<uint32_t> actual_set_shape = set_shape;

    log_debug(tt::LogOp, "Norm Begins: {}, Norm Ends: {}, Norm Steps: {}, ", begins, ends, steps);

    if (index_dims.has_value()) {
        auto id_dims = index_dims.value();
        uint32_t captured_broadcast_size = 0;  // as flag indicate broadcast size has been captured

        for (uint32_t dim = 0; dim < input_rank; ++dim) {
            bool is_index_dim = std::find(id_dims.begin(), id_dims.end(), dim) != id_dims.end();
            uint32_t size = set_shape[dim];

            if (is_index_dim && (captured_broadcast_size != 0 || size <= 1)) {
                actual_set_shape[dim] = 1;  // if is index and size <= 1
            } else {
                if (is_index_dim) {
                    captured_broadcast_size = size;
                }
                actual_set_shape[dim] = size;
            }
        }
    }
    ttnn::Shape set_shape_(set_shape);
    ttnn::Shape actual_set_shape_(actual_set_shape);
    ttnn::Shape actual_pad_set_shape_ = ttnn::operations::data_movement::pad_to_tile_shape(actual_set_shape_);
    log_debug(tt::LogOp, "Actual pad set shape: {}", actual_pad_set_shape_);

    [[maybe_unused]] uint32_t input_num_units = 32 * 32;
    uint32_t value_num_units = 32 * 32;
    [[maybe_unused]] uint32_t inner_slices_num_units = actual_set_shape_[input_rank - 1];
    uint32_t num_tiles = actual_pad_set_shape_.volume() / (ttnn::TILE_SIZE * ttnn::TILE_SIZE);
    uint32_t n_ttnn_tensors = index_dims.has_value() ? index_dims.value().size() : 0;
    std::vector<uint32_t> is_ttnn_tensor(input_rank, 0);
    if (n_ttnn_tensors > 0) {
        auto id_dims = index_dims.value();
        for (auto dim : id_dims) {
            is_ttnn_tensor[dim] = 1;
        }
    }

    log_debug(
        tt::LogOp,
        "slice num units {}, inp num units {} {}, inp rank {}",
        inner_slices_num_units,
        input_num_units,
        value_num_units,
        input_rank);
    log_debug(
        tt::LogOp,
        "Slice shape {} (actual set shape {}) from input shape {} and value shape {}",
        set_shape,
        actual_set_shape_,
        input_shape,
        value_shape);
    log_debug(tt::LogOp, "input_shape.volume(): {}, num_tiles: {}", input_shape.volume(), num_tiles);

    // buffers
    auto input_buffer = input.buffer();
    auto value_buffer = value.buffer();

    bool input_is_dram = input_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
    bool value_is_dram = value_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;

    //** 3. Split work to cores */
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    [[maybe_unused]] uint32_t num_units = num_cores_x * num_cores_y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_units_per_core_group_1, num_units_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_tiles);
    auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, false);

    //** 4. Circular Buffers Allocation */
    uint32_t next_cb_index = tt::CBIndex::c_0;
    auto SetitemCB = [&program, &all_cores, &next_cb_index](
                         uint32_t page_size, uint32_t num_pages, tt::DataFormat df, Buffer* buffer) {
        return tt::tt_metal::create_cb(next_cb_index++, program, all_cores, page_size, num_pages, df, buffer);
    };
    auto [value_cbi, value_cb] = SetitemCB(value_esize * value_num_units, 1, value_df, nullptr);
    [[maybe_unused]] auto [value_cbi_, value_cb_] = SetitemCB(value_esize * value_num_units, 1, value_df, nullptr);
    if (n_ttnn_tensors > 0) {
        auto id_dims = index_dims.value();
        for (uint32_t dim = 0; dim < id_dims.size(); ++dim) {
            auto id_esize = index_tensors[dim].value().element_size();
            auto num_units = index_tensors[dim].value().logical_shape().volume();
            tt::DataFormat id_df = datatype_to_dataformat_converter(index_tensors[dim].value().dtype());
            [[maybe_unused]] auto [id_tensor_cbi, id_tensor_cb] = SetitemCB(id_esize * num_units, 1, id_df, nullptr);
            log_debug(tt::LogOp, "Create cb {} for id tensor {}", id_tensor_cbi, dim);
        }
    }

    //** 5. Compile-time arguments */
    std::vector<uint32_t> compile_time_args = {// circular buffer
                                               value_cbi,
                                               input_is_dram,
                                               value_is_dram,
                                               // size
                                               input_rank,
                                               n_ttnn_tensors};

    std::string reader_kernel_path =
        "ttnn/cpp/ttnn/operations/bos/bos_setitem/device/kernels/dataflow/"
        "bos_setitem_reader_unpad_dims_tile_interleaved_start_id.cpp";
    tt::tt_metal::KernelHandle reader_kernel_id =
        tt::tt_metal::CreateKernel(program, reader_kernel_path, all_cores, ReaderDataMovementConfig(compile_time_args));

    std::string writer_kernel_path =
        "ttnn/cpp/ttnn/operations/bos/bos_setitem/device/kernels/dataflow/"
        "bos_setitem_writer_tile_interleaved_start_id.cpp";
    tt::tt_metal::KernelHandle writer_kernel_id =
        tt::tt_metal::CreateKernel(program, writer_kernel_path, all_cores, WriterDataMovementConfig(compile_time_args));

    auto reader_writer_runtime_args = get_bos_setitem_runtime_args(
        input_buffer,
        value_buffer,
        n_ttnn_tensors,
        input_shape,
        set_shape_,
        value_shape,
        begins,
        steps,
        is_ttnn_tensor,
        index_tensors);
    auto reader_runtime_args = reader_writer_runtime_args.first;
    auto writer_runtime_args = reader_writer_runtime_args.second;

    //** 6. Runtime arguments */
    uint32_t start_id = 0;
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores.at(i);
        bool is_group_1 = i < core_group_1.num_cores();
        uint32_t num_tiles_per_core = is_group_1 ? num_units_per_core_group_1 : num_units_per_core_group_2;
        reader_runtime_args[2] = start_id;
        reader_runtime_args[3] = num_tiles_per_core;
        writer_runtime_args[1] = start_id;
        writer_runtime_args[2] = num_tiles_per_core;
        SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
        SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);
        start_id += num_tiles_per_core;
    }
    auto override_runtime_args_callback = [cores,
                                           core_group_1,
                                           core_group_2,
                                           num_units_per_core_group_1,
                                           num_units_per_core_group_2,
                                           reader_kernel_id,
                                           writer_kernel_id,
                                           input_shape,
                                           set_shape_,
                                           value_shape,
                                           begins,
                                           ends,
                                           steps,
                                           is_ttnn_tensor,
                                           n_ttnn_tensors](
                                              const void* operation,
                                              Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>& optional_input_tensors,
                                              const std::vector<Tensor>& output_tensors) {
        auto input_buffer = input_tensors.at(0).buffer();
        auto value_buffer = input_tensors.at(1).buffer();

        auto reader_writer_runtime_args = get_bos_setitem_runtime_args(
            input_buffer,
            value_buffer,
            n_ttnn_tensors,
            input_shape,
            set_shape_,
            value_shape,
            begins,
            steps,
            is_ttnn_tensor,
            optional_input_tensors);
        auto reader_runtime_args = reader_writer_runtime_args.first;
        auto writer_runtime_args = reader_writer_runtime_args.second;
        uint32_t start_id = 0;
        for (uint32_t i = 0; i < cores.size(); i++) {
            bool is_group_1 = i < core_group_1.num_cores();
            uint32_t num_tiles_per_core = is_group_1 ? num_units_per_core_group_1 : num_units_per_core_group_2;
            reader_runtime_args[2] = start_id;
            reader_runtime_args[3] = num_tiles_per_core;
            writer_runtime_args[1] = start_id;
            writer_runtime_args[2] = num_tiles_per_core;
            SetRuntimeArgs(program, reader_kernel_id, cores[i], reader_runtime_args);
            SetRuntimeArgs(program, writer_kernel_id, cores[i], writer_runtime_args);
            start_id += num_tiles_per_core;
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}
}  // namespace ttnn::operations::bos_setitem
