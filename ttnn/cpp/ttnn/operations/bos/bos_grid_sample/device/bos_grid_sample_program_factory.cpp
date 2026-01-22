#include <algorithm>
#include <cmath>
#include <tt-metalium/math.hpp>

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/operation.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/permute/device/permute_device_operation.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"
#include <tt-metalium/constants.hpp>

#define MASK_64 0xFFFFFFFFFFFFFFC0
#define OFFSET_64 0x000000000000003F
#define MASK_16 0xFFFFFFFFFFFFFFF0
#define OFFSET_16 0x000000000000000F

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::operations::bos::bos_grid_sample {
using namespace tt::constants;
tt::tt_metal::operation::ProgramWithCallbacks sample_multi_core_nearest(
    const Tensor& input, const Tensor& grid, const bool align_corners, const Tensor& output) {
    Program program{};
    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(input.dtype());
    uint32_t input_unit_size = input.element_size();
    tt::DataFormat grid_cb_data_format = datatype_to_dataformat_converter(grid.dtype());
    uint32_t grid_unit_size = grid.element_size();

    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_unit_size = output.element_size();

    IDevice* device = output.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto core_grid = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});
    [[maybe_unused]] uint32_t num_units = num_cores_x * num_cores_y;

    const auto& input_shape = input.padded_shape();
    const auto& grid_shape = grid.padded_shape();
    const auto& output_shape = output.padded_shape();

    uint32_t num_sticks = grid_shape.volume() / grid_shape[3];  // BHoWo
    auto [num_cores, all_cores, core_group_1, core_group_2, num_sticks_per_core_group_1, num_sticks_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_sticks, true);

    uint32_t src0_cb0_index = tt::CBIndex::c_0;
    uint32_t num_src0_units = input_shape[3];  // C
    uint32_t aligned_src0_unit_size = num_src0_units * input_unit_size;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(aligned_src0_unit_size, {{src0_cb0_index, input_cb_data_format}})
            .set_page_size(src0_cb0_index, aligned_src0_unit_size);
    [[maybe_unused]] auto cb0_src0 = CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t src1_cb_index = tt::CBIndex::c_4;
    uint32_t num_src1_units = grid_shape[3];  // 2
    uint32_t aligned_src1_unit_size = num_src1_units * grid_unit_size;
    CircularBufferConfig cb_src1_config =
        CircularBufferConfig(aligned_src1_unit_size, {{src1_cb_index, grid_cb_data_format}})
            .set_page_size(src1_cb_index, aligned_src1_unit_size);
    [[maybe_unused]] auto cb_src1 = CreateCircularBuffer(program, all_cores, cb_src1_config);

    uint32_t scalar_cb_index = tt::CBIndex::c_24;
    uint32_t num_scalar_units = 1;                                                 // 1 for nearest
    uint32_t aligned_scalar_unit_size = num_scalar_units * (input_unit_size * 2);  // float32
    CircularBufferConfig cb_scalar_config =
        CircularBufferConfig(aligned_scalar_unit_size, {{scalar_cb_index, tt::DataFormat::Float32}})
            .set_page_size(scalar_cb_index, aligned_scalar_unit_size);
    [[maybe_unused]] auto cb_scalar = CreateCircularBuffer(program, all_cores, cb_scalar_config);

    uint32_t out_cb_index = tt::CBIndex::c_16;
    uint32_t num_out_units = output_shape[3];  // C
    uint32_t aligned_out_unit_size = num_out_units * output_unit_size;
    CircularBufferConfig cb_out_config =
        CircularBufferConfig(aligned_out_unit_size, {{out_cb_index, output_cb_data_format}})
            .set_page_size(out_cb_index, aligned_out_unit_size);
    [[maybe_unused]] auto cb_out = CreateCircularBuffer(program, all_cores, cb_out_config);

    auto src0_buffer = input.buffer();
    auto src1_buffer = grid.buffer();
    auto dst_buffer = output.buffer();
    bool src0_is_dram = src0_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
    bool src1_is_dram = src1_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
    bool dst_is_dram = dst_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;

    std::vector<uint32_t> reader_compile_time_args = {
        src0_cb0_index,
        src0_cb0_index,  // not used
        src0_cb0_index,  // not used
        src0_cb0_index,  // not used
        src1_cb_index,  scalar_cb_index,        out_cb_index,           src0_is_dram,          src1_is_dram,
        dst_is_dram,    aligned_src0_unit_size, aligned_src1_unit_size, aligned_out_unit_size, input_shape[3],
        input_shape[1], input_shape[2],         grid_shape[1],          grid_shape[2],         align_corners,
    };
    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/bos/bos_grid_sample/device/kernels/dataflow/"
        "bos_reader_nearest_input_sample_interleaved.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_compile_time_args));

    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/bos/bos_grid_sample/device/kernels/dataflow/"
        "bos_writer_nearest_input_sample_interleaved.cpp",
        all_cores,
        WriterDataMovementConfig(reader_compile_time_args));

    auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, true);
    uint32_t start_id = 0;
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores.at(i);
        uint32_t n_sticks_per_core;
        if (i < core_group_1.num_cores()) {
            n_sticks_per_core = num_sticks_per_core_group_1;
        } else {
            n_sticks_per_core = num_sticks_per_core_group_2;
        }
        SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {src0_buffer->address(), src1_buffer->address(), dst_buffer->address(), start_id, n_sticks_per_core});
        SetRuntimeArgs(
            program,
            writer_kernel_id,
            core,
            {src0_buffer->address(), src1_buffer->address(), dst_buffer->address(), start_id, n_sticks_per_core});
        start_id += n_sticks_per_core;
    }

    auto override_runtime_args_callback = [reader_kernel_id,
                                           writer_kernel_id,
                                           cores,
                                           core_group_1,
                                           num_sticks_per_core_group_1,
                                           num_sticks_per_core_group_2](
                                              const void* operation,
                                              const Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>&,
                                              const std::vector<Tensor>& output_tensors) {
        auto src0_buffer = input_tensors.at(0).buffer();
        auto src1_buffer = input_tensors.at(1).buffer();
        auto dst_buffer = output_tensors.at(0).buffer();
        uint32_t start_id = 0;
        for (uint32_t i = 0; i < cores.size(); i++) {
            auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel_id, cores[i]);
            reader_runtime_args[0] = src0_buffer->address();
            reader_runtime_args[1] = src1_buffer->address();
            reader_runtime_args[2] = dst_buffer->address();
            reader_runtime_args[3] = start_id;
            auto n_sticks_per_core =
                i < core_group_1.num_cores() ? num_sticks_per_core_group_1 : num_sticks_per_core_group_2;
            reader_runtime_args[4] = n_sticks_per_core;

            auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id, cores[i]);
            writer_runtime_args[0] = src0_buffer->address();
            writer_runtime_args[1] = src1_buffer->address();
            writer_runtime_args[2] = dst_buffer->address();
            writer_runtime_args[3] = start_id;
            writer_runtime_args[4] = n_sticks_per_core;
            start_id += n_sticks_per_core;
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

// Grid sample bilinear operation for multi-core
tt::tt_metal::operation::ProgramWithCallbacks sample_multi_core_bilinear(
    const Tensor& input,
    const Tensor& grid,
    const bool align_corners,
    const Tensor& output,
    const DeviceComputeKernelConfig compute_kernel_config) {
    constexpr uint32_t MAX_TILES_PER_REDUCTION = 8;

    Program program{};
    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(input.dtype());
    uint32_t input_unit_size = input.element_size();
    tt::DataFormat grid_cb_data_format = datatype_to_dataformat_converter(grid.dtype());
    uint32_t grid_unit_size = grid.element_size();

    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    [[maybe_unused]] uint32_t output_unit_size = output.element_size();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(input.device()->arch(), compute_kernel_config);

    IDevice* device = output.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto core_grid = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});
    [[maybe_unused]] uint32_t num_units = num_cores_x * num_cores_y;

    const auto& input_shape = input.padded_shape();
    const auto& grid_shape = grid.padded_shape();
    const auto& output_shape = output.padded_shape();

    uint32_t num_sticks = grid_shape.volume() / grid_shape[3];  // BHoWo
    auto [num_cores, all_cores, core_group_1, core_group_2, num_sticks_per_core_group_1, num_sticks_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_sticks, true);

    uint32_t in_ntiles_c = (uint32_t)std::ceil((float)input_shape[3] / tt::constants::TILE_WIDTH);
    uint32_t out_ntiles_c = in_ntiles_c;
    uint32_t num_input_width_blocks =
        std::ceil((float)(input_shape[3]) / (MAX_TILES_PER_REDUCTION * tt::constants::TILE_WIDTH));
    uint32_t input_stick_nbytes = input_shape[-1] * input_unit_size;
    uint32_t in0_cb_pagesize =
        std::min(tt::constants::TILE_WIDTH * input_unit_size * MAX_TILES_PER_REDUCTION, input_stick_nbytes);

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    [[maybe_unused]] uint32_t num_src0_units = input_shape[3];  // C
    uint32_t aligned_src0_unit_size = round_up_to_mul32(in0_cb_pagesize);
    // total size and page size of a circular buffer is set to the same
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(4 * aligned_src0_unit_size, {{src0_cb_index, input_cb_data_format}})
            .set_page_size(src0_cb_index, aligned_src0_unit_size);
    [[maybe_unused]] auto cb_src0 = CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t src1_cb_index = tt::CBIndex::c_4;
    uint32_t num_src1_units = grid_shape[3];  // 2
    uint32_t src1_unit_size = num_src1_units * grid_unit_size;
    uint32_t aligned_src1_unit_size =
        ((src1_unit_size - 1) & MASK_64) + 128;  // idk why i mess w this formula, but it basically page alignment
    CircularBufferConfig cb_src1_config =
        CircularBufferConfig(aligned_src1_unit_size, {{src1_cb_index, grid_cb_data_format}})
            .set_page_size(src1_cb_index, aligned_src1_unit_size);
    [[maybe_unused]] auto cb_src1 = CreateCircularBuffer(program, all_cores, cb_src1_config);

    uint32_t scalar_cb_index = tt::CBIndex::c_24;
    uint32_t num_scalar_units = 4;
    uint32_t aligned_scalar_unit_size = round_up_to_mul32(num_scalar_units * input_unit_size);  // bfloat16
    CircularBufferConfig cb_scalar_config =
        CircularBufferConfig(aligned_scalar_unit_size, {{scalar_cb_index, input_cb_data_format}})
            .set_page_size(scalar_cb_index, aligned_scalar_unit_size);
    [[maybe_unused]] auto cb_scalar = CreateCircularBuffer(program, all_cores, cb_scalar_config);

    uint32_t out_cb_index = tt::CBIndex::c_16;
    [[maybe_unused]] uint32_t num_out_units = output_shape[3];  // C
    uint32_t aligned_out_unit_size = round_up_to_mul32(in0_cb_pagesize);
    CircularBufferConfig cb_out_config =
        CircularBufferConfig(aligned_out_unit_size, {{out_cb_index, output_cb_data_format}})
            .set_page_size(out_cb_index, aligned_out_unit_size);
    [[maybe_unused]] auto cb_out = CreateCircularBuffer(program, all_cores, cb_out_config);

    auto src0_buffer = input.buffer();
    auto src1_buffer = grid.buffer();
    auto dst_buffer = output.buffer();
    bool src0_is_dram = src0_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
    bool src1_is_dram = src1_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
    bool dst_is_dram = dst_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;

    std::vector<uint32_t> reader_compile_time_args = {
        src0_cb_index,
        src1_cb_index,
        scalar_cb_index,
        src0_is_dram,
        src1_is_dram,
        input_stick_nbytes,
        aligned_src0_unit_size,
        src1_unit_size,
        input_shape[1],
        input_shape[2],
        grid_shape[1],
        grid_shape[2],
        align_corners,
        num_input_width_blocks};

    std::vector<uint32_t> writer_compile_time_args = {
        out_cb_index, in_ntiles_c, dst_is_dram, input_stick_nbytes, aligned_out_unit_size, num_input_width_blocks};

    std::vector<uint32_t> compute_compile_time_args = {
        src0_cb_index, scalar_cb_index, out_cb_index, in_ntiles_c, out_ntiles_c, num_input_width_blocks};

    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/bos/bos_grid_sample/device/kernels/dataflow/bos_reader_input_sample_interleaved.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_compile_time_args));

    auto reduce_op = ReduceOpMath::SUM;
    auto reduce_dim = ReduceOpDim::H;
    auto compute_config = ComputeConfig{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .math_approx_mode = math_approx_mode,
        .compile_args = compute_compile_time_args,
        .defines = reduce_op_utils::get_defines(reduce_op, reduce_dim)};
    KernelHandle compute_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/bos/bos_grid_sample/device/kernels/compute/bos_grid_sample_bilinear.cpp",
        all_cores,
        compute_config);

    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/bos/bos_grid_sample/device/kernels/dataflow/bos_writer_input_sample_interleaved.cpp",
        all_cores,
        WriterDataMovementConfig(writer_compile_time_args));

    auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, true);
    uint32_t start_id = 0;
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores.at(i);
        uint32_t n_sticks_per_core;
        if (i < core_group_1.num_cores()) {
            n_sticks_per_core = num_sticks_per_core_group_1;
        } else {
            n_sticks_per_core = num_sticks_per_core_group_2;
        }
        SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {src0_buffer->address(), src1_buffer->address(), dst_buffer->address(), start_id, n_sticks_per_core});
        SetRuntimeArgs(program, compute_kernel_id, core, {n_sticks_per_core});
        SetRuntimeArgs(
            program,
            writer_kernel_id,
            core,
            {src0_buffer->address(), src1_buffer->address(), dst_buffer->address(), start_id, n_sticks_per_core});
        start_id += n_sticks_per_core;
    }

    auto override_runtime_args_callback = [reader_kernel_id,
                                           compute_kernel_id,
                                           writer_kernel_id,
                                           cores,
                                           core_group_1,
                                           num_sticks_per_core_group_1,
                                           num_sticks_per_core_group_2](
                                              const void* operation,
                                              const Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>&,
                                              const std::vector<Tensor>& output_tensors) {
        auto src0_buffer = input_tensors.at(0).buffer();
        auto src1_buffer = input_tensors.at(1).buffer();
        auto dst_buffer = output_tensors.at(0).buffer();
        uint32_t start_id = 0;
        for (uint32_t i = 0; i < cores.size(); i++) {
            auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel_id, cores[i]);
            reader_runtime_args[0] = src0_buffer->address();
            reader_runtime_args[1] = src1_buffer->address();
            reader_runtime_args[2] = dst_buffer->address();
            reader_runtime_args[3] = start_id;
            auto n_sticks_per_core =
                i < core_group_1.num_cores() ? num_sticks_per_core_group_1 : num_sticks_per_core_group_2;
            reader_runtime_args[4] = n_sticks_per_core;

            auto& compute_runtime_args = GetRuntimeArgs(program, compute_kernel_id, cores[i]);
            compute_runtime_args[0] = n_sticks_per_core;

            auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id, cores[i]);
            writer_runtime_args[0] = src0_buffer->address();
            writer_runtime_args[1] = src1_buffer->address();
            writer_runtime_args[2] = dst_buffer->address();
            writer_runtime_args[3] = start_id;
            writer_runtime_args[4] = n_sticks_per_core;
            start_id += n_sticks_per_core;
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}
}  // namespace ttnn::operations::bos::bos_grid_sample
