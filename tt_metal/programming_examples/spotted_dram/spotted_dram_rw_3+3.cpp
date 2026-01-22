#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <matmul_common/bmm_op.hpp>
#include <algorithm>

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;

int main() {
    bool pass = true;

    /* Silicon accelerator setup */
    constexpr int device_id = 0;
    IDevice* device = CreateDevice(device_id);
    Program program{};
    CommandQueue& cq = device->command_queue();
    /* 입력 텐서 차원: (B, 1, HW, C) */

    /* Create source data */
    //  ([1, 1, 5 * 512 * 1024 // 20 // 32, 20 * 32], (5, 4)),

    constexpr uint32_t B = 1;      // user-defined
    constexpr uint32_t C = 640;    // user-defined
    constexpr uint32_t HW = 4096;  // user-defined
    constexpr uint32_t K = 1;      // user-defined
    uint32_t num_cores = 20;
    constexpr uint32_t single_tile_size = 4 * 1024;
    uint32_t num_tiles = HW * C / num_cores / 1024;
    uint32_t dram_buffer_A_size = 4 * B * C * HW;  // num_tiles of FP16_B
    /* input vectors */
    std::vector<uint32_t> src0_vec = std::vector<uint32_t>(B * HW * C, 0);
    std::iota(src0_vec.begin(), src0_vec.end(), 0);  // 0,1,2,3,4,5,6...
    tt::tt_metal::InterleavedBufferConfig dram_config_A{
        .device = device,
        .size = dram_buffer_A_size,
        .page_size = single_tile_size,
        .buffer_type = tt::tt_metal::BufferType::DRAM};

    /* Input vector tilizing */
    // tensor = (B,1,HW,C);
    src0_vec = tilize_nfaces(src0_vec, HW, C);

    auto src0_dram_buffer = CreateBuffer(dram_config_A);
    auto result_dram_buffer = CreateBuffer(dram_config_A);
    bool src0_is_dram = src0_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM;

    // output i don't care about outputs, but I need to make output buffers to use global circular buffer
    auto shard_spec = ShardSpecBuffer(
        CoreRangeSet(std::set<CoreRange>({CoreRange(CoreCoord(0, 0), CoreCoord(4, 3))})),
        {HW, C / num_cores},
        ShardOrientation::ROW_MAJOR,
        {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
        {HW, C});

    log_info(tt::LogTest, "page_shape : {}", shard_spec.page_shape);
    log_info(tt::LogTest, "grid : {}", shard_spec.grid());
    auto output_buffer = CreateBuffer(tt::tt_metal::ShardedBufferConfig{
        .device = device,
        .size = dram_buffer_A_size,
        .page_size = single_tile_size,
        .buffer_type = tt::tt_metal::BufferType::L1,
        .buffer_layout = TensorMemoryLayout::WIDTH_SHARDED,
        .shard_parameters = shard_spec});

    // now we have to make spotted interleaved2shard ops
    // {0 ,2, 4} => noc0 {1,3,5} =>noc1
    // we need 4 kernel ids
    // odd_reader_id, even_reader_id / odd_writer_id , even_writer_id

    tt::tt_metal::KernelHandle reader_kernel_id;
    tt::tt_metal::KernelHandle writer_kernel_id;

    CoreRange all_cores({(std::size_t)0, (std::size_t)0}, {(std::size_t)4, (std::size_t)3});

    uint32_t input_cb_idx = 0;
    uint32_t output_cd_idx = 0;

    tt::tt_metal::CircularBufferConfig output_cb_out_config =
        tt::tt_metal::CircularBufferConfig(num_tiles * single_tile_size, {{output_cd_idx, tt::DataFormat::UInt32}})
            .set_page_size(output_cd_idx, single_tile_size);

    output_cb_out_config = output_cb_out_config.set_globally_allocated_address(*output_buffer);

    auto cb_output = tt::tt_metal::CreateCircularBuffer(program, all_cores, output_cb_out_config);

    std::vector<uint32_t> reader_compile_time_args = {(std::uint32_t)input_cb_idx, (std::uint32_t)true, 20};

    reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/spotted_dram/kernel/reader_readback.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    std::vector<uint32_t> writer_compile_time_args = {output_cd_idx};

    writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/spotted_dram/kernel/writer_readback.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    for (uint32_t y = 0; y < 4; y++) {
        for (uint32_t x = 0; x < 5; x++) {
            std::vector<uint32_t> reader_run_time_args = {
                src0_dram_buffer->address(),
                num_tiles,
                1,
                0,
                20,
                num_tiles,
                x + y * 5,
                0,
            };

            std::vector<uint32_t> writer_run_time_args = {
                result_dram_buffer->address(),
                num_tiles,
                1,
                0,
                20,
                num_tiles,
                x + y * 5,
                0,
            };
            tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, CoreCoord(x, y), reader_run_time_args);
            tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, CoreCoord(x, y), writer_run_time_args);
        }
    }
    EnqueueWriteBuffer(cq, src0_dram_buffer, src0_vec.data(), false);
    EnqueueProgram(cq, program, false);
    tt_metal::detail::DumpDeviceProfileResults(device);
    CloseDevice(device);
    return 0;
}
