// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

void kernel_main() {
    std::uint32_t l1_buffer_addr0   = get_arg_val<uint32_t>(0);
    std::uint32_t l1_buffer_addr1   = get_arg_val<uint32_t>(1);
    std::uint32_t host_addr         = get_arg_val<uint32_t>(2);
    std::uint32_t data_size         = get_arg_val<uint32_t>(3);
    std::uint32_t wr_data           = get_arg_val<uint32_t>(4);

    std::uint64_t dram_addr         = (uint64_t)host_addr + 0x2100000000;

    //uint64_t pcie_noc_xy_encoding = (uint64_t)NOC_XY_PCIE_ENCODING(0, 3, 0);
    //uint64_t host_src_addr = pcie_noc_xy_encoding;
    uint64_t host_src_addr = get_noc_addr_64(5, 3, dram_addr);

    DPRINT << "DRAM addr + base: " << dram_addr << ENDL();
    DPRINT << "Get_noc_addr_64: "  << host_src_addr << ENDL();
    DPRINT << "DATA size: " <<DEC() << data_size << ENDL();

    uint32_t* ptr0 = (uint32_t*)l1_buffer_addr0;
    uint32_t* ptr1 = (uint32_t*)l1_buffer_addr1;

    for (uint32_t id = 0; id < (data_size/sizeof(uint32_t)); id++){
        ptr0[id] = wr_data;
    }

    noc_async_write(l1_buffer_addr0, host_src_addr, data_size);
    noc_async_write_barrier();
    DPRINT << "WRITE host's DRAM -> Done " << ENDL();

    noc_async_read(host_src_addr, l1_buffer_addr1, data_size);
    noc_async_read_barrier();
    DPRINT << "READ host's DRAM -> Done" << ENDL();

    bool pass=true;

    for (uint32_t id = 0; id < (data_size/sizeof(uint32_t)); id++){
        if(ptr0[id] != ptr1[id]) pass=false;
        // DPRINT << id << "-L0/L1 : " << ptr0[id] << "/" << ptr1[id] << ENDL();
    }

    if(pass){
        DPRINT << "Kernel Passes" << ENDL();
    }else{
        DPRINT << "Kernel Fail" << ENDL();
    }
}
