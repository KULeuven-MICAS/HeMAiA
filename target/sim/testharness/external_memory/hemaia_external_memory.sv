// Copyright 2026 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Simulation-only AXI4 slave with one outstanding burst per direction.
module hemaia_external_memory #(
    parameter int unsigned AddrWidth = 48,
    parameter int unsigned ChipIdWidth = 8,
    parameter longint unsigned BaseAddr = 0,
    parameter longint unsigned SizeBytes = 0,
    parameter type aw_chan_t = logic,
    parameter type ar_chan_t = logic,
    parameter type b_chan_t = logic,
    parameter type r_chan_t = logic,
    parameter type axi_req_t = logic,
    parameter type axi_rsp_t = logic
) (
    input logic clk_i,
    input logic rst_ni,
    input logic [ChipIdWidth-1:0] chip_id_i,
    input axi_req_t axi_req_i,
    output axi_rsp_t axi_rsp_o
);
    import "DPI-C" function chandle hemaia_extmem_create(input longint unsigned size_bytes);
    import "DPI-C" function void hemaia_extmem_destroy(input chandle mem);
    import "DPI-C" function int hemaia_extmem_read(
        input chandle mem, input longint unsigned offset, output bit [511:0] data
    );
    import "DPI-C" function int hemaia_extmem_write(
        input chandle mem, input longint unsigned offset, input bit [511:0] data,
        input longint unsigned strb
    );

    typedef longint unsigned addr_t;
    localparam addr_t WordMask = 64'd63;
    localparam addr_t StorageBase = BaseAddr & ~WordMask;
    localparam addr_t StorageBytes = (BaseAddr - StorageBase + SizeBytes + WordMask) & ~WordMask;

    chandle memory;
    initial begin
        memory = hemaia_extmem_create(StorageBytes);
        if (memory == null)
            $fatal(1, "Unable to create external AXI memory");
    end

    final hemaia_extmem_destroy(memory);

    typedef struct packed {
        bit valid;
        addr_t offset;
        logic [63:0] lanes;
    } beat_t;

    function automatic beat_t decode_beat(
        input logic [AddrWidth-1:0] address,
        input axi_pkg::size_t size,
        input axi_pkg::len_t len,
        input axi_pkg::burst_t burst,
        input axi_pkg::len_t beat_index
    );
        addr_t first_byte, last_byte, word_addr, chip_base;
        beat_t beat = '0;

        if (size > axi_pkg::size_t'(6))
            return beat;
        case (burst)
            axi_pkg::BURST_FIXED, axi_pkg::BURST_INCR: ;
            axi_pkg::BURST_WRAP: begin
                if (!(len inside {8'd1, 8'd3, 8'd7, 8'd15}) ||
                    axi_pkg::largest_addr_t'(address) != axi_pkg::aligned_addr(address, size))
                    return beat;
            end
            default: return beat;
        endcase

        first_byte = addr_t'(axi_pkg::beat_addr(
            address, size, len, burst, shortint'(beat_index)
        ));
        last_byte = addr_t'(axi_pkg::aligned_addr(first_byte, size)) +
                    addr_t'(axi_pkg::num_bytes(size)) - 1;
        word_addr = first_byte & ~WordMask;
        chip_base = addr_t'(chip_id_i) << (AddrWidth - ChipIdWidth);
        if (first_byte < chip_base + BaseAddr || last_byte >= chip_base + BaseAddr + SizeBytes ||
            (first_byte >> 12) != (addr_t'(address) >> 12))
            return beat;

        beat.valid = 1'b1;
        beat.offset = word_addr - (chip_base + StorageBase);
        beat.lanes = (64'hffff_ffff_ffff_ffff << first_byte[5:0]) &
                     (64'hffff_ffff_ffff_ffff >> (6'd63 - last_byte[5:0]));
        return beat;
    endfunction

    aw_chan_t aw_q;
    ar_chan_t ar_q;
    b_chan_t b_q;
    r_chan_t r_q;
    axi_pkg::len_t read_index_q, write_index_q;
    axi_pkg::resp_t write_resp_q;
    logic read_active_q, write_active_q, b_valid_q, r_valid_q;

    always_comb begin
        axi_rsp_o = '0;
        axi_rsp_o.aw_ready = rst_ni && !write_active_q && !b_valid_q;
        axi_rsp_o.w_ready = rst_ni && write_active_q;
        axi_rsp_o.ar_ready = rst_ni && !read_active_q;
        axi_rsp_o.b_valid = b_valid_q;
        axi_rsp_o.b = b_q;
        axi_rsp_o.r_valid = r_valid_q;
        axi_rsp_o.r = r_q;
    end

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            aw_q <= '0;
            ar_q <= '0;
            b_q <= '0;
            r_q <= '0;
            write_index_q <= '0;
            read_index_q <= '0;
            write_resp_q <= axi_pkg::RESP_OKAY;
            read_active_q <= 1'b0;
            write_active_q <= 1'b0;
            b_valid_q <= 1'b0;
            r_valid_q <= 1'b0;
        end else begin
            if (axi_req_i.aw_valid && axi_rsp_o.aw_ready) begin
                aw_q <= axi_req_i.aw;
                write_active_q <= 1'b1;
                write_index_q <= '0;
                write_resp_q <= axi_pkg::RESP_OKAY;
            end

            if (b_valid_q && axi_req_i.b_ready)
                b_valid_q <= 1'b0;

            if (axi_req_i.w_valid && axi_rsp_o.w_ready) begin
                beat_t beat;
                axi_pkg::resp_t response;
                bit last;
                beat = decode_beat(aw_q.addr, aw_q.size, aw_q.len, aw_q.burst, write_index_q);
                last = write_index_q == aw_q.len;
                response = axi_pkg::RESP_OKAY;
                if (!beat.valid) begin
                    response = axi_pkg::RESP_DECERR;
                end else if ((axi_req_i.w.strb & ~beat.lanes) != '0 || axi_req_i.w.last != last) begin
                    response = axi_pkg::RESP_SLVERR;
                end else if (hemaia_extmem_write(
                    memory, beat.offset, axi_req_i.w.data, axi_req_i.w.strb
                ) != 0) begin
                    $fatal(1, "External AXI memory write failed at offset %h", beat.offset);
                end
                response = axi_pkg::resp_precedence(response, write_resp_q);
                write_resp_q <= response;
                if (last) begin
                    b_q <= '{id: aw_q.id, resp: response, user: '0};
                    b_valid_q <= 1'b1;
                    write_active_q <= 1'b0;
                end else begin
                    write_index_q <= write_index_q + 1'b1;
                end
            end

            if (axi_req_i.ar_valid && axi_rsp_o.ar_ready) begin
                ar_q <= axi_req_i.ar;
                read_active_q <= 1'b1;
                read_index_q <= '0;
            end

            if (r_valid_q) begin
                if (axi_req_i.r_ready) begin
                    r_valid_q <= 1'b0;
                    if (r_q.last)
                        read_active_q <= 1'b0;
                    else
                        read_index_q <= read_index_q + 1'b1;
                end
            end else if (read_active_q) begin
                beat_t beat;
                bit [511:0] data;
                beat = decode_beat(ar_q.addr, ar_q.size, ar_q.len, ar_q.burst, read_index_q);
                data = '0;
                if (beat.valid) begin
                    if (hemaia_extmem_read(memory, beat.offset, data) != 0)
                        $fatal(1, "External AXI memory read failed at offset %h", beat.offset);
                end
                r_q <= '{id: ar_q.id, data: data, last: read_index_q == ar_q.len,
                         resp: beat.valid ? axi_pkg::RESP_OKAY : axi_pkg::RESP_DECERR, user: '0};
                r_valid_q <= 1'b1;
            end
        end
    end
endmodule
