// Copyright 2024 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
// Yunhao Deng <yunhao.deng@kuleuven.be>

module half_duplex_bus_emulator #(
    parameter BusWidth = 8,
    parameter ChannelNum = 2
)(
    input logic port1_tx_mode_i,
    input logic port2_tx_mode_i,
    input logic [BusWidth-1:0] port1_i[ChannelNum],
    input logic [BusWidth-1:0] port2_i[ChannelNum],
    output logic[BusWidth-1:0] port1_o [ChannelNum],
    output logic[BusWidth-1:0] port2_o [ChannelNum]
);
    tri [BusWidth-1:0] bus [ChannelNum];

    assign bus = port1_tx_mode_i ? port1_i : '{default: 'z};
    assign bus = port2_tx_mode_i ? port2_i : '{default: 'z};

    assign port1_o = bus;
    assign port2_o = bus;
endmodule
