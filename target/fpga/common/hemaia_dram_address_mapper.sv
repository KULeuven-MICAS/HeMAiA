// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

module hemaia_dram_address_mapper #(
  // DDRMC 0 is split at 2 GiB to leave the Versal MMIO hole free.
  parameter logic [47:0] NocDramBase0Low  = 48'h0000_0000_0000,
  parameter logic [47:0] NocDramBase0High = 48'h0008_0000_0000,
  parameter logic [47:0] NocDramBase1     = 48'h0500_0000_0000,
  parameter logic [47:0] NocDramBase2     = 48'h0600_0000_0000,
  // Reserved in the NoC system address map (PG313).
  parameter logic [47:0] NocUnmappedBase = 48'h0010_0000_0000
) (
  input  logic [ 7:0] chip_id_i,
  input  logic [47:0] aw_addr_i,
  input  logic [47:0] ar_addr_i,
  output logic [47:0] aw_addr_o,
  output logic [47:0] ar_addr_o
);
  function automatic logic [47:0] map_address(
    input logic [47:0] addr,
    input logic [ 7:0] chip_id
  );
    // Each source region starts on the same alignment as its NoC destination.
    // Retaining the byte offset preserves alignment for all legal AXI bursts.
    if (addr[47:40] == chip_id) begin
      case (addr[39:32])
        8'h20: return {addr[31] ? NocDramBase0High[47:31] : NocDramBase0Low[47:31], addr[30:0]};
        8'h21: return {NocDramBase1[47:32], addr[31:0]};
        8'h22: return {NocDramBase2[47:32], addr[31:0]};
        default: ;
      endcase
    end
    // Preserve the offset inside the 4 KiB page even for a NoC decode miss.
    return {NocUnmappedBase[47:12], addr[11:0]};
  endfunction

  assign aw_addr_o = map_address(aw_addr_i, chip_id_i);
  assign ar_addr_o = map_address(ar_addr_i, chip_id_i);

  // pragma translate_off
  initial begin
    assert (NocDramBase0Low[30:0] == '0 && NocDramBase0High[30:0] == '0 &&
            NocDramBase1[31:0] == '0 && NocDramBase2[31:0] == '0 && NocUnmappedBase[11:0] == '0)
      else $fatal(1, "The NoC address bases must be aligned to their respective region sizes");
  end
  // pragma translate_on
endmodule
