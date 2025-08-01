// Copyright lowRISC contributors.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Register Package auto-generated by `reggen` containing data structure

package hemaia_clk_rst_controller_reg_pkg;

  // Address widths within the block
  parameter int BlockAw = 6;

  ////////////////////////////
  // Typedefs for registers //
  ////////////////////////////

  typedef struct packed {
    struct packed {
      logic        q;
    } reset_c0;
    struct packed {
      logic        q;
    } reset_c1;
    struct packed {
      logic        q;
    } reset_c2;
    struct packed {
      logic        q;
    } reset_c3;
    struct packed {
      logic        q;
    } reset_c4;
    struct packed {
      logic        q;
    } reset_c5;
    struct packed {
      logic        q;
    } reset_c6;
    struct packed {
      logic        q;
    } reset_c7;
    struct packed {
      logic        q;
    } reset_c8;
    struct packed {
      logic        q;
    } reset_c9;
    struct packed {
      logic        q;
    } reset_c10;
    struct packed {
      logic        q;
    } reset_c11;
    struct packed {
      logic        q;
    } reset_c12;
    struct packed {
      logic        q;
    } reset_c13;
    struct packed {
      logic        q;
    } reset_c14;
    struct packed {
      logic        q;
    } reset_c15;
    struct packed {
      logic        q;
    } reset_c16;
    struct packed {
      logic        q;
    } reset_c17;
    struct packed {
      logic        q;
    } reset_c18;
    struct packed {
      logic        q;
    } reset_c19;
    struct packed {
      logic        q;
    } reset_c20;
    struct packed {
      logic        q;
    } reset_c21;
    struct packed {
      logic        q;
    } reset_c22;
    struct packed {
      logic        q;
    } reset_c23;
    struct packed {
      logic        q;
    } reset_c24;
    struct packed {
      logic        q;
    } reset_c25;
    struct packed {
      logic        q;
    } reset_c26;
    struct packed {
      logic        q;
    } reset_c27;
    struct packed {
      logic        q;
    } reset_c28;
    struct packed {
      logic        q;
    } reset_c29;
    struct packed {
      logic        q;
    } reset_c30;
    struct packed {
      logic        q;
    } reset_c31;
  } hemaia_clk_rst_controller_reg2hw_reset_register_reg_t;

  typedef struct packed {
    struct packed {
      logic        q;
    } valid_c0;
    struct packed {
      logic        q;
    } valid_c1;
    struct packed {
      logic        q;
    } valid_c2;
    struct packed {
      logic        q;
    } valid_c3;
    struct packed {
      logic        q;
    } valid_c4;
    struct packed {
      logic        q;
    } valid_c5;
    struct packed {
      logic        q;
    } valid_c6;
    struct packed {
      logic        q;
    } valid_c7;
    struct packed {
      logic        q;
    } valid_c8;
    struct packed {
      logic        q;
    } valid_c9;
    struct packed {
      logic        q;
    } valid_c10;
    struct packed {
      logic        q;
    } valid_c11;
    struct packed {
      logic        q;
    } valid_c12;
    struct packed {
      logic        q;
    } valid_c13;
    struct packed {
      logic        q;
    } valid_c14;
    struct packed {
      logic        q;
    } valid_c15;
    struct packed {
      logic        q;
    } valid_c16;
    struct packed {
      logic        q;
    } valid_c17;
    struct packed {
      logic        q;
    } valid_c18;
    struct packed {
      logic        q;
    } valid_c19;
    struct packed {
      logic        q;
    } valid_c20;
    struct packed {
      logic        q;
    } valid_c21;
    struct packed {
      logic        q;
    } valid_c22;
    struct packed {
      logic        q;
    } valid_c23;
    struct packed {
      logic        q;
    } valid_c24;
    struct packed {
      logic        q;
    } valid_c25;
    struct packed {
      logic        q;
    } valid_c26;
    struct packed {
      logic        q;
    } valid_c27;
    struct packed {
      logic        q;
    } valid_c28;
    struct packed {
      logic        q;
    } valid_c29;
    struct packed {
      logic        q;
    } valid_c30;
    struct packed {
      logic        q;
    } valid_c31;
  } hemaia_clk_rst_controller_reg2hw_clock_valid_register_reg_t;

  typedef struct packed {
    struct packed {
      logic [7:0]  q;
    } division_c0;
    struct packed {
      logic [7:0]  q;
    } division_c1;
    struct packed {
      logic [7:0]  q;
    } division_c2;
    struct packed {
      logic [7:0]  q;
    } division_c3;
  } hemaia_clk_rst_controller_reg2hw_clock_division_register_c0_c3_reg_t;

  typedef struct packed {
    struct packed {
      logic [7:0]  q;
    } division_c4;
    struct packed {
      logic [7:0]  q;
    } division_c5;
    struct packed {
      logic [7:0]  q;
    } division_c6;
    struct packed {
      logic [7:0]  q;
    } division_c7;
  } hemaia_clk_rst_controller_reg2hw_clock_division_register_c4_c7_reg_t;

  typedef struct packed {
    struct packed {
      logic [7:0]  q;
    } division_c8;
    struct packed {
      logic [7:0]  q;
    } division_c9;
    struct packed {
      logic [7:0]  q;
    } division_c10;
    struct packed {
      logic [7:0]  q;
    } division_c11;
  } hemaia_clk_rst_controller_reg2hw_clock_division_register_c8_c11_reg_t;

  typedef struct packed {
    struct packed {
      logic [7:0]  q;
    } division_c12;
    struct packed {
      logic [7:0]  q;
    } division_c13;
    struct packed {
      logic [7:0]  q;
    } division_c14;
    struct packed {
      logic [7:0]  q;
    } division_c15;
  } hemaia_clk_rst_controller_reg2hw_clock_division_register_c12_c15_reg_t;

  typedef struct packed {
    struct packed {
      logic [7:0]  q;
    } division_c16;
    struct packed {
      logic [7:0]  q;
    } division_c17;
    struct packed {
      logic [7:0]  q;
    } division_c18;
    struct packed {
      logic [7:0]  q;
    } division_c19;
  } hemaia_clk_rst_controller_reg2hw_clock_division_register_c16_c19_reg_t;

  typedef struct packed {
    struct packed {
      logic [7:0]  q;
    } division_c20;
    struct packed {
      logic [7:0]  q;
    } division_c21;
    struct packed {
      logic [7:0]  q;
    } division_c22;
    struct packed {
      logic [7:0]  q;
    } division_c23;
  } hemaia_clk_rst_controller_reg2hw_clock_division_register_c20_c23_reg_t;

  typedef struct packed {
    struct packed {
      logic [7:0]  q;
    } division_c24;
    struct packed {
      logic [7:0]  q;
    } division_c25;
    struct packed {
      logic [7:0]  q;
    } division_c26;
    struct packed {
      logic [7:0]  q;
    } division_c27;
  } hemaia_clk_rst_controller_reg2hw_clock_division_register_c24_c27_reg_t;

  typedef struct packed {
    struct packed {
      logic [7:0]  q;
    } division_c28;
    struct packed {
      logic [7:0]  q;
    } division_c29;
    struct packed {
      logic [7:0]  q;
    } division_c30;
    struct packed {
      logic [7:0]  q;
    } division_c31;
  } hemaia_clk_rst_controller_reg2hw_clock_division_register_c28_c31_reg_t;

  typedef struct packed {
    struct packed {
      logic        d;
      logic        de;
    } reset_c0;
    struct packed {
      logic        d;
      logic        de;
    } reset_c1;
    struct packed {
      logic        d;
      logic        de;
    } reset_c2;
    struct packed {
      logic        d;
      logic        de;
    } reset_c3;
    struct packed {
      logic        d;
      logic        de;
    } reset_c4;
    struct packed {
      logic        d;
      logic        de;
    } reset_c5;
    struct packed {
      logic        d;
      logic        de;
    } reset_c6;
    struct packed {
      logic        d;
      logic        de;
    } reset_c7;
    struct packed {
      logic        d;
      logic        de;
    } reset_c8;
    struct packed {
      logic        d;
      logic        de;
    } reset_c9;
    struct packed {
      logic        d;
      logic        de;
    } reset_c10;
    struct packed {
      logic        d;
      logic        de;
    } reset_c11;
    struct packed {
      logic        d;
      logic        de;
    } reset_c12;
    struct packed {
      logic        d;
      logic        de;
    } reset_c13;
    struct packed {
      logic        d;
      logic        de;
    } reset_c14;
    struct packed {
      logic        d;
      logic        de;
    } reset_c15;
    struct packed {
      logic        d;
      logic        de;
    } reset_c16;
    struct packed {
      logic        d;
      logic        de;
    } reset_c17;
    struct packed {
      logic        d;
      logic        de;
    } reset_c18;
    struct packed {
      logic        d;
      logic        de;
    } reset_c19;
    struct packed {
      logic        d;
      logic        de;
    } reset_c20;
    struct packed {
      logic        d;
      logic        de;
    } reset_c21;
    struct packed {
      logic        d;
      logic        de;
    } reset_c22;
    struct packed {
      logic        d;
      logic        de;
    } reset_c23;
    struct packed {
      logic        d;
      logic        de;
    } reset_c24;
    struct packed {
      logic        d;
      logic        de;
    } reset_c25;
    struct packed {
      logic        d;
      logic        de;
    } reset_c26;
    struct packed {
      logic        d;
      logic        de;
    } reset_c27;
    struct packed {
      logic        d;
      logic        de;
    } reset_c28;
    struct packed {
      logic        d;
      logic        de;
    } reset_c29;
    struct packed {
      logic        d;
      logic        de;
    } reset_c30;
    struct packed {
      logic        d;
      logic        de;
    } reset_c31;
  } hemaia_clk_rst_controller_hw2reg_reset_register_reg_t;

  typedef struct packed {
    struct packed {
      logic        d;
      logic        de;
    } valid_c0;
    struct packed {
      logic        d;
      logic        de;
    } valid_c1;
    struct packed {
      logic        d;
      logic        de;
    } valid_c2;
    struct packed {
      logic        d;
      logic        de;
    } valid_c3;
    struct packed {
      logic        d;
      logic        de;
    } valid_c4;
    struct packed {
      logic        d;
      logic        de;
    } valid_c5;
    struct packed {
      logic        d;
      logic        de;
    } valid_c6;
    struct packed {
      logic        d;
      logic        de;
    } valid_c7;
    struct packed {
      logic        d;
      logic        de;
    } valid_c8;
    struct packed {
      logic        d;
      logic        de;
    } valid_c9;
    struct packed {
      logic        d;
      logic        de;
    } valid_c10;
    struct packed {
      logic        d;
      logic        de;
    } valid_c11;
    struct packed {
      logic        d;
      logic        de;
    } valid_c12;
    struct packed {
      logic        d;
      logic        de;
    } valid_c13;
    struct packed {
      logic        d;
      logic        de;
    } valid_c14;
    struct packed {
      logic        d;
      logic        de;
    } valid_c15;
    struct packed {
      logic        d;
      logic        de;
    } valid_c16;
    struct packed {
      logic        d;
      logic        de;
    } valid_c17;
    struct packed {
      logic        d;
      logic        de;
    } valid_c18;
    struct packed {
      logic        d;
      logic        de;
    } valid_c19;
    struct packed {
      logic        d;
      logic        de;
    } valid_c20;
    struct packed {
      logic        d;
      logic        de;
    } valid_c21;
    struct packed {
      logic        d;
      logic        de;
    } valid_c22;
    struct packed {
      logic        d;
      logic        de;
    } valid_c23;
    struct packed {
      logic        d;
      logic        de;
    } valid_c24;
    struct packed {
      logic        d;
      logic        de;
    } valid_c25;
    struct packed {
      logic        d;
      logic        de;
    } valid_c26;
    struct packed {
      logic        d;
      logic        de;
    } valid_c27;
    struct packed {
      logic        d;
      logic        de;
    } valid_c28;
    struct packed {
      logic        d;
      logic        de;
    } valid_c29;
    struct packed {
      logic        d;
      logic        de;
    } valid_c30;
    struct packed {
      logic        d;
      logic        de;
    } valid_c31;
  } hemaia_clk_rst_controller_hw2reg_clock_valid_register_reg_t;

  // Register -> HW type
  typedef struct packed {
    hemaia_clk_rst_controller_reg2hw_reset_register_reg_t reset_register; // [319:288]
    hemaia_clk_rst_controller_reg2hw_clock_valid_register_reg_t clock_valid_register; // [287:256]
    hemaia_clk_rst_controller_reg2hw_clock_division_register_c0_c3_reg_t clock_division_register_c0_c3; // [255:224]
    hemaia_clk_rst_controller_reg2hw_clock_division_register_c4_c7_reg_t clock_division_register_c4_c7; // [223:192]
    hemaia_clk_rst_controller_reg2hw_clock_division_register_c8_c11_reg_t clock_division_register_c8_c11; // [191:160]
    hemaia_clk_rst_controller_reg2hw_clock_division_register_c12_c15_reg_t clock_division_register_c12_c15; // [159:128]
    hemaia_clk_rst_controller_reg2hw_clock_division_register_c16_c19_reg_t clock_division_register_c16_c19; // [127:96]
    hemaia_clk_rst_controller_reg2hw_clock_division_register_c20_c23_reg_t clock_division_register_c20_c23; // [95:64]
    hemaia_clk_rst_controller_reg2hw_clock_division_register_c24_c27_reg_t clock_division_register_c24_c27; // [63:32]
    hemaia_clk_rst_controller_reg2hw_clock_division_register_c28_c31_reg_t clock_division_register_c28_c31; // [31:0]
  } hemaia_clk_rst_controller_reg2hw_t;

  // HW -> register type
  typedef struct packed {
    hemaia_clk_rst_controller_hw2reg_reset_register_reg_t reset_register; // [127:64]
    hemaia_clk_rst_controller_hw2reg_clock_valid_register_reg_t clock_valid_register; // [63:0]
  } hemaia_clk_rst_controller_hw2reg_t;

  // Register offsets
  parameter logic [BlockAw-1:0] HEMAIA_CLK_RST_CONTROLLER_RESET_REGISTER_OFFSET = 6'h 0;
  parameter logic [BlockAw-1:0] HEMAIA_CLK_RST_CONTROLLER_CLOCK_VALID_REGISTER_OFFSET = 6'h 4;
  parameter logic [BlockAw-1:0] HEMAIA_CLK_RST_CONTROLLER_CLOCK_DIVISION_REGISTER_C0_C3_OFFSET = 6'h 8;
  parameter logic [BlockAw-1:0] HEMAIA_CLK_RST_CONTROLLER_CLOCK_DIVISION_REGISTER_C4_C7_OFFSET = 6'h c;
  parameter logic [BlockAw-1:0] HEMAIA_CLK_RST_CONTROLLER_CLOCK_DIVISION_REGISTER_C8_C11_OFFSET = 6'h 10;
  parameter logic [BlockAw-1:0] HEMAIA_CLK_RST_CONTROLLER_CLOCK_DIVISION_REGISTER_C12_C15_OFFSET = 6'h 14;
  parameter logic [BlockAw-1:0] HEMAIA_CLK_RST_CONTROLLER_CLOCK_DIVISION_REGISTER_C16_C19_OFFSET = 6'h 18;
  parameter logic [BlockAw-1:0] HEMAIA_CLK_RST_CONTROLLER_CLOCK_DIVISION_REGISTER_C20_C23_OFFSET = 6'h 1c;
  parameter logic [BlockAw-1:0] HEMAIA_CLK_RST_CONTROLLER_CLOCK_DIVISION_REGISTER_C24_C27_OFFSET = 6'h 20;
  parameter logic [BlockAw-1:0] HEMAIA_CLK_RST_CONTROLLER_CLOCK_DIVISION_REGISTER_C28_C31_OFFSET = 6'h 24;

  // Register index
  typedef enum int {
    HEMAIA_CLK_RST_CONTROLLER_RESET_REGISTER,
    HEMAIA_CLK_RST_CONTROLLER_CLOCK_VALID_REGISTER,
    HEMAIA_CLK_RST_CONTROLLER_CLOCK_DIVISION_REGISTER_C0_C3,
    HEMAIA_CLK_RST_CONTROLLER_CLOCK_DIVISION_REGISTER_C4_C7,
    HEMAIA_CLK_RST_CONTROLLER_CLOCK_DIVISION_REGISTER_C8_C11,
    HEMAIA_CLK_RST_CONTROLLER_CLOCK_DIVISION_REGISTER_C12_C15,
    HEMAIA_CLK_RST_CONTROLLER_CLOCK_DIVISION_REGISTER_C16_C19,
    HEMAIA_CLK_RST_CONTROLLER_CLOCK_DIVISION_REGISTER_C20_C23,
    HEMAIA_CLK_RST_CONTROLLER_CLOCK_DIVISION_REGISTER_C24_C27,
    HEMAIA_CLK_RST_CONTROLLER_CLOCK_DIVISION_REGISTER_C28_C31
  } hemaia_clk_rst_controller_id_e;

  // Register width information to check illegal writes
  parameter logic [3:0] HEMAIA_CLK_RST_CONTROLLER_PERMIT [10] = '{
    4'b 1111, // index[0] HEMAIA_CLK_RST_CONTROLLER_RESET_REGISTER
    4'b 1111, // index[1] HEMAIA_CLK_RST_CONTROLLER_CLOCK_VALID_REGISTER
    4'b 1111, // index[2] HEMAIA_CLK_RST_CONTROLLER_CLOCK_DIVISION_REGISTER_C0_C3
    4'b 1111, // index[3] HEMAIA_CLK_RST_CONTROLLER_CLOCK_DIVISION_REGISTER_C4_C7
    4'b 1111, // index[4] HEMAIA_CLK_RST_CONTROLLER_CLOCK_DIVISION_REGISTER_C8_C11
    4'b 1111, // index[5] HEMAIA_CLK_RST_CONTROLLER_CLOCK_DIVISION_REGISTER_C12_C15
    4'b 1111, // index[6] HEMAIA_CLK_RST_CONTROLLER_CLOCK_DIVISION_REGISTER_C16_C19
    4'b 1111, // index[7] HEMAIA_CLK_RST_CONTROLLER_CLOCK_DIVISION_REGISTER_C20_C23
    4'b 1111, // index[8] HEMAIA_CLK_RST_CONTROLLER_CLOCK_DIVISION_REGISTER_C24_C27
    4'b 1111  // index[9] HEMAIA_CLK_RST_CONTROLLER_CLOCK_DIVISION_REGISTER_C28_C31
  };

endpackage

