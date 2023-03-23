// Copyright lowRISC contributors.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Register Package auto-generated by `reggen` containing data structure

package axi_tlb_reg_pkg;

  // Address widths within the block
  parameter int BlockAw = 9;

  ////////////////////////////
  // Typedefs for registers //
  ////////////////////////////

  typedef struct packed {
    logic        q;
  } axi_tlb_reg2hw_tlb_enable_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } axi_tlb_reg2hw_tlb_entry_0_pagein_first_low_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } axi_tlb_reg2hw_tlb_entry_0_pagein_first_high_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } axi_tlb_reg2hw_tlb_entry_0_pagein_last_low_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } axi_tlb_reg2hw_tlb_entry_0_pagein_last_high_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } axi_tlb_reg2hw_tlb_entry_0_pageout_low_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } axi_tlb_reg2hw_tlb_entry_0_pageout_high_reg_t;

  typedef struct packed {
    struct packed {
      logic        q;
    } valid;
    struct packed {
      logic        q;
    } read_only;
  } axi_tlb_reg2hw_tlb_entry_0_flags_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } axi_tlb_reg2hw_tlb_entry_1_pagein_first_low_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } axi_tlb_reg2hw_tlb_entry_1_pagein_first_high_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } axi_tlb_reg2hw_tlb_entry_1_pagein_last_low_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } axi_tlb_reg2hw_tlb_entry_1_pagein_last_high_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } axi_tlb_reg2hw_tlb_entry_1_pageout_low_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } axi_tlb_reg2hw_tlb_entry_1_pageout_high_reg_t;

  typedef struct packed {
    struct packed {
      logic        q;
    } valid;
    struct packed {
      logic        q;
    } read_only;
  } axi_tlb_reg2hw_tlb_entry_1_flags_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } axi_tlb_reg2hw_tlb_entry_2_pagein_first_low_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } axi_tlb_reg2hw_tlb_entry_2_pagein_first_high_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } axi_tlb_reg2hw_tlb_entry_2_pagein_last_low_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } axi_tlb_reg2hw_tlb_entry_2_pagein_last_high_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } axi_tlb_reg2hw_tlb_entry_2_pageout_low_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } axi_tlb_reg2hw_tlb_entry_2_pageout_high_reg_t;

  typedef struct packed {
    struct packed {
      logic        q;
    } valid;
    struct packed {
      logic        q;
    } read_only;
  } axi_tlb_reg2hw_tlb_entry_2_flags_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } axi_tlb_reg2hw_tlb_entry_3_pagein_first_low_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } axi_tlb_reg2hw_tlb_entry_3_pagein_first_high_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } axi_tlb_reg2hw_tlb_entry_3_pagein_last_low_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } axi_tlb_reg2hw_tlb_entry_3_pagein_last_high_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } axi_tlb_reg2hw_tlb_entry_3_pageout_low_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } axi_tlb_reg2hw_tlb_entry_3_pageout_high_reg_t;

  typedef struct packed {
    struct packed {
      logic        q;
    } valid;
    struct packed {
      logic        q;
    } read_only;
  } axi_tlb_reg2hw_tlb_entry_3_flags_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } axi_tlb_reg2hw_tlb_entry_4_pagein_first_low_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } axi_tlb_reg2hw_tlb_entry_4_pagein_first_high_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } axi_tlb_reg2hw_tlb_entry_4_pagein_last_low_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } axi_tlb_reg2hw_tlb_entry_4_pagein_last_high_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } axi_tlb_reg2hw_tlb_entry_4_pageout_low_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } axi_tlb_reg2hw_tlb_entry_4_pageout_high_reg_t;

  typedef struct packed {
    struct packed {
      logic        q;
    } valid;
    struct packed {
      logic        q;
    } read_only;
  } axi_tlb_reg2hw_tlb_entry_4_flags_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } axi_tlb_reg2hw_tlb_entry_5_pagein_first_low_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } axi_tlb_reg2hw_tlb_entry_5_pagein_first_high_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } axi_tlb_reg2hw_tlb_entry_5_pagein_last_low_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } axi_tlb_reg2hw_tlb_entry_5_pagein_last_high_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } axi_tlb_reg2hw_tlb_entry_5_pageout_low_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } axi_tlb_reg2hw_tlb_entry_5_pageout_high_reg_t;

  typedef struct packed {
    struct packed {
      logic        q;
    } valid;
    struct packed {
      logic        q;
    } read_only;
  } axi_tlb_reg2hw_tlb_entry_5_flags_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } axi_tlb_reg2hw_tlb_entry_6_pagein_first_low_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } axi_tlb_reg2hw_tlb_entry_6_pagein_first_high_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } axi_tlb_reg2hw_tlb_entry_6_pagein_last_low_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } axi_tlb_reg2hw_tlb_entry_6_pagein_last_high_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } axi_tlb_reg2hw_tlb_entry_6_pageout_low_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } axi_tlb_reg2hw_tlb_entry_6_pageout_high_reg_t;

  typedef struct packed {
    struct packed {
      logic        q;
    } valid;
    struct packed {
      logic        q;
    } read_only;
  } axi_tlb_reg2hw_tlb_entry_6_flags_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } axi_tlb_reg2hw_tlb_entry_7_pagein_first_low_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } axi_tlb_reg2hw_tlb_entry_7_pagein_first_high_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } axi_tlb_reg2hw_tlb_entry_7_pagein_last_low_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } axi_tlb_reg2hw_tlb_entry_7_pagein_last_high_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } axi_tlb_reg2hw_tlb_entry_7_pageout_low_reg_t;

  typedef struct packed {
    logic [31:0] q;
  } axi_tlb_reg2hw_tlb_entry_7_pageout_high_reg_t;

  typedef struct packed {
    struct packed {
      logic        q;
    } valid;
    struct packed {
      logic        q;
    } read_only;
  } axi_tlb_reg2hw_tlb_entry_7_flags_reg_t;

  // Register -> HW type
  typedef struct packed {
    axi_tlb_reg2hw_tlb_enable_reg_t tlb_enable; // [1552:1552]
    axi_tlb_reg2hw_tlb_entry_0_pagein_first_low_reg_t tlb_entry_0_pagein_first_low; // [1551:1520]
    axi_tlb_reg2hw_tlb_entry_0_pagein_first_high_reg_t tlb_entry_0_pagein_first_high; // [1519:1488]
    axi_tlb_reg2hw_tlb_entry_0_pagein_last_low_reg_t tlb_entry_0_pagein_last_low; // [1487:1456]
    axi_tlb_reg2hw_tlb_entry_0_pagein_last_high_reg_t tlb_entry_0_pagein_last_high; // [1455:1424]
    axi_tlb_reg2hw_tlb_entry_0_pageout_low_reg_t tlb_entry_0_pageout_low; // [1423:1392]
    axi_tlb_reg2hw_tlb_entry_0_pageout_high_reg_t tlb_entry_0_pageout_high; // [1391:1360]
    axi_tlb_reg2hw_tlb_entry_0_flags_reg_t tlb_entry_0_flags; // [1359:1358]
    axi_tlb_reg2hw_tlb_entry_1_pagein_first_low_reg_t tlb_entry_1_pagein_first_low; // [1357:1326]
    axi_tlb_reg2hw_tlb_entry_1_pagein_first_high_reg_t tlb_entry_1_pagein_first_high; // [1325:1294]
    axi_tlb_reg2hw_tlb_entry_1_pagein_last_low_reg_t tlb_entry_1_pagein_last_low; // [1293:1262]
    axi_tlb_reg2hw_tlb_entry_1_pagein_last_high_reg_t tlb_entry_1_pagein_last_high; // [1261:1230]
    axi_tlb_reg2hw_tlb_entry_1_pageout_low_reg_t tlb_entry_1_pageout_low; // [1229:1198]
    axi_tlb_reg2hw_tlb_entry_1_pageout_high_reg_t tlb_entry_1_pageout_high; // [1197:1166]
    axi_tlb_reg2hw_tlb_entry_1_flags_reg_t tlb_entry_1_flags; // [1165:1164]
    axi_tlb_reg2hw_tlb_entry_2_pagein_first_low_reg_t tlb_entry_2_pagein_first_low; // [1163:1132]
    axi_tlb_reg2hw_tlb_entry_2_pagein_first_high_reg_t tlb_entry_2_pagein_first_high; // [1131:1100]
    axi_tlb_reg2hw_tlb_entry_2_pagein_last_low_reg_t tlb_entry_2_pagein_last_low; // [1099:1068]
    axi_tlb_reg2hw_tlb_entry_2_pagein_last_high_reg_t tlb_entry_2_pagein_last_high; // [1067:1036]
    axi_tlb_reg2hw_tlb_entry_2_pageout_low_reg_t tlb_entry_2_pageout_low; // [1035:1004]
    axi_tlb_reg2hw_tlb_entry_2_pageout_high_reg_t tlb_entry_2_pageout_high; // [1003:972]
    axi_tlb_reg2hw_tlb_entry_2_flags_reg_t tlb_entry_2_flags; // [971:970]
    axi_tlb_reg2hw_tlb_entry_3_pagein_first_low_reg_t tlb_entry_3_pagein_first_low; // [969:938]
    axi_tlb_reg2hw_tlb_entry_3_pagein_first_high_reg_t tlb_entry_3_pagein_first_high; // [937:906]
    axi_tlb_reg2hw_tlb_entry_3_pagein_last_low_reg_t tlb_entry_3_pagein_last_low; // [905:874]
    axi_tlb_reg2hw_tlb_entry_3_pagein_last_high_reg_t tlb_entry_3_pagein_last_high; // [873:842]
    axi_tlb_reg2hw_tlb_entry_3_pageout_low_reg_t tlb_entry_3_pageout_low; // [841:810]
    axi_tlb_reg2hw_tlb_entry_3_pageout_high_reg_t tlb_entry_3_pageout_high; // [809:778]
    axi_tlb_reg2hw_tlb_entry_3_flags_reg_t tlb_entry_3_flags; // [777:776]
    axi_tlb_reg2hw_tlb_entry_4_pagein_first_low_reg_t tlb_entry_4_pagein_first_low; // [775:744]
    axi_tlb_reg2hw_tlb_entry_4_pagein_first_high_reg_t tlb_entry_4_pagein_first_high; // [743:712]
    axi_tlb_reg2hw_tlb_entry_4_pagein_last_low_reg_t tlb_entry_4_pagein_last_low; // [711:680]
    axi_tlb_reg2hw_tlb_entry_4_pagein_last_high_reg_t tlb_entry_4_pagein_last_high; // [679:648]
    axi_tlb_reg2hw_tlb_entry_4_pageout_low_reg_t tlb_entry_4_pageout_low; // [647:616]
    axi_tlb_reg2hw_tlb_entry_4_pageout_high_reg_t tlb_entry_4_pageout_high; // [615:584]
    axi_tlb_reg2hw_tlb_entry_4_flags_reg_t tlb_entry_4_flags; // [583:582]
    axi_tlb_reg2hw_tlb_entry_5_pagein_first_low_reg_t tlb_entry_5_pagein_first_low; // [581:550]
    axi_tlb_reg2hw_tlb_entry_5_pagein_first_high_reg_t tlb_entry_5_pagein_first_high; // [549:518]
    axi_tlb_reg2hw_tlb_entry_5_pagein_last_low_reg_t tlb_entry_5_pagein_last_low; // [517:486]
    axi_tlb_reg2hw_tlb_entry_5_pagein_last_high_reg_t tlb_entry_5_pagein_last_high; // [485:454]
    axi_tlb_reg2hw_tlb_entry_5_pageout_low_reg_t tlb_entry_5_pageout_low; // [453:422]
    axi_tlb_reg2hw_tlb_entry_5_pageout_high_reg_t tlb_entry_5_pageout_high; // [421:390]
    axi_tlb_reg2hw_tlb_entry_5_flags_reg_t tlb_entry_5_flags; // [389:388]
    axi_tlb_reg2hw_tlb_entry_6_pagein_first_low_reg_t tlb_entry_6_pagein_first_low; // [387:356]
    axi_tlb_reg2hw_tlb_entry_6_pagein_first_high_reg_t tlb_entry_6_pagein_first_high; // [355:324]
    axi_tlb_reg2hw_tlb_entry_6_pagein_last_low_reg_t tlb_entry_6_pagein_last_low; // [323:292]
    axi_tlb_reg2hw_tlb_entry_6_pagein_last_high_reg_t tlb_entry_6_pagein_last_high; // [291:260]
    axi_tlb_reg2hw_tlb_entry_6_pageout_low_reg_t tlb_entry_6_pageout_low; // [259:228]
    axi_tlb_reg2hw_tlb_entry_6_pageout_high_reg_t tlb_entry_6_pageout_high; // [227:196]
    axi_tlb_reg2hw_tlb_entry_6_flags_reg_t tlb_entry_6_flags; // [195:194]
    axi_tlb_reg2hw_tlb_entry_7_pagein_first_low_reg_t tlb_entry_7_pagein_first_low; // [193:162]
    axi_tlb_reg2hw_tlb_entry_7_pagein_first_high_reg_t tlb_entry_7_pagein_first_high; // [161:130]
    axi_tlb_reg2hw_tlb_entry_7_pagein_last_low_reg_t tlb_entry_7_pagein_last_low; // [129:98]
    axi_tlb_reg2hw_tlb_entry_7_pagein_last_high_reg_t tlb_entry_7_pagein_last_high; // [97:66]
    axi_tlb_reg2hw_tlb_entry_7_pageout_low_reg_t tlb_entry_7_pageout_low; // [65:34]
    axi_tlb_reg2hw_tlb_entry_7_pageout_high_reg_t tlb_entry_7_pageout_high; // [33:2]
    axi_tlb_reg2hw_tlb_entry_7_flags_reg_t tlb_entry_7_flags; // [1:0]
  } axi_tlb_reg2hw_t;

  // Register offsets
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENABLE_OFFSET = 9'h 0;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_0_PAGEIN_FIRST_LOW_OFFSET = 9'h 20;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_0_PAGEIN_FIRST_HIGH_OFFSET = 9'h 24;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_0_PAGEIN_LAST_LOW_OFFSET = 9'h 28;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_0_PAGEIN_LAST_HIGH_OFFSET = 9'h 2c;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_0_PAGEOUT_LOW_OFFSET = 9'h 30;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_0_PAGEOUT_HIGH_OFFSET = 9'h 34;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_0_FLAGS_OFFSET = 9'h 38;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_1_PAGEIN_FIRST_LOW_OFFSET = 9'h 40;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_1_PAGEIN_FIRST_HIGH_OFFSET = 9'h 44;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_1_PAGEIN_LAST_LOW_OFFSET = 9'h 48;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_1_PAGEIN_LAST_HIGH_OFFSET = 9'h 4c;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_1_PAGEOUT_LOW_OFFSET = 9'h 50;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_1_PAGEOUT_HIGH_OFFSET = 9'h 54;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_1_FLAGS_OFFSET = 9'h 58;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_2_PAGEIN_FIRST_LOW_OFFSET = 9'h 60;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_2_PAGEIN_FIRST_HIGH_OFFSET = 9'h 64;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_2_PAGEIN_LAST_LOW_OFFSET = 9'h 68;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_2_PAGEIN_LAST_HIGH_OFFSET = 9'h 6c;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_2_PAGEOUT_LOW_OFFSET = 9'h 70;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_2_PAGEOUT_HIGH_OFFSET = 9'h 74;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_2_FLAGS_OFFSET = 9'h 78;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_3_PAGEIN_FIRST_LOW_OFFSET = 9'h 80;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_3_PAGEIN_FIRST_HIGH_OFFSET = 9'h 84;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_3_PAGEIN_LAST_LOW_OFFSET = 9'h 88;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_3_PAGEIN_LAST_HIGH_OFFSET = 9'h 8c;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_3_PAGEOUT_LOW_OFFSET = 9'h 90;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_3_PAGEOUT_HIGH_OFFSET = 9'h 94;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_3_FLAGS_OFFSET = 9'h 98;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_4_PAGEIN_FIRST_LOW_OFFSET = 9'h a0;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_4_PAGEIN_FIRST_HIGH_OFFSET = 9'h a4;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_4_PAGEIN_LAST_LOW_OFFSET = 9'h a8;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_4_PAGEIN_LAST_HIGH_OFFSET = 9'h ac;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_4_PAGEOUT_LOW_OFFSET = 9'h b0;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_4_PAGEOUT_HIGH_OFFSET = 9'h b4;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_4_FLAGS_OFFSET = 9'h b8;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_5_PAGEIN_FIRST_LOW_OFFSET = 9'h c0;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_5_PAGEIN_FIRST_HIGH_OFFSET = 9'h c4;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_5_PAGEIN_LAST_LOW_OFFSET = 9'h c8;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_5_PAGEIN_LAST_HIGH_OFFSET = 9'h cc;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_5_PAGEOUT_LOW_OFFSET = 9'h d0;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_5_PAGEOUT_HIGH_OFFSET = 9'h d4;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_5_FLAGS_OFFSET = 9'h d8;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_6_PAGEIN_FIRST_LOW_OFFSET = 9'h e0;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_6_PAGEIN_FIRST_HIGH_OFFSET = 9'h e4;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_6_PAGEIN_LAST_LOW_OFFSET = 9'h e8;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_6_PAGEIN_LAST_HIGH_OFFSET = 9'h ec;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_6_PAGEOUT_LOW_OFFSET = 9'h f0;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_6_PAGEOUT_HIGH_OFFSET = 9'h f4;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_6_FLAGS_OFFSET = 9'h f8;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_7_PAGEIN_FIRST_LOW_OFFSET = 9'h 100;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_7_PAGEIN_FIRST_HIGH_OFFSET = 9'h 104;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_7_PAGEIN_LAST_LOW_OFFSET = 9'h 108;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_7_PAGEIN_LAST_HIGH_OFFSET = 9'h 10c;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_7_PAGEOUT_LOW_OFFSET = 9'h 110;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_7_PAGEOUT_HIGH_OFFSET = 9'h 114;
  parameter logic [BlockAw-1:0] AXI_TLB_TLB_ENTRY_7_FLAGS_OFFSET = 9'h 118;

  // Register index
  typedef enum int {
    AXI_TLB_TLB_ENABLE,
    AXI_TLB_TLB_ENTRY_0_PAGEIN_FIRST_LOW,
    AXI_TLB_TLB_ENTRY_0_PAGEIN_FIRST_HIGH,
    AXI_TLB_TLB_ENTRY_0_PAGEIN_LAST_LOW,
    AXI_TLB_TLB_ENTRY_0_PAGEIN_LAST_HIGH,
    AXI_TLB_TLB_ENTRY_0_PAGEOUT_LOW,
    AXI_TLB_TLB_ENTRY_0_PAGEOUT_HIGH,
    AXI_TLB_TLB_ENTRY_0_FLAGS,
    AXI_TLB_TLB_ENTRY_1_PAGEIN_FIRST_LOW,
    AXI_TLB_TLB_ENTRY_1_PAGEIN_FIRST_HIGH,
    AXI_TLB_TLB_ENTRY_1_PAGEIN_LAST_LOW,
    AXI_TLB_TLB_ENTRY_1_PAGEIN_LAST_HIGH,
    AXI_TLB_TLB_ENTRY_1_PAGEOUT_LOW,
    AXI_TLB_TLB_ENTRY_1_PAGEOUT_HIGH,
    AXI_TLB_TLB_ENTRY_1_FLAGS,
    AXI_TLB_TLB_ENTRY_2_PAGEIN_FIRST_LOW,
    AXI_TLB_TLB_ENTRY_2_PAGEIN_FIRST_HIGH,
    AXI_TLB_TLB_ENTRY_2_PAGEIN_LAST_LOW,
    AXI_TLB_TLB_ENTRY_2_PAGEIN_LAST_HIGH,
    AXI_TLB_TLB_ENTRY_2_PAGEOUT_LOW,
    AXI_TLB_TLB_ENTRY_2_PAGEOUT_HIGH,
    AXI_TLB_TLB_ENTRY_2_FLAGS,
    AXI_TLB_TLB_ENTRY_3_PAGEIN_FIRST_LOW,
    AXI_TLB_TLB_ENTRY_3_PAGEIN_FIRST_HIGH,
    AXI_TLB_TLB_ENTRY_3_PAGEIN_LAST_LOW,
    AXI_TLB_TLB_ENTRY_3_PAGEIN_LAST_HIGH,
    AXI_TLB_TLB_ENTRY_3_PAGEOUT_LOW,
    AXI_TLB_TLB_ENTRY_3_PAGEOUT_HIGH,
    AXI_TLB_TLB_ENTRY_3_FLAGS,
    AXI_TLB_TLB_ENTRY_4_PAGEIN_FIRST_LOW,
    AXI_TLB_TLB_ENTRY_4_PAGEIN_FIRST_HIGH,
    AXI_TLB_TLB_ENTRY_4_PAGEIN_LAST_LOW,
    AXI_TLB_TLB_ENTRY_4_PAGEIN_LAST_HIGH,
    AXI_TLB_TLB_ENTRY_4_PAGEOUT_LOW,
    AXI_TLB_TLB_ENTRY_4_PAGEOUT_HIGH,
    AXI_TLB_TLB_ENTRY_4_FLAGS,
    AXI_TLB_TLB_ENTRY_5_PAGEIN_FIRST_LOW,
    AXI_TLB_TLB_ENTRY_5_PAGEIN_FIRST_HIGH,
    AXI_TLB_TLB_ENTRY_5_PAGEIN_LAST_LOW,
    AXI_TLB_TLB_ENTRY_5_PAGEIN_LAST_HIGH,
    AXI_TLB_TLB_ENTRY_5_PAGEOUT_LOW,
    AXI_TLB_TLB_ENTRY_5_PAGEOUT_HIGH,
    AXI_TLB_TLB_ENTRY_5_FLAGS,
    AXI_TLB_TLB_ENTRY_6_PAGEIN_FIRST_LOW,
    AXI_TLB_TLB_ENTRY_6_PAGEIN_FIRST_HIGH,
    AXI_TLB_TLB_ENTRY_6_PAGEIN_LAST_LOW,
    AXI_TLB_TLB_ENTRY_6_PAGEIN_LAST_HIGH,
    AXI_TLB_TLB_ENTRY_6_PAGEOUT_LOW,
    AXI_TLB_TLB_ENTRY_6_PAGEOUT_HIGH,
    AXI_TLB_TLB_ENTRY_6_FLAGS,
    AXI_TLB_TLB_ENTRY_7_PAGEIN_FIRST_LOW,
    AXI_TLB_TLB_ENTRY_7_PAGEIN_FIRST_HIGH,
    AXI_TLB_TLB_ENTRY_7_PAGEIN_LAST_LOW,
    AXI_TLB_TLB_ENTRY_7_PAGEIN_LAST_HIGH,
    AXI_TLB_TLB_ENTRY_7_PAGEOUT_LOW,
    AXI_TLB_TLB_ENTRY_7_PAGEOUT_HIGH,
    AXI_TLB_TLB_ENTRY_7_FLAGS
  } axi_tlb_id_e;

  // Register width information to check illegal writes
  parameter logic [3:0] AXI_TLB_PERMIT [57] = '{
    4'b 0001, // index[ 0] AXI_TLB_TLB_ENABLE
    4'b 1111, // index[ 1] AXI_TLB_TLB_ENTRY_0_PAGEIN_FIRST_LOW
    4'b 1111, // index[ 2] AXI_TLB_TLB_ENTRY_0_PAGEIN_FIRST_HIGH
    4'b 1111, // index[ 3] AXI_TLB_TLB_ENTRY_0_PAGEIN_LAST_LOW
    4'b 1111, // index[ 4] AXI_TLB_TLB_ENTRY_0_PAGEIN_LAST_HIGH
    4'b 1111, // index[ 5] AXI_TLB_TLB_ENTRY_0_PAGEOUT_LOW
    4'b 1111, // index[ 6] AXI_TLB_TLB_ENTRY_0_PAGEOUT_HIGH
    4'b 0001, // index[ 7] AXI_TLB_TLB_ENTRY_0_FLAGS
    4'b 1111, // index[ 8] AXI_TLB_TLB_ENTRY_1_PAGEIN_FIRST_LOW
    4'b 1111, // index[ 9] AXI_TLB_TLB_ENTRY_1_PAGEIN_FIRST_HIGH
    4'b 1111, // index[10] AXI_TLB_TLB_ENTRY_1_PAGEIN_LAST_LOW
    4'b 1111, // index[11] AXI_TLB_TLB_ENTRY_1_PAGEIN_LAST_HIGH
    4'b 1111, // index[12] AXI_TLB_TLB_ENTRY_1_PAGEOUT_LOW
    4'b 1111, // index[13] AXI_TLB_TLB_ENTRY_1_PAGEOUT_HIGH
    4'b 0001, // index[14] AXI_TLB_TLB_ENTRY_1_FLAGS
    4'b 1111, // index[15] AXI_TLB_TLB_ENTRY_2_PAGEIN_FIRST_LOW
    4'b 1111, // index[16] AXI_TLB_TLB_ENTRY_2_PAGEIN_FIRST_HIGH
    4'b 1111, // index[17] AXI_TLB_TLB_ENTRY_2_PAGEIN_LAST_LOW
    4'b 1111, // index[18] AXI_TLB_TLB_ENTRY_2_PAGEIN_LAST_HIGH
    4'b 1111, // index[19] AXI_TLB_TLB_ENTRY_2_PAGEOUT_LOW
    4'b 1111, // index[20] AXI_TLB_TLB_ENTRY_2_PAGEOUT_HIGH
    4'b 0001, // index[21] AXI_TLB_TLB_ENTRY_2_FLAGS
    4'b 1111, // index[22] AXI_TLB_TLB_ENTRY_3_PAGEIN_FIRST_LOW
    4'b 1111, // index[23] AXI_TLB_TLB_ENTRY_3_PAGEIN_FIRST_HIGH
    4'b 1111, // index[24] AXI_TLB_TLB_ENTRY_3_PAGEIN_LAST_LOW
    4'b 1111, // index[25] AXI_TLB_TLB_ENTRY_3_PAGEIN_LAST_HIGH
    4'b 1111, // index[26] AXI_TLB_TLB_ENTRY_3_PAGEOUT_LOW
    4'b 1111, // index[27] AXI_TLB_TLB_ENTRY_3_PAGEOUT_HIGH
    4'b 0001, // index[28] AXI_TLB_TLB_ENTRY_3_FLAGS
    4'b 1111, // index[29] AXI_TLB_TLB_ENTRY_4_PAGEIN_FIRST_LOW
    4'b 1111, // index[30] AXI_TLB_TLB_ENTRY_4_PAGEIN_FIRST_HIGH
    4'b 1111, // index[31] AXI_TLB_TLB_ENTRY_4_PAGEIN_LAST_LOW
    4'b 1111, // index[32] AXI_TLB_TLB_ENTRY_4_PAGEIN_LAST_HIGH
    4'b 1111, // index[33] AXI_TLB_TLB_ENTRY_4_PAGEOUT_LOW
    4'b 1111, // index[34] AXI_TLB_TLB_ENTRY_4_PAGEOUT_HIGH
    4'b 0001, // index[35] AXI_TLB_TLB_ENTRY_4_FLAGS
    4'b 1111, // index[36] AXI_TLB_TLB_ENTRY_5_PAGEIN_FIRST_LOW
    4'b 1111, // index[37] AXI_TLB_TLB_ENTRY_5_PAGEIN_FIRST_HIGH
    4'b 1111, // index[38] AXI_TLB_TLB_ENTRY_5_PAGEIN_LAST_LOW
    4'b 1111, // index[39] AXI_TLB_TLB_ENTRY_5_PAGEIN_LAST_HIGH
    4'b 1111, // index[40] AXI_TLB_TLB_ENTRY_5_PAGEOUT_LOW
    4'b 1111, // index[41] AXI_TLB_TLB_ENTRY_5_PAGEOUT_HIGH
    4'b 0001, // index[42] AXI_TLB_TLB_ENTRY_5_FLAGS
    4'b 1111, // index[43] AXI_TLB_TLB_ENTRY_6_PAGEIN_FIRST_LOW
    4'b 1111, // index[44] AXI_TLB_TLB_ENTRY_6_PAGEIN_FIRST_HIGH
    4'b 1111, // index[45] AXI_TLB_TLB_ENTRY_6_PAGEIN_LAST_LOW
    4'b 1111, // index[46] AXI_TLB_TLB_ENTRY_6_PAGEIN_LAST_HIGH
    4'b 1111, // index[47] AXI_TLB_TLB_ENTRY_6_PAGEOUT_LOW
    4'b 1111, // index[48] AXI_TLB_TLB_ENTRY_6_PAGEOUT_HIGH
    4'b 0001, // index[49] AXI_TLB_TLB_ENTRY_6_FLAGS
    4'b 1111, // index[50] AXI_TLB_TLB_ENTRY_7_PAGEIN_FIRST_LOW
    4'b 1111, // index[51] AXI_TLB_TLB_ENTRY_7_PAGEIN_FIRST_HIGH
    4'b 1111, // index[52] AXI_TLB_TLB_ENTRY_7_PAGEIN_LAST_LOW
    4'b 1111, // index[53] AXI_TLB_TLB_ENTRY_7_PAGEIN_LAST_HIGH
    4'b 1111, // index[54] AXI_TLB_TLB_ENTRY_7_PAGEOUT_LOW
    4'b 1111, // index[55] AXI_TLB_TLB_ENTRY_7_PAGEOUT_HIGH
    4'b 0001  // index[56] AXI_TLB_TLB_ENTRY_7_FLAGS
  };

endpackage

