// Copyright 2026 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
// Yunhao Deng <yunhao.deng@kuleuven.be>
// Fanchen Kong <fanchen.kong@kuleuven.be>

// This is highly customized for the 11x6 chiplet interconnect in HeMAiA. It is not intended to be reused for other purposes.
module hemaia_chiplet_interconnect_11x6 (
    L_R0_C0,
    L_R0_C1,
    L_R0_C2,
    L_R0_C3,
    L_R0_C4,
    L_R0_C5,
    L_R1_C0,
    L_R1_C1,
    L_R1_C2,
    L_R1_C3,
    L_R1_C4,
    L_R1_C5,
    L_R2_C0,
    L_R2_C1,
    L_R2_C2,
    L_R2_C3,
    L_R2_C4,
    L_R2_C5,
    L_R3_C0,
    L_R3_C1,
    L_R3_C2,
    L_R3_C3,
    L_R3_C4,
    L_R3_C5,
    L_R4_C0,
    L_R4_C1,
    L_R4_C2,
    L_R4_C3,
    L_R4_C4,
    L_R4_C5,
    L_R5_C0,
    L_R5_C1,
    L_R5_C2,
    L_R5_C3,
    L_R5_C4,
    L_R5_C5,
    L_R6_C0,
    L_R6_C1,
    L_R6_C2,
    L_R6_C3,
    L_R6_C4,
    L_R6_C5,
    L_R7_C0,
    L_R7_C1,
    L_R7_C2,
    L_R7_C3,
    L_R7_C4,
    L_R7_C5,
    L_R8_C0,
    L_R8_C1,
    L_R8_C2,
    L_R8_C3,
    L_R8_C4,
    L_R8_C5,
    L_R9_C0,
    L_R9_C1,
    L_R9_C2,
    L_R9_C3,
    L_R9_C4,
    L_R9_C5,
    L_R10_C0,
    L_R10_C1,
    L_R10_C2,
    L_R10_C3,
    L_R10_C4,
    R_R0_C1,
    R_R0_C2,
    R_R0_C3,
    R_R0_C4,
    R_R0_C5,
    R_R1_C0,
    R_R1_C1,
    R_R1_C2,
    R_R1_C3,
    R_R1_C4,
    R_R1_C5,
    R_R2_C0,
    R_R2_C1,
    R_R2_C2,
    R_R2_C3,
    R_R2_C4,
    R_R2_C5,
    R_R3_C0,
    R_R3_C1,
    R_R3_C2,
    R_R3_C3,
    R_R3_C4,
    R_R3_C5,
    R_R4_C0,
    R_R4_C1,
    R_R4_C2,
    R_R4_C3,
    R_R4_C4,
    R_R4_C5,
    R_R5_C0,
    R_R5_C1,
    R_R5_C2,
    R_R5_C3,
    R_R5_C4,
    R_R5_C5,
    R_R6_C0,
    R_R6_C1,
    R_R6_C2,
    R_R6_C3,
    R_R6_C4,
    R_R6_C5,
    R_R7_C0,
    R_R7_C1,
    R_R7_C2,
    R_R7_C3,
    R_R7_C4,
    R_R7_C5,
    R_R8_C0,
    R_R8_C1,
    R_R8_C2,
    R_R8_C3,
    R_R8_C4,
    R_R8_C5,
    R_R9_C0,
    R_R9_C1,
    R_R9_C2,
    R_R9_C3,
    R_R9_C4,
    R_R9_C5,
    R_R10_C0,
    R_R10_C1,
    R_R10_C2,
    R_R10_C3,
    R_R10_C4,
    R_R10_C5
);

  inout  L_R0_C0, L_R0_C1, L_R0_C2, L_R0_C3, L_R0_C4, L_R0_C5, L_R1_C0, 
    L_R1_C1, L_R1_C2, L_R1_C3, L_R1_C4, L_R1_C5, L_R2_C0, L_R2_C1, 
    L_R2_C2, L_R2_C3, L_R2_C4, L_R2_C5, L_R3_C0, L_R3_C1, L_R3_C2, 
    L_R3_C3, L_R3_C4, L_R3_C5, L_R4_C0, L_R4_C1, L_R4_C2, L_R4_C3, 
    L_R4_C4, L_R4_C5, L_R5_C0, L_R5_C1, L_R5_C2, L_R5_C3, L_R5_C4, 
    L_R5_C5, L_R6_C0, L_R6_C1, L_R6_C2, L_R6_C3, L_R6_C4, L_R6_C5, 
    L_R7_C0, L_R7_C1, L_R7_C2, L_R7_C3, L_R7_C4, L_R7_C5, L_R8_C0, 
    L_R8_C1, L_R8_C2, L_R8_C3, L_R8_C4, L_R8_C5, L_R9_C0, L_R9_C1, 
    L_R9_C2, L_R9_C3, L_R9_C4, L_R9_C5, L_R10_C0, L_R10_C1, L_R10_C2, 
    L_R10_C3, L_R10_C4, R_R0_C1, R_R0_C2, R_R0_C3, R_R0_C4, R_R0_C5, 
    R_R1_C0, R_R1_C1, R_R1_C2, R_R1_C3, R_R1_C4, R_R1_C5, R_R2_C0, 
    R_R2_C1, R_R2_C2, R_R2_C3, R_R2_C4, R_R2_C5, R_R3_C0, R_R3_C1, 
    R_R3_C2, R_R3_C3, R_R3_C4, R_R3_C5, R_R4_C0, R_R4_C1, R_R4_C2, 
    R_R4_C3, R_R4_C4, R_R4_C5, R_R5_C0, R_R5_C1, R_R5_C2, R_R5_C3, 
    R_R5_C4, R_R5_C5, R_R6_C0, R_R6_C1, R_R6_C2, R_R6_C3, R_R6_C4, 
    R_R6_C5, R_R7_C0, R_R7_C1, R_R7_C2, R_R7_C3, R_R7_C4, R_R7_C5, 
    R_R8_C0, R_R8_C1, R_R8_C2, R_R8_C3, R_R8_C4, R_R8_C5, R_R9_C0, 
    R_R9_C1, R_R9_C2, R_R9_C3, R_R9_C4, R_R9_C5, R_R10_C0, R_R10_C1, 
    R_R10_C2, R_R10_C3, R_R10_C4, R_R10_C5;


  cds_thru I64 (
      .src(L_R0_C4),
      .dst(R_R0_C5)
  );

  cds_thru I63 (
      .src(L_R0_C3),
      .dst(R_R0_C4)
  );

  cds_thru I62 (
      .src(L_R0_C2),
      .dst(R_R0_C3)
  );

  cds_thru I61 (
      .src(L_R0_C1),
      .dst(R_R0_C2)
  );

  cds_thru I60 (
      .src(L_R0_C0),
      .dst(R_R0_C1)
  );

  cds_thru I59 (
      .src(L_R1_C4),
      .dst(R_R1_C5)
  );

  cds_thru I58 (
      .src(L_R1_C3),
      .dst(R_R1_C4)
  );

  cds_thru I57 (
      .src(L_R1_C2),
      .dst(R_R1_C3)
  );

  cds_thru I56 (
      .src(L_R1_C1),
      .dst(R_R1_C2)
  );

  cds_thru I55 (
      .src(L_R1_C0),
      .dst(R_R1_C1)
  );

  cds_thru I54 (
      .src(L_R2_C4),
      .dst(R_R2_C5)
  );

  cds_thru I53 (
      .src(L_R2_C3),
      .dst(R_R2_C4)
  );

  cds_thru I52 (
      .src(L_R2_C2),
      .dst(R_R2_C3)
  );

  cds_thru I51 (
      .src(L_R2_C1),
      .dst(R_R2_C2)
  );

  cds_thru I50 (
      .src(L_R2_C0),
      .dst(R_R2_C1)
  );

  cds_thru I49 (
      .src(L_R3_C4),
      .dst(R_R3_C5)
  );

  cds_thru I48 (
      .src(L_R3_C3),
      .dst(R_R3_C4)
  );

  cds_thru I47 (
      .src(L_R3_C2),
      .dst(R_R3_C3)
  );

  cds_thru I46 (
      .src(L_R3_C1),
      .dst(R_R3_C2)
  );

  cds_thru I45 (
      .src(L_R3_C0),
      .dst(R_R3_C1)
  );

  cds_thru I44 (
      .src(L_R4_C4),
      .dst(R_R4_C5)
  );

  cds_thru I43 (
      .src(L_R4_C3),
      .dst(R_R4_C4)
  );

  cds_thru I42 (
      .src(L_R4_C2),
      .dst(R_R4_C3)
  );

  cds_thru I41 (
      .src(L_R4_C1),
      .dst(R_R4_C2)
  );

  cds_thru I40 (
      .src(L_R4_C0),
      .dst(R_R4_C1)
  );

  cds_thru I39 (
      .src(L_R5_C4),
      .dst(R_R5_C5)
  );

  cds_thru I38 (
      .src(L_R5_C3),
      .dst(R_R5_C4)
  );

  cds_thru I37 (
      .src(L_R5_C2),
      .dst(R_R5_C3)
  );

  cds_thru I36 (
      .src(L_R5_C1),
      .dst(R_R5_C2)
  );

  cds_thru I35 (
      .src(L_R5_C0),
      .dst(R_R5_C1)
  );

  cds_thru I34 (
      .src(L_R6_C4),
      .dst(R_R6_C5)
  );

  cds_thru I33 (
      .src(L_R6_C3),
      .dst(R_R6_C4)
  );

  cds_thru I32 (
      .src(L_R6_C2),
      .dst(R_R6_C3)
  );

  cds_thru I31 (
      .src(L_R6_C1),
      .dst(R_R6_C2)
  );

  cds_thru I30 (
      .src(L_R6_C0),
      .dst(R_R6_C1)
  );

  cds_thru I29 (
      .src(L_R7_C4),
      .dst(R_R7_C5)
  );

  cds_thru I28 (
      .src(L_R7_C3),
      .dst(R_R7_C4)
  );

  cds_thru I27 (
      .src(L_R7_C2),
      .dst(R_R7_C3)
  );

  cds_thru I26 (
      .src(L_R7_C1),
      .dst(R_R7_C2)
  );

  cds_thru I25 (
      .src(L_R7_C0),
      .dst(R_R7_C1)
  );

  cds_thru I24 (
      .src(L_R8_C4),
      .dst(R_R8_C5)
  );

  cds_thru I23 (
      .src(L_R8_C3),
      .dst(R_R8_C4)
  );

  cds_thru I22 (
      .src(L_R8_C2),
      .dst(R_R8_C3)
  );

  cds_thru I21 (
      .src(L_R8_C1),
      .dst(R_R8_C2)
  );

  cds_thru I20 (
      .src(L_R8_C0),
      .dst(R_R8_C1)
  );

  cds_thru I19 (
      .src(L_R9_C4),
      .dst(R_R9_C5)
  );

  cds_thru I18 (
      .src(L_R9_C3),
      .dst(R_R9_C4)
  );

  cds_thru I17 (
      .src(L_R9_C2),
      .dst(R_R9_C3)
  );

  cds_thru I16 (
      .src(L_R9_C1),
      .dst(R_R9_C2)
  );

  cds_thru I15 (
      .src(L_R9_C0),
      .dst(R_R9_C1)
  );

  cds_thru I14 (
      .src(L_R10_C4),
      .dst(R_R10_C5)
  );

  cds_thru I13 (
      .src(L_R10_C3),
      .dst(R_R10_C4)
  );

  cds_thru I12 (
      .src(L_R10_C2),
      .dst(R_R10_C3)
  );

  cds_thru I11 (
      .src(L_R10_C1),
      .dst(R_R10_C2)
  );

  cds_thru I10 (
      .src(L_R10_C0),
      .dst(R_R10_C1)
  );

  cds_thru I9 (
      .src(L_R9_C5),
      .dst(R_R10_C0)
  );

  cds_thru I8 (
      .src(L_R8_C5),
      .dst(R_R9_C0)
  );

  cds_thru I7 (
      .src(L_R7_C5),
      .dst(R_R8_C0)
  );

  cds_thru I6 (
      .src(L_R6_C5),
      .dst(R_R7_C0)
  );

  cds_thru I5 (
      .src(L_R5_C5),
      .dst(R_R6_C0)
  );

  cds_thru I4 (
      .src(L_R4_C5),
      .dst(R_R5_C0)
  );

  cds_thru I3 (
      .src(L_R3_C5),
      .dst(R_R4_C0)
  );

  cds_thru I2 (
      .src(L_R2_C5),
      .dst(R_R3_C0)
  );

  cds_thru I1 (
      .src(L_R1_C5),
      .dst(R_R2_C0)
  );

  cds_thru I0 (
      .src(L_R0_C5),
      .dst(R_R1_C0)
  );

endmodule
