// Copyright 2026 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
// Yunhao Deng <yunhao.deng@kuleuven.be>


module cds_thru (
    inout wire src,
    inout wire dst
);

  tran (src, dst);

endmodule
