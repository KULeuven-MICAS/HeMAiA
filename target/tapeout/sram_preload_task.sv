// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Author: Yunhao Deng <yunhao.deng@kuleuven.be>
// Always copy the following code into the memory macros that need the initialization

task automatic load_data(input string filename);
  reg [numWordAddr:0] w;
  reg [numWordAddr-numCMAddr-1:0] row;
  reg [numCMAddr-1:0] col;
  begin
    // Initialization of the memory array
    foreach (PRELOAD[i]) PRELOAD[i] = '0;
    // Load data from file
    $readmemh(filename, PRELOAD);
      for (w = 0; w < numWord; w = w + 1) begin
        {row, col} = w;
        MEMORY[row][col] = PRELOAD[w];
    end
    $display("Loaded data from %s into memory", filename);
  end
endtask
