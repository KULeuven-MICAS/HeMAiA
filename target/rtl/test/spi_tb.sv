`define SEEK_SET 0
`define SEEK_CUR 1
`define SEEK_END 2

task automatic spi_init();
  reg [7:0] cmd;  // SPI command code
  integer i;

  // Wait for a clock edge to align
  @(posedge spis_sck_i);
  spis_csb_i = 0;

  // Switch SPI to Quad SPI mode
  cmd = 8'h1;
  for (i = 7; i >= 0; i--) begin
    @(negedge spis_sck_i);
    spis_sd_i[0] = cmd[i];  // Send 1 bit at a time on MOSI
  end
  // Enable Quad SPI mode by writing 0x01 to the status register
  for (i = 7; i >= 0; i--) begin
    @(negedge spis_sck_i);
    spis_sd_i[0] = cmd[i];  // Send 1 bit at a time on MOSI
  end

  @(posedge spis_sck_i);
  spis_csb_i = 1;  // Bring CSB high to end the transaction

endtask

task automatic spi_read(input integer length, input logic [31:0] addr);
  // Inputs:
  //   addr   - 32-bit Address to read from
  //   length - Number of bytes to read
  // Output:
  //   data   - Array to store read data

  reg [7:0] cmd;  // SPI read command code
  integer i, j, k;
  reg [3:0] mosi_data;  // Data to send over SPI (master out)
  reg [3:0] miso_data;  // Data received from SPI (slave out)

  // Wait for a clock edge to align
  @(posedge spis_sck_i);
  spis_csb_i = 0;  // Bring CSB high to end the transaction

  // Set the SPI Read MEM code
  cmd = 8'hB;

  // Send the command code (8 bits) over 4 data lines (2 clock cycles)
  for (i = 7; i >= 0; i -= 4) begin
    @(negedge spis_sck_i);
    if (i >= 3) begin
      mosi_data = cmd[i-:4];
    end else begin
      // For i = 3 to 0
      mosi_data = cmd[3:0];
      mosi_data = mosi_data << (3 - i);  // Left-align to 4 bits
    end
    spis_sd_i = mosi_data;  // Drive data lines
  end

  // Send the 32-bit address over 4 data lines (8 clock cycles)
  for (i = 31; i >= 0; i -= 4) begin
    @(negedge spis_sck_i);
    if (i >= 3) begin
      mosi_data = addr[i-:4];
    end else begin
      // For i = 3 to 0
      mosi_data = addr[3:0];
      mosi_data = mosi_data << (3 - i);  // Left-align to 4 bits
    end
    spis_sd_i = mosi_data;  // Drive data lines
  end

  @(negedge spis_sck_i);  // Wait for last data to be sent


  // Insert dummy cycles if required (e.g., 32 cycles)
  // This is the bug of ETH: @spi_slave_rx.sv, the counter count one more cycles
  for (i = 0; i <= 32; i = i + 1) begin
    @(posedge spis_sck_i);
    // Do nothing, just wait
  end

  // Now read the data from the slave
  // Becareful that the data is coming out from SPI in reversed order (Most Significant Byte first, most significant bit inside one byte first)
  for (i = 0; i < length; i = i + 4) begin
    reg [7:0] byte_data[4] = '{default: 8'h00};

    for (j = 3; j >= 0; j -= 1) begin
      for (k = 7; k >= 0; k -= 4) begin
        @(posedge spis_sck_i);
        miso_data = spis_sd_o;  // Read 4 bits from slave
        if (k >= 3) begin
          byte_data[j][k-:4] = miso_data;
        end else begin
          // For j = 3 to 0
          byte_data[j][3:0] = miso_data >> (3 - j);
        end
      end
    end
    for (j = 0; j < 4; j = j + 1) begin
      $display("Read byte %0d: %h", (i + j), byte_data[j]);  // Print the byte to the console
    end
  end

  // Bring CSB high to end the transaction
  @(negedge spis_sck_i);
  spis_csb_i = 1;
endtask

task automatic spi_read_u32(input logic [31:0] addr);
  spi_read(4, addr);
endtask

task automatic spi_write_u32(input logic [31:0] data, input logic [31:0] addr);
  // Inputs:
  //   data - 8-bit data to write
  //   addr - 32-bit Address to write to

  reg [7:0] cmd;  // SPI write command code
  integer i, j;
  reg [3:0] mosi_data;  // Data to send over SPI (master out)

  // Wait for a clock edge to align
  @(posedge spis_sck_i);
  spis_csb_i = 0;  // Bring CSB high to end the transaction

  // Set the SPI Write MEM code
  cmd = 8'h2;

  // Send the command code (8 bits) over 4 data lines (2 clock cycles)
  for (i = 7; i >= 0; i -= 4) begin
    @(negedge spis_sck_i);
    if (i >= 3) begin
      mosi_data = cmd[i-:4];
    end else begin
      // For i = 3 to 0
      mosi_data = cmd[3:0];
      mosi_data = mosi_data << (3 - i);  // Left-align to 4 bits
    end
    spis_sd_i = mosi_data;  // Drive data lines
  end

  // Send the 32-bit address over 4 data lines (8 clock cycles)
  for (i = 31; i >= 0; i -= 4) begin
    @(negedge spis_sck_i);
    if (i >= 3) begin
      mosi_data = addr[i-:4];
    end else begin
      // For i = 3 to 0
      mosi_data = addr[3:0];
      mosi_data = mosi_data << (3 - i);  // Left-align to 4 bits
    end
    spis_sd_i = mosi_data;  // Drive data lines
  end

  // Send the 32-bit data over 4 data lines (8 clock cycles)
  for (i = 31; i >= 0; i -= 4) begin
    @(negedge spis_sck_i);
    mosi_data = data[i-:4];
    spis_sd_i = mosi_data;  // Drive data lines

  end
  $display("Wrote to address %h finished", addr);
  @(negedge spis_sck_i);

  // Bring CSB high to end the transaction
  spis_csb_i = 1;
endtask

task automatic spi_write_image(input string path, input logic [31:0] addr);
  // Inputs:
  //   path   - Path to the file to write
  //   addr   - 32-bit Address to read from

  reg [7:0] cmd;  // SPI write command code
  integer i, j, k;
  reg [3:0] mosi_data;  // Data to send over SPI (master out)
  reg [3:0] miso_data;  // Data received from SPI (slave out)
  integer file;
  integer file_size;

  // Start to load binaries from file
  // Wait for a clock edge to align
  @(posedge spis_sck_i);
  spis_csb_i = 0;  // Bring CSB high to end the transaction

  // Set the SPI Write MEM code
  cmd = 8'h2;

  // Send the command code (8 bits) over 4 data lines (2 clock cycles)
  for (i = 7; i >= 0; i -= 4) begin
    @(negedge spis_sck_i);
    if (i >= 3) begin
      mosi_data = cmd[i-:4];
    end else begin
      // For i = 3 to 0
      mosi_data = cmd[3:0];
      mosi_data = mosi_data << (3 - i);  // Left-align to 4 bits
    end
    spis_sd_i = mosi_data;  // Drive data lines
  end

  // Send the 32-bit address over 4 data lines (8 clock cycles)
  for (i = 31; i >= 0; i -= 4) begin
    @(negedge spis_sck_i);
    if (i >= 3) begin
      mosi_data = addr[i-:4];
    end else begin
      // For i = 3 to 0
      mosi_data = addr[3:0];
      mosi_data = mosi_data << (3 - i);  // Left-align to 4 bits
    end
    spis_sd_i = mosi_data;  // Drive data lines
  end

  // Now write the data to the slave
  // Open the file for reading and get the size of the file
  file = $fopen(path, "r");
  if (file == 0) begin
    $display("Error: Could not open file %s", path);
    return;
  end
  $fseek(file, 0, `SEEK_END);
  file_size = $ftell(file);
  $fseek(file, 0, `SEEK_SET);

  // Read the file in chunks of 4 bytes
  for (i = 0; i < file_size; i = i + 4) begin
    reg [7:0] byte_data[4] = '{default: 8'h00};

    for (j = 0; j < 4; j = j + 1) begin
      byte_data[j] = $fgetc(file);
    end

    for (j = 3; j >= 0; j -= 1) begin
      for (k = 7; k >= 0; k -= 4) begin
        @(posedge spis_sck_i);
        if (k >= 3) begin
          mosi_data = byte_data[j][k-:4];
        end else begin
          // For j = 3 to 0
          mosi_data = byte_data[j][3:0] << (3 - j);
        end
        spis_sd_i = mosi_data;  // Drive data lines
      end
    end
    for (j = 0; j < 4; j = j + 1) begin
      $display("Wrote byte %0d: %h", (i + j), byte_data[j]);  // Print the byte to the console
    end
  end
  $fclose(file);
  $display("Wrote to address %h finished", addr);
  @(negedge spis_sck_i);

  // Bring CSB high to end the transaction
  spis_csb_i = 1;
  force i_occamy.rst_ni = 0;  // Force reset the chip
  #10ns;
  release i_occamy.rst_ni;  // Release reset
endtask
