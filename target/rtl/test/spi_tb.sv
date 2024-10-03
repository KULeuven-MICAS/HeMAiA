task automatic spi_read(input logic [31:0] addr, input integer length);
  // Inputs:
  //   addr   - 32-bit Address to read from
  //   length - Number of bytes to read
  // Output:
  //   data   - Array to store read data

  reg [7:0] cmd;  // SPI read command code
  integer i, j;
  reg [3:0] mosi_data;  // Data to send over SPI (master out)
  reg [3:0] miso_data;  // Data received from SPI (slave out)

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
  spis_csb_i = 1; // Bring CSB high to end the transaction
  #1us;  // Wait for a clock edge to align
  @(posedge spis_sck_i);
  spis_csb_i = 0; // Bring CSB high to end the transaction

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
  for (i = 0; i < length; i = i + 1) begin
    reg [7:0] byte_data;
    for (j = 7; j >= 0; j -= 4) begin
      @(posedge spis_sck_i);
      miso_data = spis_sd_o;  // Read 4 bits from slave
      if (j >= 3) begin
        byte_data[j-:4] = miso_data;
      end else begin
        // For j = 3 to 0
        byte_data[3:0] = miso_data >> (3 - j);
      end
    end
    $display("Read byte %0d: %h", i, byte_data);  // Print the byte to the console
  end

  // Bring CSB high to end the transaction
  @(negedge spis_sck_i);
  spis_csb_i = 1;
endtask
