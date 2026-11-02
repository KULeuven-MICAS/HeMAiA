// Stimulus for tb_sched_latency (included inside the bingo harness module).
//
//   REMOTE edge: chip 0 core 1 runs A (task 1), whose dep set targets chip 1 row 0 (core 0);
//                chip 1 core 0 runs C (task 2), which checks column 1.
//   LOCAL edge:  chip 0 core 1 runs B (task 3), whose dep set targets chip 0 row 0;
//                chip 0 core 0 runs D (task 4), which checks column 1.
// C and D are pushed first, so their cores sit in a blocking CSR read of their ready queue when
// the dependency resolves, as a DM core waiting for its next level does.
//
// Probes (bingo clock = the harness's 10 ns cycle; the scheduler runs at the 1 GHz host clock):
//   t_done   the core's done CSR write handshake (task done -> done queue)
//   t_tx     the dep-set AXI-Lite AW leaving the sender (to_remote_chiplet_axi_lite)
//   t_rx     that write's AW arriving at the receiver (from_remote_axi_lite)
//   t_ready  the dependent task pushed into its ready queue
//   t_rsp    the dependent core's CSR read returning that task
localparam int unsigned EXPECTED_TASK_COUNT     = 4;
localparam int unsigned DEADLOCK_THRESHOLD      = 5000;
localparam int unsigned DEP_MATRIX_LOG_INTERVAL = 0;

bingo_hw_manager_task_desc_full_t sa = pack_normal_task(
    2'b00, 1, 0, 0, 1,
    1'b0, '0,
    1'b1, 1'b0, 1, 0, bingo_hw_manager_dep_code_t'(1 << 0));
bingo_hw_manager_task_desc_full_t sc = pack_normal_task(
    2'b00, 2, 1, 0, 0,
    1'b1, bingo_hw_manager_dep_code_t'(1 << 1),
    1'b0, 1'b0, 0, 0, '0);
bingo_hw_manager_task_desc_full_t sb = pack_normal_task(
    2'b00, 3, 0, 0, 1,
    1'b0, '0,
    1'b1, 1'b0, 0, 0, bingo_hw_manager_dep_code_t'(1 << 0));
bingo_hw_manager_task_desc_full_t sd = pack_normal_task(
    2'b00, 4, 0, 0, 0,
    1'b1, bingo_hw_manager_dep_code_t'(1 << 1),
    1'b0, 1'b0, 0, 0, '0);

initial begin : push_sequence
    automatic axi_pkg::resp_t resp;
    wait (rst_ni);
    @(posedge clk_i);
    task_queue_master[1].write(task_queue_base[1], '0, sc, '1, resp);
    task_queue_master[0].write(task_queue_base[0], '0, sd, '1, resp);
    #200;
    task_queue_master[0].write(task_queue_base[0], '0, sa, '1, resp);
    #2000;   // B after the remote edge has settled, so the two edges do not overlap
    task_queue_master[0].write(task_queue_base[0], '0, sb, '1, resp);
end

realtime t_done_a = 0, t_tx = 0, t_rx = 0, t_ready_c = 0, t_rsp_c = 0;
realtime t_done_b = 0, t_ready_d = 0, t_rsp_d = 0;
logic [31:0] last_ready_chip0_core0;
always @(posedge clk_i) begin
    // done CSR writes (core 1 of chip 0); A is its first task, B its second
    if (csr_req_valid[0][1][0] && csr_req_ready[0][1][0] && csr_req[0][1][0].write) begin
        if (t_done_a == 0) t_done_a = $realtime;
        else if (t_done_b == 0) t_done_b = $realtime;
    end
    if (h2h_axi_lite_xbar_in_req[0].aw_valid && h2h_axi_lite_xbar_in_resp[0].aw_ready && t_tx == 0)
        t_tx = $realtime;
    if (h2h_axi_lite_xbar_out_req[1].aw_valid && h2h_axi_lite_xbar_out_resp[1].aw_ready && t_rx == 0)
        t_rx = $realtime;
    if (gen_dut[1].i_dut.ready_queue_push[0][0] &&
        gen_dut[1].i_dut.ready_queue_data_in[0][0].task_id == 2) t_ready_c = $realtime;
    if (gen_dut[0].i_dut.ready_queue_push[0][0] &&
        gen_dut[0].i_dut.ready_queue_data_in[0][0].task_id == 4) t_ready_d = $realtime;
    if (csr_resp_valid[1][0][0] && csr_resp_ready[1][0][0] && csr_resp[1][0][0].data == 2 && t_rsp_c == 0)
        t_rsp_c = $realtime;
    if (csr_resp_valid[0][0][0] && csr_resp_ready[0][0][0] && csr_resp[0][0][0].data == 4 && t_rsp_d == 0)
        t_rsp_d = $realtime;
end

final begin
    $display("[SCHED] cycle=%0t", CyclTime);
    $display("[SCHED] remote: done=%0t tx=%0t rx=%0t ready=%0t rsp=%0t", t_done_a, t_tx, t_rx, t_ready_c, t_rsp_c);
    $display("[SCHED] remote: sender done->tx = %0d cycles", int'((t_tx - t_done_a) / CyclTime));
    $display("[SCHED] remote: harness h2h xbar tx->rx = %0d cycles (replaced by the real path)", int'((t_rx - t_tx) / CyclTime));
    $display("[SCHED] remote: receiver rx->ready = %0d, ready->rsp = %0d, rx->rsp = %0d cycles",
             int'((t_ready_c - t_rx) / CyclTime), int'((t_rsp_c - t_ready_c) / CyclTime), int'((t_rsp_c - t_rx) / CyclTime));
    $display("[SCHED] local: done=%0t ready=%0t rsp=%0t", t_done_b, t_ready_d, t_rsp_d);
    $display("[SCHED] local: done->ready = %0d, ready->rsp = %0d, done->rsp = %0d cycles",
             int'((t_ready_d - t_done_b) / CyclTime), int'((t_rsp_d - t_ready_d) / CyclTime), int'((t_rsp_d - t_done_b) / CyclTime));
end
