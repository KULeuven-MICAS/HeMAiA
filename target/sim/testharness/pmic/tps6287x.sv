// Copyright 2026 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// Behavioral I2C model, based on TI SLVSGC5E sections 7.5 and 8:
// https://www.ti.com/lit/ds/symlink/tps62873.pdf
// Models single-register writes/reads, repeated START, ACK/NACK, register reset,
// and nominal DVS slew. No switching-power-stage, load, thermal or RC pad model.
// rst_ni represents a testbench power cycle. The boot rail is already settled
// when reset is released (the SoC is powered by this rail before software runs).
module tps6287x #(
    parameter logic [6:0] I2C_ADDRESS = 7'h41,
    // Z0 defaults; override for another ordering-code/VSEL combination.
    parameter int BOOT_MV = I2C_ADDRESS == 7'h41 ? 750 :
                            I2C_ADDRESS == 7'h42 ? 875 : 800,
    parameter bit VERBOSE = 1'b1
) (
    input wire rst_ni,
    input wire scl,
    inout wire sda
);
    timeunit 1ns;
    timeprecision 1ps;

    typedef enum {IDLE, ADDRESS, REGISTER, WRITE_DATA, SLAVE_ACK,
                  READ_DATA, MASTER_ACK, IGNORE} state_t;
    state_t state = IDLE, after_ack = IDLE;
    logic [7:0] regs [0:4];
    logic [7:0] pointer = 0, rx_byte = 0, tx_byte = 0;
    logic drive_low = 0, ack_low = 0, ack_sampled = 0;
    logic pending_write = 0, master_nack = 0;
    logic prev_scl = 1, prev_sda = 1;
    int bits_received = 0;
    int unsigned read_count = 0, write_count = 0, vset_write_count = 0;
    int unsigned start_count = 0, stop_count = 0, nack_count = 0;
    real ramp_from_mv, target_mv, ramp_ns;
    realtime ramp_started, ready_at;

    // Open drain: the target never actively drives a one. Pullups are on the bus.
    assign sda = drive_low ? 1'b0 : 1'bz;

    function automatic real output_voltage_mv();
        real elapsed;
        elapsed = $realtime - ramp_started;
        if (ramp_ns <= 0.0 || elapsed >= ramp_ns) return target_mv;
        return ramp_from_mv + (target_mv - ramp_from_mv) * elapsed / ramp_ns;
    endfunction

    function automatic bit voltage_settled();
        return ($realtime - ramp_started) >= ramp_ns;
    endfunction

    function automatic real setpoint_mv();
        case (regs[2][3:2])
            0: return 400.0 + int'(regs[0]) * 1.25;
            1: return 400.0 + int'(regs[0]) * 2.5;
            2: return 400.0 + int'(regs[0]) * 5.0;
            3: return 800.0 + int'(regs[0]) * 10.0;
        endcase
    endfunction

    task automatic reset_registers();
        regs[0] = 8'((BOOT_MV - 400) / 5);
        regs[1] = 8'h2a;
        regs[2] = 8'h09;
        regs[3] = 8'h00;
        regs[4] = 8'h02;
        ramp_from_mv = real'(BOOT_MV);
        target_mv = real'(BOOT_MV);
        ramp_started = $realtime;
        ramp_ns = 0.0;
    endtask

    task automatic start_ramp();
        real slew, delta;
        ramp_from_mv = output_voltage_mv();
        target_mv = regs[1][5] ? setpoint_mv() : 0.0;
        case (regs[1][1:0])
            0: slew = 10.0;
            1: slew = 5.0;
            2: slew = 1.25;
            3: slew = 0.5;
        endcase
        delta = target_mv - ramp_from_mv;
        if (delta < 0.0) delta = -delta;
        ramp_ns = delta / slew * 1000.0;
        ramp_started = $realtime;
    endtask

    task automatic commit_write();
        write_count++;
        case (pointer)
            0: begin
                regs[0] = rx_byte;
                vset_write_count++;
                start_ramp();
            end
            1: begin
                if (rx_byte[7]) begin
                    reset_registers();
                    ready_at = $realtime + 100us;
                end else begin
                    regs[1] = rx_byte & 8'h7f;
                    if (!regs[1][5] || target_mv == 0.0) start_ramp();
                end
            end
            // A VRANGE change is applied to VOUT by the following VSET write.
            2: regs[2] = rx_byte & 8'h0f;
            3: regs[3] = rx_byte & 8'h03;
            default: ; // STATUS is read-only; reserved pointers are NACKed.
        endcase
        if (VERBOSE)
            $display("[pmic][%m] WRITE addr=0x%02x reg=0x%02x data=0x%02x target=%0.2f mV ramp=%0.2f us at %0t",
                     I2C_ADDRESS, pointer, rx_byte, target_mv, ramp_ns / 1000.0, $time);
    endtask

    initial begin
        if (I2C_ADDRESS < 7'h40 || I2C_ADDRESS > 7'h43)
            $fatal(1, "TPS6287x address must be 0x40..0x43");
        if (BOOT_MV < 400 || BOOT_MV > 1675 || (BOOT_MV - 400) % 5 != 0)
            $fatal(1, "TPS6287x BOOT_MV must fit the reset 5 mV range");
        reset_registers();
        ready_at = 0;
    end

    // One event process owns the protocol state. SDA changes made by this model
    // occur with SCL low; START/STOP detection requires SCL high on both samples.
    always @(scl or sda or rst_ni) begin
        if (!rst_ni) begin
            state = IDLE;
            drive_low = 0;
            pending_write = 0;
            pointer = 0;
            read_count = 0;
            write_count = 0;
            vset_write_count = 0;
            start_count = 0;
            stop_count = 0;
            nack_count = 0;
            ready_at = 0;
            reset_registers();
        end else if (scl === 1'b1 && prev_scl === 1'b1 &&
                     sda === 1'b0 && prev_sda === 1'b1) begin
            state = ADDRESS;
            bits_received = 0;
            pending_write = 0;
            drive_low = 0;
            start_count++;
        end else if (scl === 1'b1 && prev_scl === 1'b1 &&
                     sda === 1'b1 && prev_sda === 1'b0) begin
            state = IDLE;
            drive_low = 0;
            pending_write = 0;
            stop_count++;
        end else if (scl === 1'b1 && prev_scl === 1'b0) begin
            case (state)
                ADDRESS, REGISTER, WRITE_DATA: begin
                    rx_byte = {rx_byte[6:0], sda};
                    bits_received++;
                    if (bits_received == 8) begin
                        ack_low = 1;
                        after_ack = IGNORE;
                        case (state)
                            ADDRESS: begin
                                ack_low = rx_byte[7:1] == I2C_ADDRESS && $realtime >= ready_at;
                                if (ack_low) after_ack = rx_byte[0] ? READ_DATA : REGISTER;
                            end
                            REGISTER: begin
                                pointer = rx_byte;
                                ack_low = rx_byte <= 4;
                                if (ack_low) after_ack = WRITE_DATA;
                            end
                            WRITE_DATA: begin
                                ack_low = pointer < 4;
                                pending_write = ack_low;
                            end
                            default: ;
                        endcase
                        if (!ack_low) nack_count++;
                        state = SLAVE_ACK;
                        ack_sampled = 0;
                    end
                end
                SLAVE_ACK: ack_sampled = 1;
                READ_DATA: begin
                    bits_received++;
                    if (bits_received == 8) begin
                        read_count++;
                        if (pointer == 4) regs[4] = regs[1][5] ? 8'h00 : 8'h02;
                        if (VERBOSE)
                            $display("[pmic][%m] READ addr=0x%02x reg=0x%02x data=0x%02x at %0t",
                                     I2C_ADDRESS, pointer, tx_byte, $time);
                    end
                end
                MASTER_ACK: master_nack = sda;
                default: ;
            endcase
        end else if (scl === 1'b0 && prev_scl === 1'b1) begin
            case (state)
                SLAVE_ACK: begin
                    if (!ack_sampled) drive_low = ack_low;
                    else begin
                        drive_low = 0;
                        if (pending_write) commit_write();
                        pending_write = 0;
                        state = after_ack;
                        bits_received = 0;
                        if (state == READ_DATA) begin
                            tx_byte = pointer <= 4 ? regs[pointer] : 8'hff;
                            drive_low = !tx_byte[7];
                        end
                    end
                end
                READ_DATA: begin
                    if (bits_received == 8) begin
                        drive_low = 0;
                        state = MASTER_ACK;
                    end else drive_low = !tx_byte[7-bits_received];
                end
                MASTER_ACK: begin
                    // Datasheet defines single-byte reads. Await STOP/reSTART.
                    drive_low = 0;
                    state = IGNORE;
                end
                default: ;
            endcase
        end
        prev_scl = scl;
        prev_sda = sda;
    end
endmodule
