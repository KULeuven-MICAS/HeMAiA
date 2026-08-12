// Copyright 2020 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
// Licensed under Solderpad Hardware License, Version 0.51, see LICENSE for details.
{
  param_list: [
    { name: "NumScratchRegs",
      desc: "Number of scratch registers",
      type: "int",
      default: "4"
    },
    { name: "NumPads",
      desc: "Number of GPIO pads in the chip.",
      type: "int",
      default: "31"
    },
    { name: "NumMailboxRegs",
      desc: "Number of Mailbox scartch registers",
      type: "int",
      default: "${nr_clusters}"
    },
    { name: "NumKernelTabRegs",
      desc: "Number of Kernel Table registers",
      type: "int",
      default: "4"
    }
  ],
  name: "${name}_SoC",
  clock_primary: "clk_i",
  bus_interfaces: [
    { protocol: "reg_iface", direction: "device" }
  ],
  regwidth: 32,
  registers: [
    { name: "VERSION",
      desc: "Version register, should read 1.",
      swaccess: "ro",
      hwaccess: "none",
      fields: [
        {
          bits: "15:0",
          resval: "1",
          name: "VERSION",
          desc: '''
                System version.
                '''
        }
      ]
    },
    { name: "CHIP_ID",
      desc: "Id of chip for multi-chip systems.",
      swaccess: "ro",
      hwaccess: "hwo",
      hwqe:     "true",
      hwext:    "true",
      fields: [
        {
          bits: "1:0",
          resval: "0",
          name: "CHIP_ID",
          desc: '''
                Id of chip for multi-chip systems.
                '''
        }
      ]
    },
    { multireg:
      { name: "SCRATCH",
        desc: "Scratch register for SW to write to.",
        swaccess: "rw",
        hwaccess: "none",
        count: "NumScratchRegs",
        cname: "scratch",
        fields: [
          { bits: "31:0",
            resval: "0",
            name: "SCRATCH",
            desc: '''
                  Scratch register for software to read/write.
                  '''
          }
        ]
      }
    },

    { name: "BOOT_MODE",
      desc: "Selected boot mode exposed a register.",
      swaccess: "ro",
      hwaccess: "hwo",
      hwqe:     "true",
      hwext:    "true",
      fields: [
        { bits: "1:0",
          name: "MODE",
          desc: "Selected boot mode.",
          enum: [
               { value: "0", name: "idle", desc: "Governor idles in bootrom." },
               { value: "1", name: "serial", desc: "Governor jumps to the base of the serial." },
               { value: "2", name: "i2c", desc: "Governor tries to boot from I2C." }
          ]
        }
      ]
    },
    { name: "NUM_QUADRANTS",
      desc: "Number of quadrants per chip.",
      swaccess: "ro",
      hwaccess: "none",
      hwqe:     "true",
      hwext:    "true",
      fields: [
        {
          bits: "31:0",
          resval: ${nr_s1_quadrants},
          name: "NUM_QUADRANTS",
          desc: '''
                Number of quadrants per chip.
                '''
        }
      ]
    },
    { multireg:
      { name: "PAD",
        desc: "GPIO pad configuration.",
        swaccess: "rw",
        hwaccess: "hro",
        count: "NumPads",
        cname: "pad",
        fields: [
          { bits: "0",
            name: "SLW",
            resval: "0",
            desc: '''
                    Slew control.
                    1: when VDDIO = 1.5/1.2V
                    0: when VDDIO = 1.8V
                  '''
          },
          { bits: "1",
            name: "SMT",
            resval: "0",
            desc: "Active high Schmitt Trigger enable."
          },
          { bits: "3:2",
            name: "DRV",
            resval: "2",
            desc: "Drive strength."
          }
        ]
      }
    },
    { multireg:
      { name: "MAILBOX_SCRATCH",
        desc: "Scratch register holding the mailbox pointers.",
        swaccess: "rw",
        hwaccess: "none",
        count: "NumMailboxRegs",
        cname: "mailbox_scratch",
        fields: [
          { bits: "31:0",
            resval: "0",
            name: "MAILBOX_PTR",
            desc: '''
                  Scratch register holding the mailbox pointers.
                  '''
          }
        ]
      }
    },
    { multireg:
      { name: "KERNEL_TAB_SCRATCH",
        desc: "Scratch register holding the kernel tabel. 1. Ready 2. Kernel Table Start Addr 3. Kernel Table End Addr. 4. Offload Type",
        swaccess: "rw",
        hwaccess: "none",
        count: "NumKernelTabRegs",
        cname: "kernel_tab_scratch",
        fields: [
          { bits: "31:0",
            resval: "0",
            name: "KERNAL_TAB_REGS",
            desc: '''
                  Scratch register holding the mailbox pointers.
                  '''
          }
        ]
      }
    },

    { name: "IO_DRIVE_STRENGTH",
      desc: '''
            Drive strength of the chip-level digital IO pads, grouped by peripheral domain.
            Each field is wired to the DS pins of every output-capable pad in its group.
            A higher value gives more drive current and a faster edge, at the cost of
            simultaneous-switching noise on the IO supply and of overshoot/ringing into an
            unterminated board trace.

            The reset value is the lowest setting that still meets the edge-rate requirement of
            every boot-critical pin at the slow corner for the assumed board load, while staying
            inside the simultaneous-switching budget of the IO power pairs. Only a narrow band
            of settings satisfies both bounds at once, and the assumed board load is
            load-bearing: it moves the slew floor and the noise ceiling in opposite directions.
            The drive current, output impedance and driving-factor tables behind this choice are
            in the IO library databook, which is under NDA and is deliberately not reproduced
            here.

            Software may retune any group once it is up; only change a group while its pins are
            idle.

            Input-only pads ignore DS and are not covered here, and the die-to-die PHY pads have
            their own configuration inside the D2D link.
            ''',
      swaccess: "rw",
      hwaccess: "hro",
      fields: [
        { bits: "3:0",
          name: "MISC",
          resval: "3",
          desc: "PLL lock and observation clock outputs (pll_lock_o, clk_obs_o)."
        },
        { bits: "7:4",
          name: "D2D",
          resval: "3",
          desc: '''
                Die-to-die sideband outputs on all four sides: test_request_o,
                flow_control_rts_o and flow_control_cts_o. Not the D2D PHY data pads.
                '''
        },
        { bits: "11:8",
          name: "UART",
          resval: "3",
          desc: "UART outputs (uart_tx_o, uart_rts_no)."
        },
        { bits: "15:12",
          name: "GPIO",
          resval: "3",
          desc: "GPIO pads, when driven as outputs."
        },
        { bits: "19:16",
          name: "SPIM",
          resval: "3",
          desc: "SPI master outputs (spim_sck_o, spim_csb_o, spim_sd)."
        },
        { bits: "23:20",
          name: "SPIS",
          resval: "3",
          desc: "SPI slave data pads, when driven as outputs."
        },
        { bits: "27:24",
          name: "I2C",
          resval: "3",
          desc: '''
                I2C pads (i2c_sda, i2c_scl). These are open-drain with an external pull-up and
                are the only pads with a real DC load; they generally want the weakest setting
                that still meets the bus V_OL.
                '''
        },
        { bits: "31:28",
          name: "JTAG",
          resval: "3",
          desc: "JTAG data output (jtag_tdo_o)."
        }
      ]
    }
  ]
}
