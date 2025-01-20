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
    }
  ],
  name: "${name}_SoC",
  clock_primary: "clk_i",
  bus_interfaces: [
    { protocol: "reg_iface", direction: "device" }
  ],
  interrupt_list: [
    { name: "ecc_narrow_uncorrectable",
      desc: "Detected an uncorrectable ECC error on system narrow SRAM."},
    { name: "ecc_narrow_correctable",
      desc: "Detected a correctable ECC error on system narrow SRAM."},
    { name: "ecc_wide_uncorrectable",
      desc: "Detected an uncorrectable ECC error on system wide SRAM."},
    { name: "ecc_wide_correctable",
      desc: "Detected a correctable ECC error on system wide SRAM."}
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
    { name: "CLK_TRIM",
      desc: "TRIM bits for CLK GEN.",
      swaccess: "rw",
      hwaccess: "hrw",
      fields: [
        { bits: "31:0",
          name: "clk_trim",
          desc: "default trim bits are 9'b0_0011_0000",
          resval: "48"
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

    { name: "BOOT_ADDR_DEFAULT",
      desc: "Default Boot Address.",
      swaccess: "ro",
      hwaccess: "hrw",
      fields: [
        { bits: "31:0",
          name: "boot_addr_default",
          desc: "default boot address ${hex_default_boot_addr}",
          resval: "${int_default_boot_addr}"
        }
      ]
    },

    { name: "BOOT_ADDR_BACKUP",
      desc: "Backup Boot Address.",
      swaccess: "ro",
      hwaccess: "hrw",
      fields: [
        { bits: "31:0",
          name: "boot_addr_backup",
          desc: "backup boot address ${hex_backup_boot_addr}",
          resval: "${int_backup_boot_addr}"
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
    }
  ]
}
