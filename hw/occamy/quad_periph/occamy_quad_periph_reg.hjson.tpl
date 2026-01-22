// Copyright 2020 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
// Licensed under Solderpad Hardware License, Version 0.51, see LICENSE for details.
{
   param_list: [
    { name: "NumClusters",
      desc: "Number of clusters in the chip.",
      type: "int",
      default: "${nr_clusters}"
    }
  ], 
  name: "occamy_quad_periph",
  clock_primary: "clk_i",
  bus_interfaces: [
    { protocol: "reg_iface", direction: "device" }
  ],
  regwidth: 32,
  registers: [
    { multireg:
      { name: "ARG_PTR_LIST_BASE_ADDR",
        desc: "The base address of the arg ptrs.",
        swaccess: "rw",
        hwaccess: "none",
        count: "NumClusters",
        cname: "arg_ptr_list_base_addr",
        fields: [
          {
            bits: "31:0",
            resval: "0",
            name: "ARG_PTR_LIST_BASE_ADDR",
            desc: '''
                  The base address of the arg ptrs.
                  '''
          }
        ]
      }
    },
    { multireg:
      { name: "KERNEL_PTR_LIST_BASE_ADDR",
        desc: "The base address of the kernel ptrs.",
        swaccess: "rw",
        hwaccess: "none",
        count: "NumClusters",
        cname: "kernel_ptr_list_base_addr",
        fields: [
          {
            bits: "31:0",
            resval: "0",
            name: "KERNEL_PTR_LIST_BASE_ADDR",
            desc: '''
                  The base address of the kernel ptrs.
                  '''
          }
        ]
      }
    },
    { multireg:
      { name: "GLOBAL_TASK_ID_TO_DEV_TASK_ID_BASE_ADDR",
        desc: "The base address of the global task id to device task id mapping.",
        swaccess: "rw",
        hwaccess: "none",
        count: "NumClusters",
        cname: "global_task_id_to_dev_task_id_base_addr",
        fields: [
          {
            bits: "31:0",
            resval: "0",
            name: "GLOBAL_TASK_ID_TO_DEV_TASK_ID_BASE_ADDR",
            desc: '''
                  The base address of the global task id to device task id mapping.
                  '''
          }
        ]
      }
    },
    { name: "TASK_DESC_LIST_BASE_ADDR_HI",
      desc: "The higher 32bit base address of the task desciptor lists.",
      swaccess: "rw",
      hwaccess: "hro",
      fields: [
        {
          bits: "31:0",
          resval: "0",
          name: "TASK_DESC_LIST_BASE_ADDR_HI",
          desc: '''
                The higher 32bit base address of the task descriptor lists.
                '''
        }
      ]
    },
    { name: "TASK_DESC_LIST_BASE_ADDR_LO",
      desc: "The lower 32bit base address of the task desciptor lists.",
      swaccess: "rw",
      hwaccess: "hro",
      fields: [
        {
          bits: "31:0",
          resval: "0",
          name: "TASK_DESC_LIST_BASE_ADDR_LO",
          desc: '''
                The lower 32bit base address of the task descriptor lists.
                '''
        }
      ]
    },
    { name: "NUM_TASK",
      desc: "The number of task.",
      swaccess: "rw",
      hwaccess: "hro",
      fields: [
        {
          bits: "31:0",
          resval: "0",
          name: "NUM_TASK",
          desc: '''
                The number of tasks.
                '''
        }
      ]
    },
    { name: "START_BINGO_HW_MANAGER",
      desc: "Start the BINGO HW Manager",
      swaccess: "rw",
      hwaccess: "hrw",
      fields: [
        {
          bits: "31:0",
          resval: "0",
          name: "START_BINGO_HW_MANAGER",
          desc: '''
                Start the BINGO HW Manager.
                '''
        }
      ]
    },
    { name: "HOST_INIT_DONE",
      desc: "Host initialization done",
      swaccess: "rw",
      hwaccess: "hro",
      fields: [
        {
          bits: "31:0",
          resval: "0",
          name: "HOST_INIT_DONE",
          desc: '''
                Host initialization done.
                '''
        }
      ]
    },
  ]
}
