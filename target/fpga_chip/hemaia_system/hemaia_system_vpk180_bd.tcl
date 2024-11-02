
################################################################
# This is a generated script based on design: hemaia_system
#
# Though there are limitations about the generated script,
# the main purpose of this utility is to make learning
# IP Integrator Tcl commands easier.
################################################################

namespace eval _tcl {
proc get_script_folder {} {
   set script_path [file normalize [info script]]
   set script_folder [file dirname $script_path]
   return $script_folder
}
}
variable script_folder
set script_folder [_tcl::get_script_folder]

################################################################
# Check if script is running in correct Vivado version.
################################################################
set scripts_vivado_version 2023.2
set current_vivado_version [version -short]

if { [string first $scripts_vivado_version $current_vivado_version] == -1 } {
   puts ""
   if { [string compare $scripts_vivado_version $current_vivado_version] > 0 } {
      catch {common::send_gid_msg -ssname BD::TCL -id 2042 -severity "ERROR" " This script was generated using Vivado <$scripts_vivado_version> and is being run in <$current_vivado_version> of Vivado. Sourcing the script failed since it was created with a future version of Vivado."}

   } else {
     catch {common::send_gid_msg -ssname BD::TCL -id 2041 -severity "ERROR" "This script was generated using Vivado <$scripts_vivado_version> and is being run in <$current_vivado_version> of Vivado. Please run the script in Vivado <$scripts_vivado_version> then open the design in Vivado <$current_vivado_version>. Upgrade the design by running \"Tools => Report => Report IP Status...\", then run write_bd_tcl to create an updated script."}

   }

   return 1
}

################################################################
# START
################################################################

# To test this script, run the following commands from Vivado Tcl console:
# source hemaia_system_script.tcl

# If there is no project opened, this script will create a
# project, but make sure you do not have an existing project
# <./myproj/project_1.xpr> in the current working folder.

set list_projs [get_projects -quiet]
if { $list_projs eq "" } {
   create_project project_1 myproj -part xcvp1802-lsvc4072-2MP-e-S
   set_property BOARD_PART xilinx.com:vpk180:part0:1.2 [current_project]
}


# CHANGE DESIGN NAME HERE
variable design_name
set design_name hemaia_system

# If you do not already have an existing IP Integrator design open,
# you can create a design using the following command:
#    create_bd_design $design_name

# Creating design if needed
set errMsg ""
set nRet 0

set cur_design [current_bd_design -quiet]
set list_cells [get_bd_cells -quiet]

if { ${design_name} eq "" } {
   # USE CASES:
   #    1) Design_name not set

   set errMsg "Please set the variable <design_name> to a non-empty value."
   set nRet 1

} elseif { ${cur_design} ne "" && ${list_cells} eq "" } {
   # USE CASES:
   #    2): Current design opened AND is empty AND names same.
   #    3): Current design opened AND is empty AND names diff; design_name NOT in project.
   #    4): Current design opened AND is empty AND names diff; design_name exists in project.

   if { $cur_design ne $design_name } {
      common::send_gid_msg -ssname BD::TCL -id 2001 -severity "INFO" "Changing value of <design_name> from <$design_name> to <$cur_design> since current design is empty."
      set design_name [get_property NAME $cur_design]
   }
   common::send_gid_msg -ssname BD::TCL -id 2002 -severity "INFO" "Constructing design in IPI design <$cur_design>..."

} elseif { ${cur_design} ne "" && $list_cells ne "" && $cur_design eq $design_name } {
   # USE CASES:
   #    5) Current design opened AND has components AND same names.

   set errMsg "Design <$design_name> already exists in your project, please set the variable <design_name> to another value."
   set nRet 1
} elseif { [get_files -quiet ${design_name}.bd] ne "" } {
   # USE CASES: 
   #    6) Current opened design, has components, but diff names, design_name exists in project.
   #    7) No opened design, design_name exists in project.

   set errMsg "Design <$design_name> already exists in your project, please set the variable <design_name> to another value."
   set nRet 2

} else {
   # USE CASES:
   #    8) No opened design, design_name not in project.
   #    9) Current opened design, has components, but diff names, design_name not in project.

   common::send_gid_msg -ssname BD::TCL -id 2003 -severity "INFO" "Currently there is no design <$design_name> in project, so creating one..."

   create_bd_design $design_name

   common::send_gid_msg -ssname BD::TCL -id 2004 -severity "INFO" "Making design <$design_name> as current_bd_design."
   current_bd_design $design_name

}

common::send_gid_msg -ssname BD::TCL -id 2005 -severity "INFO" "Currently the variable <design_name> is equal to \"$design_name\"."

if { $nRet != 0 } {
   catch {common::send_gid_msg -ssname BD::TCL -id 2006 -severity "ERROR" $errMsg}
   return $nRet
}

set bCheckIPsPassed 1
##################################################################
# CHECK IPs
##################################################################
set bCheckIPs 1
if { $bCheckIPs == 1 } {
   set list_check_ips "\ 
xilinx.com:ip:xlconstant:1.1\
xilinx.com:ip:xlconcat:2.1\
xilinx.com:ip:util_vector_logic:2.0\
xilinx.com:ip:util_reduced_logic:2.0\
xilinx.com:ip:xlslice:1.0\
MICAS_KUL:user:occamy_chip:1.0\
xilinx.com:ip:axis_vio:1.0\
xilinx.com:ip:versal_cips:3.4\
xilinx.com:ip:util_ds_buf:2.2\
"

   set list_ips_missing ""
   common::send_gid_msg -ssname BD::TCL -id 2011 -severity "INFO" "Checking if the following IPs exist in the project's IP catalog: $list_check_ips ."

   foreach ip_vlnv $list_check_ips {
      set ip_obj [get_ipdefs -all $ip_vlnv]
      if { $ip_obj eq "" } {
         lappend list_ips_missing $ip_vlnv
      }
   }

   if { $list_ips_missing ne "" } {
      catch {common::send_gid_msg -ssname BD::TCL -id 2012 -severity "ERROR" "The following IPs are not found in the IP Catalog:\n  $list_ips_missing\n\nResolution: Please add the repository containing the IP(s) to the project." }
      set bCheckIPsPassed 0
   }

}

if { $bCheckIPsPassed != 1 } {
  common::send_gid_msg -ssname BD::TCL -id 2023 -severity "WARNING" "Will not continue with creation of design due to the error(s) above."
  return 3
}

##################################################################
# DESIGN PROCs
##################################################################



# Procedure to create entire design; Provide argument to make
# procedure reusable. If parentCell is "", will use root.
proc create_root_design { parentCell } {

  variable script_folder
  variable design_name

  if { $parentCell eq "" } {
     set parentCell [get_bd_cells /]
  }

  # Get object for parentCell
  set parentObj [get_bd_cells $parentCell]
  if { $parentObj == "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2090 -severity "ERROR" "Unable to find parent cell <$parentCell>!"}
     return
  }

  # Make sure parentObj is hier blk
  set parentType [get_property TYPE $parentObj]
  if { $parentType ne "hier" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2091 -severity "ERROR" "Parent <$parentObj> has TYPE = <$parentType>. Expected to be <hier>."}
     return
  }

  # Save current instance; Restore later
  set oldCurInst [current_bd_instance .]

  # Set parent object as current
  current_bd_instance $parentObj


  # Create interface ports

  # Create ports
  set reset [ create_bd_port -dir I -type rst reset ]
  set_property -dict [ list \
   CONFIG.POLARITY {ACTIVE_HIGH} \
 ] $reset
  set uart_rx_i [ create_bd_port -dir I uart_rx_i ]
  set uart_tx_o [ create_bd_port -dir O uart_tx_o ]
  set vref_vdd_o [ create_bd_port -dir O -from 0 -to 0 vref_vdd_o ]
  set vref_gnd_o [ create_bd_port -dir O -from 0 -to 0 vref_gnd_o ]
  set jtag_tdo_o [ create_bd_port -dir O jtag_tdo_o ]
  set jtag_tdi_i [ create_bd_port -dir I jtag_tdi_i ]
  set jtag_tms_i [ create_bd_port -dir I jtag_tms_i ]
  set jtag_tck_i [ create_bd_port -dir I -type clk -freq_hz 5000000 jtag_tck_i ]
  set uart_cts_ni [ create_bd_port -dir I uart_cts_ni ]
  set uart_rts_no [ create_bd_port -dir O uart_rts_no ]
  set gpio_d_o [ create_bd_port -dir O -from 3 -to 0 gpio_d_o ]
  set spis_sd_io [ create_bd_port -dir IO -from 3 -to 0 spis_sd_io ]
  set spis_sck_i [ create_bd_port -dir I spis_sck_i ]
  set spis_csb_i [ create_bd_port -dir I spis_csb_i ]

  # Create instance: c_high, and set properties
  set c_high [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 c_high ]

  # Create instance: c_low, and set properties
  set c_low [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 c_low ]
  set_property CONFIG.CONST_VAL {0} $c_low


  # Create instance: concat_rst_core, and set properties
  set concat_rst_core [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 concat_rst_core ]

  # Create instance: rst_core_inv, and set properties
  set rst_core_inv [ create_bd_cell -type ip -vlnv xilinx.com:ip:util_vector_logic:2.0 rst_core_inv ]
  set_property -dict [list \
    CONFIG.C_OPERATION {not} \
    CONFIG.C_SIZE {1} \
  ] $rst_core_inv


  # Create instance: rst_or_core, and set properties
  set rst_or_core [ create_bd_cell -type ip -vlnv xilinx.com:ip:util_reduced_logic:2.0 rst_or_core ]
  set_property -dict [list \
    CONFIG.C_OPERATION {or} \
    CONFIG.C_SIZE {2} \
  ] $rst_or_core


  # Create instance: xlslice_1, and set properties
  set xlslice_1 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlslice:1.0 xlslice_1 ]
  set_property -dict [list \
    CONFIG.DIN_FROM {3} \
    CONFIG.DOUT_WIDTH {4} \
  ] $xlslice_1


  # Create instance: occamy_chip, and set properties
  set occamy_chip [ create_bd_cell -type ip -vlnv MICAS_KUL:user:occamy_chip:1.0 occamy_chip ]

  # Create instance: axis_vio_0, and set properties
  set axis_vio_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_vio:1.0 axis_vio_0 ]
  set_property -dict [list \
    CONFIG.C_NUM_PROBE_IN {0} \
    CONFIG.C_NUM_PROBE_OUT {3} \
    CONFIG.C_PROBE_OUT1_WIDTH {2} \
  ] $axis_vio_0


  # Create instance: versal_cips_0, and set properties
  set versal_cips_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:versal_cips:3.4 versal_cips_0 ]
  set_property -dict [list \
    CONFIG.BOOT_MODE {Custom} \
    CONFIG.CLOCK_MODE {Custom} \
    CONFIG.DDR_MEMORY_MODE {Custom} \
    CONFIG.DEBUG_MODE {JTAG} \
    CONFIG.DESIGN_MODE {0} \
    CONFIG.DEVICE_INTEGRITY_MODE {Custom} \
    CONFIG.PS_PMC_CONFIG { \
      BOOT_MODE {Custom} \
      CLOCK_MODE {Custom} \
      DDR_MEMORY_MODE {Custom} \
      DEBUG_MODE {JTAG} \
      DESIGN_MODE {0} \
      DEVICE_INTEGRITY_MODE {Custom} \
      PMC_ALT_REF_CLK_FREQMHZ {33.333} \
      PMC_BANK_0_IO_STANDARD {LVCMOS1.8} \
      PMC_BANK_1_IO_STANDARD {LVCMOS1.8} \
      PMC_CRP_EFUSE_REF_CTRL_SRCSEL {IRO_CLK/4} \
      PMC_CRP_HSM0_REF_CTRL_FREQMHZ {33.333} \
      PMC_CRP_HSM1_REF_CTRL_FREQMHZ {133.333} \
      PMC_CRP_LSBUS_REF_CTRL_FREQMHZ {100} \
      PMC_CRP_NOC_REF_CTRL_FREQMHZ {960} \
      PMC_CRP_PL0_REF_CTRL_FREQMHZ {50} \
      PMC_CRP_PL1_REF_CTRL_FREQMHZ {3.2768} \
      PMC_CRP_PL5_REF_CTRL_FREQMHZ {400} \
      PMC_MIO0 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}} \
      PMC_MIO1 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}} \
      PMC_PL_ALT_REF_CLK_FREQMHZ {33.333} \
      PMC_QSPI_FBCLK {{ENABLE 1} {IO {PMC_MIO 6}}} \
      PMC_QSPI_PERIPHERAL_ENABLE {0} \
      PMC_REF_CLK_FREQMHZ {33.3333333} \
      PMC_SD0 {{CD_ENABLE 0} {CD_IO {PMC_MIO 24}} {POW_ENABLE 0} {POW_IO {PMC_MIO 17}} {RESET_ENABLE 0} {RESET_IO {PMC_MIO 17}} {WP_ENABLE 0} {WP_IO {PMC_MIO 25}}} \
      PMC_SD0_PERIPHERAL {{CLK_100_SDR_OTAP_DLY 0x00} {CLK_200_SDR_OTAP_DLY 0x00} {CLK_50_DDR_ITAP_DLY 0x00} {CLK_50_DDR_OTAP_DLY 0x00} {CLK_50_SDR_ITAP_DLY 0x00} {CLK_50_SDR_OTAP_DLY 0x00} {ENABLE 0}\
{IO {PMC_MIO 13 .. 25}}} \
      PMC_SD0_SLOT_TYPE {SD 2.0} \
      PMC_SMAP_PERIPHERAL {{ENABLE 0} {IO {32 Bit}}} \
      PMC_USE_PMC_NOC_AXI0 {0} \
      PS_BOARD_INTERFACE {Custom} \
      PS_HSDP_EGRESS_TRAFFIC {JTAG} \
      PS_HSDP_INGRESS_TRAFFIC {JTAG} \
      PS_HSDP_MODE {NONE} \
      PS_USE_FPD_CCI_NOC {0} \
      PS_USE_FPD_CCI_NOC0 {0} \
      PS_USE_NOC_LPD_AXI0 {0} \
      PS_USE_PMCPL_CLK0 {1} \
      PS_USE_PMCPL_CLK1 {1} \
      PS_USE_PMCPL_IRO_CLK {0} \
      SMON_ALARMS {Set_Alarms_On} \
      SMON_ENABLE_TEMP_AVERAGING {0} \
      SMON_MEAS126 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {2 V unipolar}} {NAME VCCAUX} {SUPPLY_NUM 0}} \
      SMON_MEAS127 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {2 V unipolar}} {NAME VCCAUX_PMC} {SUPPLY_NUM 1}} \
      SMON_MEAS148 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {2 V unipolar}} {NAME VCC_PMC} {SUPPLY_NUM 2}} \
      SMON_MEAS149 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {2 V unipolar}} {NAME VCC_PSFP} {SUPPLY_NUM 3}} \
      SMON_MEAS150 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {2 V unipolar}} {NAME VCC_PSLP} {SUPPLY_NUM 4}} \
      SMON_MEAS152 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {2 V unipolar}} {NAME VCC_SOC} {SUPPLY_NUM 5}} \
      SMON_MEAS153 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {2 V unipolar}} {NAME VP_VN} {SUPPLY_NUM 6}} \
      SMON_TEMP_AVERAGING_SAMPLES {0} \
    } \
  ] $versal_cips_0


  # Create instance: spis_iobuf, and set properties
  set spis_iobuf [ create_bd_cell -type ip -vlnv xilinx.com:ip:util_ds_buf:2.2 spis_iobuf ]
  set_property -dict [list \
    CONFIG.C_BUF_TYPE {IOBUF} \
    CONFIG.C_SIZE {4} \
  ] $spis_iobuf


  # Create port connections
  connect_bd_net -net Net3 [get_bd_ports spis_sd_io] [get_bd_pins spis_iobuf/IOBUF_IO_IO]
  connect_bd_net -net axis_vio_0_probe_out2 [get_bd_pins axis_vio_0/probe_out2] [get_bd_pins occamy_chip/test_mode_i]
  connect_bd_net -net bootmode [get_bd_pins axis_vio_0/probe_out1] [get_bd_pins occamy_chip/boot_mode_i]
  connect_bd_net -net c_high_dout [get_bd_pins c_high/dout] [get_bd_ports vref_vdd_o] [get_bd_pins occamy_chip/jtag_trst_ni]
  connect_bd_net -net clk_wizard_0_clk_core [get_bd_pins versal_cips_0/pl0_ref_clk] [get_bd_pins axis_vio_0/clk] [get_bd_pins occamy_chip/clk_i] [get_bd_pins occamy_chip/clk_periph_i]
  connect_bd_net -net const_low_dout [get_bd_pins c_low/dout] [get_bd_ports vref_gnd_o] [get_bd_pins occamy_chip/test_mode_i] [get_bd_pins occamy_chip/gpio_d_i] [get_bd_pins occamy_chip/ext_irq_i]
  connect_bd_net -net i2c_scl_iobuf_IOBUF_IO_O [get_bd_pins i2c_scl_iobuf/IOBUF_IO_O] [get_bd_pins occamy_chip/i2c_scl_i]
  connect_bd_net -net i2c_sda_iobuf_IOBUF_IO_O [get_bd_pins i2c_sda_iobuf/IOBUF_IO_O] [get_bd_pins occamy_chip/i2c_sda_i]
  connect_bd_net -net jtag_tck_i_1 [get_bd_ports jtag_tck_i] [get_bd_pins occamy_chip/jtag_tck_i]
  set_property HDL_ATTRIBUTE.DEBUG {true} [get_bd_nets jtag_tck_i_1]
  connect_bd_net -net jtag_tdi_i_1 [get_bd_ports jtag_tdi_i] [get_bd_pins occamy_chip/jtag_tdi_i]
  set_property HDL_ATTRIBUTE.DEBUG {true} [get_bd_nets jtag_tdi_i_1]
  connect_bd_net -net jtag_tms_i_1 [get_bd_ports jtag_tms_i] [get_bd_pins occamy_chip/jtag_tms_i]
  set_property HDL_ATTRIBUTE.DEBUG {true} [get_bd_nets jtag_tms_i_1]
  connect_bd_net -net occamy_chip_0_gpio_d_o [get_bd_pins occamy_chip/gpio_d_o] [get_bd_pins xlslice_1/Din]
  connect_bd_net -net occamy_chip_0_jtag_tdo_o [get_bd_pins occamy_chip/jtag_tdo_o] [get_bd_ports jtag_tdo_o]
  set_property HDL_ATTRIBUTE.DEBUG {true} [get_bd_nets occamy_chip_0_jtag_tdo_o]
  connect_bd_net -net occamy_chip_0_uart_rts_no [get_bd_pins occamy_chip/uart_rts_no] [get_bd_ports uart_rts_no]
  set_property HDL_ATTRIBUTE.DEBUG {true} [get_bd_nets occamy_chip_0_uart_rts_no]
  connect_bd_net -net occamy_chip_0_uart_tx_o [get_bd_pins occamy_chip/uart_tx_o] [get_bd_ports uart_tx_o]
  set_property HDL_ATTRIBUTE.DEBUG {true} [get_bd_nets occamy_chip_0_uart_tx_o]
  connect_bd_net -net occamy_chip_spis_sd_en_o [get_bd_pins occamy_chip/spis_sd_en_o] [get_bd_pins spis_iobuf/IOBUF_IO_T]
  connect_bd_net -net occamy_chip_spis_sd_o [get_bd_pins occamy_chip/spis_sd_o] [get_bd_pins spis_iobuf/IOBUF_IO_I]
  connect_bd_net -net occamy_rst [get_bd_pins rst_or_core/Res] [get_bd_pins rst_core_inv/Op1]
  connect_bd_net -net occamy_rstn [get_bd_pins rst_core_inv/Res] [get_bd_pins occamy_chip/rst_ni] [get_bd_pins occamy_chip/rst_periph_ni]
  connect_bd_net -net reset [get_bd_pins axis_vio_0/probe_out0] [get_bd_pins concat_rst_core/In1]
  connect_bd_net -net reset_button [get_bd_ports reset] [get_bd_pins concat_rst_core/In0]
  connect_bd_net -net spis_csb_i_0_1 [get_bd_ports spis_csb_i] [get_bd_pins occamy_chip/spis_csb_i]
  connect_bd_net -net spis_iobuf_IOBUF_IO_O [get_bd_pins spis_iobuf/IOBUF_IO_O] [get_bd_pins occamy_chip/spis_sd_i]
  connect_bd_net -net spis_sck_i_0_1 [get_bd_ports spis_sck_i] [get_bd_pins occamy_chip/spis_sck_i]
  connect_bd_net -net uart_cts_ni_0_1 [get_bd_ports uart_cts_ni] [get_bd_pins occamy_chip/uart_cts_ni]
  set_property HDL_ATTRIBUTE.DEBUG {true} [get_bd_nets uart_cts_ni_0_1]
  connect_bd_net -net uart_rx_i_0_1 [get_bd_ports uart_rx_i] [get_bd_pins occamy_chip/uart_rx_i]
  set_property HDL_ATTRIBUTE.DEBUG {true} [get_bd_nets uart_rx_i_0_1]
  connect_bd_net -net versal_cips_0_pl1_ref_clk [get_bd_pins versal_cips_0/pl1_ref_clk] [get_bd_pins occamy_chip/rtc_i]
  connect_bd_net -net xlconcat_2_dout [get_bd_pins concat_rst_core/dout] [get_bd_pins rst_or_core/Op1]
  connect_bd_net -net xlslice_1_Dout [get_bd_pins xlslice_1/Dout] [get_bd_ports gpio_d_o]

  # Create address segments


  # Restore current instance
  current_bd_instance $oldCurInst

  save_bd_design
}
# End of create_root_design()


##################################################################
# MAIN FLOW
##################################################################

create_root_design ""


common::send_gid_msg -ssname BD::TCL -id 2053 -severity "WARNING" "This Tcl script was generated from a block design that has not been validated. It is possible that design <$design_name> may result in errors during validation."

