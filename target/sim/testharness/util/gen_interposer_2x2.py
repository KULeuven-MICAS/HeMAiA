#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Generate hemaia_chiplet_interposer_2x2.sv
#
# Each hemaia_io_pad has two sides:
#   - IO signal ports (hemaia RTL wires): io_clk_i, io_east_d2d, etc.
#   - Pad ports (physical pads): CRBL_*, D2DE_*, M_*, etc.
# Both sides are exposed from hemaia_chiplet_interposer_2x2.
#
# Internal D2D pads (between adjacent chiplets) are routed through
# hemaia_chiplet_interconnect_11x6 instances using intermediate wires.
# Everything else goes to module ports.
#
# Pad-to-interconnect mapping (verified from LVS netlist):
#   E-W: L_R{r}_C{c} <-> D2DE_X{c}_Y{r}, R_R{r}_C{c} <-> D2DW_X{c}_Y{r}
#   N-S: L_R{r}_C{c} <-> D2DS_X{r}_Y{(c+1)%6}, R_R{r}_C{c} <-> D2DN_X{r}_Y{(c+5)%6}

import os
import re

# ======================================================================
# Extract port lists from hemaia_io_pad.sv
# ======================================================================

def extract_io_pad_ports(io_pad_path):
    """Extract IO signal ports and pad ports from hemaia_io_pad.sv.

    Returns:
        io_signal_ports: list of (name, width_str) for IO signal ports
        pad_ports: list of pad port names (scalar inout wires)
    """
    with open(io_pad_path) as f:
        content = f.read()

    # Extract module port section (between first '(' and matching ')')
    mod_match = re.search(r'module\s+hemaia_io_pad\s*\((.*?)\);', content, re.DOTALL)
    if not mod_match:
        raise ValueError("Cannot find module hemaia_io_pad in " + io_pad_path)
    port_section = mod_match.group(1)

    io_signal_ports = []
    pad_ports = []

    pad_prefixes = ('CRBL_', 'CRBR_', 'CRTL_', 'CRTR_', 'M_', 'D2DE_', 'D2DW_', 'D2DN_', 'D2DS_')

    for m in re.finditer(r'inout\s+wire\s+(\[.*?\]\s*)?(\w+)', port_section):
        width = m.group(1).strip() if m.group(1) else ''
        name = m.group(2)
        if any(name.startswith(p) for p in pad_prefixes):
            pad_ports.append(name)
        else:
            io_signal_ports.append((name, width))

    return io_signal_ports, pad_ports


def format_port_decl(name, width):
    """Format a port declaration line."""
    if width:
        return f"    inout wire {width:14s} {{prefix}}_{name}"
    else:
        return f"    inout wire              {{prefix}}_{name}"


# ======================================================================
# Interconnect pin enumeration and mapping
# ======================================================================

def l_pins():
    for r in range(11):
        for c in range(6):
            if r == 10 and c == 5:
                continue
            yield (r, c)

def r_pins():
    for r in range(11):
        for c in range(6):
            if r == 0 and c == 0:
                continue
            yield (r, c)

def ew_l_pad(r, c): return f"D2DE_X{c}_Y{r}"
def ew_r_pad(r, c): return f"D2DW_X{c}_Y{r}"
def ns_l_pad(r, c): return f"D2DS_X{r}_Y{(c + 1) % 6}"
def ns_r_pad(r, c): return f"D2DN_X{r}_Y{(c + 5) % 6}"

INTERCONNECTS = [
    ("conn_00_10", "ew", "00", "10"),
    ("conn_00_01", "ns", "00", "01"),
    ("conn_01_11", "ew", "01", "11"),
    ("conn_10_11", "ns", "10", "11"),
]

CHIP_INTERNAL = {
    "00": {"east": ("conn_00_10", "L", "ew"), "south": ("conn_00_01", "L", "ns")},
    "10": {"west": ("conn_00_10", "R", "ew"), "south": ("conn_10_11", "L", "ns")},
    "01": {"east": ("conn_01_11", "L", "ew"), "north": ("conn_00_01", "R", "ns")},
    "11": {"west": ("conn_01_11", "R", "ew"), "north": ("conn_10_11", "R", "ns")},
}


def wire_name(conn_name, side, r, c):
    return f"{conn_name}_{side}_R{r}_C{c}"


def pad_name_for_pin(side, mapping_type, r, c):
    if mapping_type == "ew":
        return ew_l_pad(r, c) if side == "L" else ew_r_pad(r, c)
    else:
        return ns_l_pad(r, c) if side == "L" else ns_r_pad(r, c)


def build_internal_pad_set(chip):
    internal_pads = set()
    for direction, (conn_name, side, mapping_type) in CHIP_INTERNAL.get(chip, {}).items():
        pin_iter = l_pins() if side == "L" else r_pins()
        for r, c in pin_iter:
            internal_pads.add(pad_name_for_pin(side, mapping_type, r, c))
    return internal_pads


def build_internal_pad_to_wire(chip):
    pad_to_wire = {}
    for direction, (conn_name, side, mapping_type) in CHIP_INTERNAL.get(chip, {}).items():
        pin_iter = l_pins() if side == "L" else r_pins()
        for r, c in pin_iter:
            pad = pad_name_for_pin(side, mapping_type, r, c)
            pad_to_wire[pad] = wire_name(conn_name, side, r, c)
    return pad_to_wire


# ======================================================================
# Generate the module
# ======================================================================

def generate(io_pad_path):
    io_signal_ports, all_pad_ports = extract_io_pad_ports(io_pad_path)
    chips = ["00", "01", "10", "11"]
    lines = []

    def emit(s=""):
        lines.append(s)

    emit("// Copyright 2026 KU Leuven.")
    emit("// Solderpad Hardware License, Version 0.51, see LICENSE for details.")
    emit("// SPDX-License-Identifier: SHL-0.51")
    emit("// Yunhao Deng <yunhao.deng@kuleuven.be>")
    emit("// Fanchen Kong <fanchen.kong@kuleuven.be>")
    emit("//")
    emit("// Auto-generated by gen_interposer_2x2.py")
    emit("// Top-level interposer simulation model for 2x2 HeMAiA chiplet array.")
    emit("// Instantiates 4 hemaia_io_pad + 4 hemaia_chiplet_interconnect_11x6.")
    emit("//")
    emit("// Per chiplet, exposes both:")
    emit("//   - IO signal ports (hemaia RTL wires): io_clk_i, io_east_d2d, etc.")
    emit("//   - Pad ports (physical pads): CRBL_*, D2DE_*, M_*, etc.")
    emit("// Internal D2D pads (between adjacent chiplets) are routed through")
    emit("// hemaia_chiplet_interconnect_11x6 and are NOT exposed as module ports.")
    emit()
    emit("module hemaia_chiplet_interposer_2x2 (")

    # --- Module ports ---
    port_lines = []
    for chip in chips:
        internal_pads = build_internal_pad_set(chip)
        prefix = f"chip{chip}"

        # IO signal ports
        for name, width in io_signal_ports:
            if width:
                port_lines.append(f"    inout wire {width:14s} {prefix}_{name}")
            else:
                port_lines.append(f"    inout wire              {prefix}_{name}")

        # Pad ports (only those NOT routed through interconnect)
        for pad in all_pad_ports:
            if pad not in internal_pads:
                port_lines.append(f"    inout wire              {prefix}_{pad}")

    emit(",\n".join(port_lines))
    emit(");")
    emit()

    # --- Wire declarations for interconnect ---
    emit("  // ============================================")
    emit("  // Intermediate wires: internal D2D pad <-> interconnect")
    emit("  // ============================================")
    for conn_name, conn_type, left_chip, right_chip in INTERCONNECTS:
        left_dir = "east" if conn_type == "ew" else "south"
        right_dir = "west" if conn_type == "ew" else "north"
        emit(f"  // {conn_name}: chip({left_chip[0]},{left_chip[1]}) {left_dir}"
             f" <-> chip({right_chip[0]},{right_chip[1]}) {right_dir}")
        for r, c in l_pins():
            emit(f"  wire {wire_name(conn_name, 'L', r, c)};")
        for r, c in r_pins():
            emit(f"  wire {wire_name(conn_name, 'R', r, c)};")
        emit()

    # --- IO pad instances ---
    emit("  // ============================================")
    emit("  // hemaia_io_pad instances")
    emit("  // ============================================")

    for chip in chips:
        cx, cy = chip[0], chip[1]
        prefix = f"chip{chip}"
        internal_pads = build_internal_pad_set(chip)
        pad_to_wire = build_internal_pad_to_wire(chip)

        emit(f"  // ---- Chip ({cx}, {cy}) ----")
        emit(f"  hemaia_io_pad chip_{cx}_{cy} (")
        conns = []

        # IO signal ports -> module ports
        for name, width in io_signal_ports:
            conns.append(f"      .{name}({prefix}_{name})")

        # Pad ports
        for pad in all_pad_ports:
            if pad in pad_to_wire:
                conns.append(f"      .{pad}({pad_to_wire[pad]})")
            else:
                conns.append(f"      .{pad}({prefix}_{pad})")

        emit(",\n".join(conns))
        emit("  );")
        emit()

    # --- Interconnect instances ---
    emit("  // ============================================")
    emit("  // hemaia_chiplet_interconnect_11x6 instances")
    emit("  // ============================================")

    for conn_name, conn_type, left_chip, right_chip in INTERCONNECTS:
        left_dir = "east" if conn_type == "ew" else "south"
        right_dir = "west" if conn_type == "ew" else "north"
        emit(f"  // {conn_name}: chip({left_chip[0]},{left_chip[1]}) {left_dir}"
             f" <-> chip({right_chip[0]},{right_chip[1]}) {right_dir}")
        emit(f"  hemaia_chiplet_interconnect_11x6 {conn_name} (")
        conn_ports = []
        for r, c in l_pins():
            conn_ports.append(f"      .L_R{r}_C{c}({wire_name(conn_name, 'L', r, c)})")
        for r, c in r_pins():
            conn_ports.append(f"      .R_R{r}_C{c}({wire_name(conn_name, 'R', r, c)})")
        conn_ports.append(f"      .VSS(1'b0)")
        emit(",\n".join(conn_ports))
        emit("  );")
        emit()

    emit("endmodule")
    return "\n".join(lines)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    interposer_dir = os.path.join(script_dir, "..", "interposer")
    io_pad_path = os.path.join(interposer_dir, "hemaia_io_pad.sv")
    output_file = os.path.join(interposer_dir, "hemaia_chiplet_interposer_2x2.sv")

    content = generate(io_pad_path)

    with open(output_file, "w") as f:
        f.write(content)
        f.write("\n")

    n_lines = len(content.splitlines())
    n_io_ports = content.count("inout wire")
    print(f"Generated {output_file}")
    print(f"  Lines: {n_lines}, Module ports: {n_io_ports}")
