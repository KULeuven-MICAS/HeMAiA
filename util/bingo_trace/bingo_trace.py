#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>

import re
import json
import argparse
import os
import glob

def parse_perf_tracing_header(header_path):
    """Parses perf_tracing.h to extract event IDs and names."""
    event_map = {}
    with open(header_path, 'r') as f:
        for line in f:
            match = re.search(r'#define\s+(BINGO_TRACE_\w+)\s+(0x[0-9a-fA-F]+|\d+)', line)
            if match:
                name = match.group(1)
                value_str = match.group(2)
                value = int(value_str, 16) if value_str.startswith('0x') else int(value_str)
                event_map[value] = name
    return event_map

def parse_trace_file(file_path, event_map):
    """Parses a single trace file for Magic NOP events."""
    events = []
    
    # regex to match trace lines and capture the first column (simulation time)
    # Example 1 (.txt style): 369000       54        M 0x01000000 ... ori zero, zero, 272 ...
    #   First col: simulation time in picoseconds (ps) -> convert to ns
    #   Second col: cycle counter (kept for compatibility)
    #   Third group: rest of the instruction text
    txt_pattern = re.compile(r'^\s*(\d+)\s+(\d+)\s+\w\s+\S+\s+(.*)$')

    # Example 2 (.log style, Host Core/Hart 0): 447675ns    74605 M 00000000800004ca 0 20106013 ori            x0, 513
    #   First col: simulation time in nanoseconds
    #   Second col: cycle counter
    #   Third group: rest of the instruction text (after other fields)
    # Capture an optional 'ns' suffix so we can treat those values as already in ns
    log_pattern = re.compile(r'^\s*(\d+(?:ns)?)\s+(\d+)\s+\w\s+[0-9a-fA-F]+\s+\d+\s+[0-9a-fA-F]+\s+(.*)$')
    
    # regex to match Magic NOP: ori/xori x0, x0, IMM or ori/xori zero, zero, IMM
    # Captures the immediate value
    magic_nop_pattern = re.compile(r'(?:ori|xori)\s+(?:x0|zero)(?:,\s*(?:x0|zero))?,\s*(-?\d+|0x[0-9a-fA-F]+)')
    
    is_log_file = file_path.endswith('.log')
    pattern = log_pattern if is_log_file else txt_pattern

    with open(file_path, 'r') as f:
        for line in f:
            line_match = pattern.match(line)
            if line_match:
                raw_ts_str = line_match.group(1)
                cycle_str = line_match.group(2)
                cc = int(cycle_str)
                # If this is a host log and the value ends with 'ns', it's already
                # in nanoseconds. Otherwise the trace first column is in
                # picoseconds (ps) and we convert to nanoseconds via integer
                # division.
                if is_log_file and raw_ts_str.endswith('ns'):
                    ts = int(raw_ts_str[:-2])
                else:
                    ts = int(raw_ts_str) // 1000

                instr_str = line_match.group(3)
                
                nop_match = magic_nop_pattern.search(instr_str)
                imm = None

                if nop_match:
                    imm_str = nop_match.group(1)
                    imm = int(imm_str, 16) if imm_str.startswith('0x') else int(imm_str)
                
                if imm is not None:
                    if imm in event_map:
                        event_name = event_map[imm]
                        
                        # Determine phase based on suffix
                        ph = 'i' # Instant by default
                        clean_name = event_name
                        
                        if event_name.endswith('_START'):
                            ph = 'B'
                            clean_name = event_name[:-6] # Remove _START
                        elif event_name.endswith('_END'):
                            ph = 'E'
                            clean_name = event_name[:-4] # Remove _END
                            
                        events.append({
                            'ts': ts,
                            'cc': cc,
                            'name': clean_name,
                            'ph': ph,
                            'id': imm
                        })
    return events

def convert_to_complete_events(events):
    """Converts B/E event pairs into X (Complete) events with duration."""
    # Ensure sorted by timestamp
    events.sort(key=lambda x: x['ts'])
    
    stack = []
    complete_events = []
    
    for ev in events:
        if ev['ph'] == 'B':
            stack.append(ev)
        elif ev['ph'] == 'E':
            # Search stack backwards for matching name
            match_idx = -1
            for i in range(len(stack) - 1, -1, -1):
                if stack[i]['name'] == ev['name']:
                    match_idx = i
                    break
            
            if match_idx != -1:
                start_ev = stack.pop(match_idx)
                dur = ev['ts'] - start_ev['ts']
                
                # Append Complete Event (X)
                # Inherits properties from start event but uses 'X' phase
                new_ev = start_ev.copy()
                new_ev['ph'] = 'X'
                new_ev['dur'] = dur
                # Compute duration in cycles if cycle counters available
                if 'cc' in start_ev and 'cc' in ev:
                    new_ev['dur_cc'] = ev['cc'] - start_ev['cc']
                complete_events.append(new_ev)
            else:
                # Unmatched End event, treat as instant or ignore?
                # Keeping as Instant to show *something* happened
                ev['ph'] = 'i'
                complete_events.append(ev)
        else:
            # Instant events
            complete_events.append(ev)
            
    # Handle leftover starts
    for ev in stack:
        # Treat as instant? Or B without E?
        # Perfetto handles B without E (slice goes to end)
        complete_events.append(ev)
        
    return complete_events

def main():
    parser = argparse.ArgumentParser(description="Generate Perfetto JSON trace from simulation logs.")
    parser.add_argument("--trace-header", required=True, help="Path to perf_tracing.h")
    parser.add_argument("--log-dir", required=True, help="Directory containing trace log files")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    
    args = parser.parse_args()
    
    # 1. Parse Event Map
    print(f"Parsing header: {args.trace_header}")
    event_map = parse_perf_tracing_header(args.trace_header)
    print(f"Found {len(event_map)} event types.")

    # 2. Find Trace Files (Both .txt and .log)
    trace_files = glob.glob(os.path.join(args.log_dir, "trace_chip_*_hart_*.txt"))
    trace_files += glob.glob(os.path.join(args.log_dir, "trace_chip_*_hart_00000.log"))
    
    print(f"Found {len(trace_files)} trace files in {args.log_dir}")

    all_trace_events = []
    
    # 3. Process each trace file
    for trace_file in trace_files:
        basename = os.path.basename(trace_file)
        # Parse filename: trace_chip_XX_hart_XXXXX.txt
        match = re.search(r'trace_chip_(\d+)_hart_(\d+)\.(txt|log)', basename)
        if not match:
            continue
            
        chip_id = int(match.group(1))
        hart_id = int(match.group(2))
        
        # Determine PID/TID mappings
        # PID = Chip ID
        # TID = Hart ID
        # Metadata mostly
        
        print(f"Processing {basename} (Chip {chip_id}, Hart {hart_id})...")
        file_events = parse_trace_file(trace_file, event_map)
        
        # Convert B/E pairs to Complete (X) events to calculate duration
        file_events = convert_to_complete_events(file_events)
        
        for ev in file_events:
            perfetto_event = {
                "name": ev['name'],
                "cat": "bingo",
                "ph": ev['ph'],
                "ts": ev['ts'], # ns
                "pid": f"Chip {chip_id}",
                "tid": f"Core {hart_id}" if hart_id != 0 else "Host Core",
                "args": {}
            }
            
            # If it's a complete event, add duration field to the event object
            if ev['ph'] == 'X':
                perfetto_event['dur'] = ev['dur']
                k_type = ev['name'].replace('BINGO_TRACE_', '')
                perfetto_event['args']['kernel_type'] = k_type
                # Use absolute timestamps in nanoseconds for start/end/duration
                perfetto_event['args']['start_ns'] = ev['ts']
                perfetto_event['args']['end_ns'] = ev['ts'] + ev['dur']
                perfetto_event['args']['dur_ns'] = ev['dur']
                # Include duration in cycles if available
                if 'dur_cc' in ev:
                    perfetto_event['args']['dur_cc'] = ev['dur_cc']
                    # Add frequency calculation
                    ref_freq_mhz = 1000  # Reference frequency in MHz
                    if ev['dur'] != 0:
                        freq_mhz = round((ev['dur_cc'] / ev['dur']) * 1000, 2)  # Convert to MHz
                        ratio = round(ref_freq_mhz/freq_mhz, 2)
                        perfetto_event['args']['freq_MHz'] = freq_mhz
                        perfetto_event['args']['ref_freq_MHz'] = ref_freq_mhz
                        perfetto_event['args']['freq_ratio'] = ratio
            else:
                # For others, keep raw_id logic or none?
                # User said "instead of raw_id". Safe to use raw_id for instant events if needed.
                perfetto_event['args']['raw_id'] = ev['id']
                
            all_trace_events.append(perfetto_event)

    # 4. Add Metadata events for Thread Names
    # We can infer threads from the data we just processed
    threads = set()
    for ev in all_trace_events:
        threads.add((ev['pid'], ev['tid']))
        
    for pid, tid in threads:
        all_trace_events.append({
            "name": "thread_name",
            "ph": "M",
            "pid": pid,
            "tid": tid,
            "args": {
                "name": tid
            }
        })

    # Sort events by timestamp to ensure monotonic time for the viewer
    # This is critical for Perfetto to correctly pair B/E events and calculate spans
    # Metadata events (ph='M') don't have 'ts', so use 0 to place them at the start
    all_trace_events.sort(key=lambda x: x.get('ts', 0))

    # 5. Write Output
    output_data = {
        "traceEvents": all_trace_events,
        "displayTimeUnit": "ns"
    }
    
    print(f"Writing {len(all_trace_events)} events to {args.output}")
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

if __name__ == "__main__":
    main()
