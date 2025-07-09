# Copyright 2024 KU Leuven.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51

# Fanchen Kong <fanchen.kong@kuleuven.be>
import re
def parser_mem_info(transcript):
    with open(transcript, "r") as file:
        log_text = file.read()
    # Split the input text by '#################################################################################'
    # This split is defined in tc_sram_impl
    divider = "#################################################################################"
    split_log = log_text.split(divider)
    # Extract the memory information, which is everything between the first and last divider
    # We skip any unrelated blocks before and after the dividers
    if len(split_log) > 2:
        # Everything between the first and last divider
        mem_info = split_log[1:-1]
    else:
        mem_info = []
        raise Exception("Sorry, no mem info found in the transcript")
    return mem_info

def get_sim_mem_info(transcript):
    mem_info = parser_mem_info(transcript)
    # Now, let's further split the memory info based on each block's content
    # Regular expressions for extracting key memory info fields
    instance_pattern = r"Instance:\s*(.*)"
    num_ports_pattern = r"Number of ports\s+\(dec\):\s*(\d+)"
    num_words_pattern = r"Number of words\s+\(dec\):\s*(\d+)"
    data_width_pattern = r"Data width\s+\(dec\):\s*(\d+)"
    # List to store the extracted information
    sim_mem_info = {}
    # Process each memory info block
    for block in mem_info:
        # Clean up the block by stripping excess whitespace
        block = block.strip()
        # Search for the instance, number of words, and data width
        instance = re.search(instance_pattern, block)
        num_ports = re.search(num_ports_pattern, block)
        num_words = re.search(num_words_pattern, block)
        data_width = re.search(data_width_pattern, block)
        # If all fields are found, store them in the mem_configs list
        if instance and num_ports and num_words and data_width:
            instance_val = instance.group(1).strip()
            num_ports = int(num_ports.group(1))
            num_words_val = int(num_words.group(1))
            data_width_val = int(data_width.group(1))
            # Use the instance as the key and the (num_ports, num_words, data_width) tuple as the value
            sim_mem_info[instance_val] = (num_ports, num_words_val, data_width_val)
    return sim_mem_info