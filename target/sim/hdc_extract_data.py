# This program extracts data from programs for processing

import subprocess

# List strings
list_str = []

# Some fixed values for convenience
line_number = 0

data_extract = {
    'DMTI': 0,
    'DMTP': 0,
    'CGTP': 0,
    'CRTP': 0,
    'CGTL': 0,
    'CRTL': 0,
}

acceptable_tags = ['DMTI', 'DMTP', 'CGTP', 'CRTP', 'CGTL', 'CRTL']

with open("uart0.log", 'r') as f:
        for current_line, line in enumerate(f, start=1):
            if current_line >= line_number:
                line_list = line.strip().split()[1:]
                
                if line_list[0] in acceptable_tags:
                    data_extract[line_list[0]] += int(line_list[1])



total_non_compute = data_extract['DMTI'] + data_extract['CGTP'] + data_extract['CGTL']
total_compute = data_extract['CRTP'] + data_extract['CRTL']
total_time = total_non_compute + total_compute

# print(f"DMTI: {data_extract['DMTI']}")
# print(f"DMTP: {data_extract['DMTP']}")
# print(f"CGTP: {data_extract['CGTP']}")
# print(f"CRTP: {data_extract['CRTP']}")
# print(f"CGTL: {data_extract['CGTL']}")
# print(f"CRTL: {data_extract['CRTL']}")
# print(f"TOTT: {total_time}")
# print(f"TOTC: {total_compute}")
# print(f"TOTU: {total_compute/total_time}")


print(f"{data_extract['DMTI']}")
print(f"{data_extract['DMTP']}")
print(f"{data_extract['CGTP']}")
print(f"{data_extract['CRTP']}")
print(f"{data_extract['CGTL']}")
print(f"{data_extract['CRTL']}")
print("break")
print(f"{total_time}")
print(f"{total_compute}")
print(f"{total_compute/total_time}")
