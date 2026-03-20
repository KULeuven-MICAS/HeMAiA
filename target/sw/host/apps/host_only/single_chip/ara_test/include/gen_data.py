# Copyright 2021 ETH Zurich and University of Bologna.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Generate input data for fdotp benchmark
# arg: #elements per vector

import numpy as np
import random
from functools import reduce
import sys

def emit(name, array, dtype, alignment='8'):
  print(f"{dtype} {name}[] __attribute__((aligned({alignment}))) = {{")
  first = True
  for x in array.flatten():
      if not first:
          print(", ", end="")
      print(f"{x}", end="")
      first = False
  print("};")

def emit_scalar(name, value, dtype):
  print(f"{dtype} {name} = {value};")

# Vector length
vsize = 256

avl32 = int(vsize)
avl16 = int(vsize)

# Create the vectors
v32a = np.random.randint(-2**(20), high=2**(20)-1, size=avl32, dtype=np.int32)
v32b = np.random.randint(-2**(20), high=2**(20)-1, size=avl32, dtype=np.int32)
v16a = np.random.randint(-2**(10), high=2**(10)-1, size=avl16, dtype=np.int16)
v16b = np.random.randint(-2**(10), high=2**(10)-1, size=avl16, dtype=np.int16)

# Create the golden output
gold32 = reduce(lambda a, b: a+b, np.multiply(v32a, v32b))
gold16 = reduce(lambda a, b: a+b, np.multiply(v16a, v16b))
gold16 = np.array([gold16, gold16])


# Print information on file
print("#ifndef DATA_H")
print("#define DATA_H")
print("#include <stdint.h>")
print("")

emit_scalar("vsize", vsize, "uint64_t")
print("")
emit("v32a", v32a, "int32_t", "NR_LANES*4")
emit("v32b", v32b, "int32_t", "NR_LANES*4")
print("")
emit("v16a", v16a, "int16_t", "NR_LANES*4")
emit("v16b", v16b, "int16_t", "NR_LANES*4")

print("")
emit_scalar("gold32", gold32, "int32_t")
emit_scalar("gold16", gold16[0], "int16_t") # gold16 is an array in python code, taking first element

print("")
print("#endif")
