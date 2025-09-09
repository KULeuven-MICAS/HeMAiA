// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
// The memory layout of the tcdm is as follows:
//
// High Addr (TCDM End Address)
// +------------------------------------------+
// | CLS (Cluster-Local Storage)              | 
// |   +------------------------------------+ | Here we put:
// |   | .cbss (un-init cluster shared data | |             1. heap allocator
// |   +------------------------------------+ |             2. offload unit
// |   | .cdata (init cluster shareed data) | |             
// |   +------------------------------------+ |
// +------------------------------------------+
// +------------------------------------------+
// | Return value (8 Byte)                    | 
// |   - Shared by all harts                  |
// +------------------------------------------+
// | Hart 0 stack + tls (stack_size + 8)      | <── Hart 0 SP
// |   +------------------------------------+ |
// |   | Stack (stack_size - tls_size)      | |  // Function call stack
// |   |                                    | |
// |   +------------------------------------+ |
// |   | TLS (tls_size)                     | |  // (Thread-local Storage)
// |   | - .tdata (initialized TLS)         | |
// |   | - .tbss (un-init TLS)              | |
// |   +------------------------------------+ | 
// +------------------------------------------+
// | Hart 1 stack (stack_size + 8)            | <── Hart 1 SP
// |   +------------------------------------+ |
// |   | Stack region                       | |
// |   +------------------------------------+ |
// |   | TLS region                         | |
// |   +------------------------------------+ | 
// +------------------------------------------+
// | ... (other harts)                        |
// +------------------------------------------+
// | Hart (n-1) stack (stack_size + 8)        | <── Hart n-1 SP (DM core)
// |   +------------------------------------+ |
// |   | Stack region                       | |
// |   +------------------------------------+ |
// |   | TLS region                         | |
// |   +------------------------------------+ |
// +------------------------------------------+ 
// +------------------------------------------+
// | Leave some margin here                   | (leave about 4kB) 
// +------------------------------------------+
// +------------------------------------------+ <── align this with 256B(8B*32Bank) for the tcdm width
// | Heap region (heap size)                  | 
// |   +------------------------------------+ |
// |   | .cbss (un-init cluster shared data | |
// |   +------------------------------------+ |
// |   | .cdata (init cluster shareed data) | |
// |   +------------------------------------+ |
// +------------------------------------------+
// Low Addr (TCDM Start Address)
#define NUM_CLS_SHARED_PTRS 16 // Number of shared pointers in the cls structure

typedef struct {
    snrt_l1_allocator_t l1_allocator; //shared pointer to the L1 allocator
    bingo_offload_unit_t bingo_offload_unit; // shared pointer to the offload unit
    uint32_t* shared_ptrs[NUM_CLS_SHARED_PTRS]; // shared pointers for the arguments
} cls_t;

inline cls_t* cls();
