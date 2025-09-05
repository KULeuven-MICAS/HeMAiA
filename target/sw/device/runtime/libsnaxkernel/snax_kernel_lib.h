// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
#pragma once
#define SNAX_LIB_NAME_MAX_LEN 64
// symtab data structure
typedef struct __attribute__((packed)){
    char name[SNAX_LIB_NAME_MAX_LEN];      // function name
    uint32_t addr;                         // function addr
} snax_symbol_t;

#define SNAX_SYMTAB_END_FN_NAME "SYMTAB_END"
#define SNAX_SYMTAB_END_FN_ADDR (uint32_t)(0xBAADF00D)

#define SNAX_LIB_DEFINE __attribute__((used))
#define SNAX_SYMTAB_SECTION __attribute__((used, section(".snax_symtab")))
#define SNAX_EXPORT_FUNC(name) { #name, (uint32_t)name }
#define SNAX_SYMTAB_END { SNAX_SYMTAB_END_FN_NAME, SNAX_SYMTAB_END_FN_ADDR }

// Getter function to retrive the cls shared pointers
inline uint32_t **get_cls_shared_ptrs(){
    return (uint32_t **)&(cls()->shared_ptrs);
}