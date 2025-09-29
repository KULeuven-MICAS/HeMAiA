// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
extern __thread cls_t* _cls_ptr;

inline cls_t* cls() { return _cls_ptr; }

// Getter function to retrive the cls shared pointers
inline uint32_t **get_cls_shared_ptrs() {
    return (uint32_t **)&(cls()->shared_ptrs);
}
