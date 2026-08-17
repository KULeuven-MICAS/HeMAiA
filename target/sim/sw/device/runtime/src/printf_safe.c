// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Thread-safe printf: serializes concurrent prints from different cores
// (compute cores, DM core, ...) so their output doesn't interleave on the
// UART. Mirrors printf_safe() from the KULeuven-MICAS/HeMAiA main branch's
// target/sw/shared/runtime/uart.c, adapted for this branch's printf/UART
// backend (vendored tiny-printf + putchar_chip.c instead of a custom
// vprintf()).

#include <stdarg.h>

#include "snrt.h"

int printf_safe(const char *fmt, ...) {
    mutex_tas_acquire(get_shared_lock());
    va_list ap;
    va_start(ap, fmt);
    int ret = vprintf(fmt, ap);
    va_end(ap);
    mutex_release(get_shared_lock());
    return ret;
}
