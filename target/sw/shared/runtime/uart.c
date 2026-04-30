// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
#include "uart.h"

static uintptr_t base_address;

static inline int out_ch(char c) {
    print_char(base_address, c);
    return 0;
}

// Helper for 64-bit division on 32-bit architectures without libgcc
static uint32_t umod64(uint64_t *n, uint32_t base) {
    uint32_t rem;
    if (base == 16) {
        rem = (uint32_t)(*n & 0xF);
        *n >>= 4;
    } else if (base == 10) {
        // Simple long division by 10
        // To avoid linking __udivdi3, we implement division manually
        uint64_t q = 0;
        uint32_t r = 0;
        for (int i = 63; i >= 0; i--) {
            r = (r << 1) | (uint32_t)((*n >> i) & 1);
            if (r >= 10) {
                r -= 10;
                q |= (1ULL << i);
            }
        }
        *n = q;
        rem = r;
    } else {
        rem = 0;
        *n = 0; 
    }
    return rem;
}

static int print_unsigned(uint64_t value, uint32_t base, int uppercase,
                          int width, int zero_pad) {
    char buf[64];  // enough for 64-bit entries
    int idx = 0;

    // 1. Convert to string in reverse order
    do {
        uint32_t digit = umod64(&value, base);
        buf[idx++] = (digit < 10) ? ('0' + digit)
                                  : ((uppercase ? 'A' : 'a') + (digit - 10));
    } while (value && idx < (int)sizeof(buf));

    // 2. Padding (if width > length)
    int pad_len = (width > idx) ? (width - idx) : 0;
    for (int i = 0; i < pad_len; ++i) out_ch(zero_pad ? '0' : ' ');

    // 3. Output in correct order
    int count = pad_len;
    while (idx--) count += out_ch(buf[idx]);

    return count;
}

// ---------------------------------------------------------------------------
// print_signed() – handles sign then delegates to print_unsigned().
// ---------------------------------------------------------------------------
static int print_signed(int64_t value, int width, int zero_pad) {
    int count = 0;
    uint64_t abs_val;
    if (value < 0) {
        count += out_ch('-');
        abs_val = (uint64_t)(-(value + 1)) + 1; // Handle INT64_MIN safely
        if (width) width--;  // sign consumed one slot
    } else {
        abs_val = (uint64_t)value;
    }
    count += print_unsigned(abs_val, 10, 0, width, zero_pad);
    return count;
}

// ---------------------------------------------------------------------------
// print_float() – simple %f implementation without libm/libgcc dependency
// Prints float with 'precision' decimal digits (default 6).
// Handles negative, zero, inf, nan.  No exponential notation.
// ---------------------------------------------------------------------------
static int print_float(double val, int width, int precision) {
    int count = 0;

    // Handle special cases
    // NaN: exponent all 1s, mantissa non-zero
    // Use bit-level check to avoid pulling in fpclassify/isnan from libm
    union { double d; uint64_t u; } pun;
    pun.d = val;
    uint64_t bits = pun.u;
    uint64_t exp_mask = 0x7FF0000000000000ULL;
    uint64_t man_mask = 0x000FFFFFFFFFFFFFULL;
    if ((bits & exp_mask) == exp_mask) {
        if (bits & man_mask) {
            // NaN
            const char *s = "nan";
            for (int i = 0; s[i]; i++) count += out_ch(s[i]);
            return count;
        } else {
            // Inf
            if (bits >> 63) count += out_ch('-');
            const char *s = "inf";
            for (int i = 0; s[i]; i++) count += out_ch(s[i]);
            return count;
        }
    }

    // Sign
    if (val < 0.0) {
        count += out_ch('-');
        val = -val;
        if (width) width--;
    }

    // Default precision
    if (precision < 0) precision = 6;
    if (precision > 9) precision = 9;  // cap to avoid overflow in multiplier

    // Compute multiplier for decimal digits: 10^precision
    uint32_t mult = 1;
    for (int i = 0; i < precision; i++) mult *= 10;

    // Split into integer and fractional parts
    // Add 0.5 * 10^(-precision) for rounding
    double round_add = 0.5;
    for (int i = 0; i < precision; i++) round_add /= 10.0;
    val += round_add;

    // Integer part: extract via repeated subtraction to avoid __fixunsdfdi
    // For values up to ~4 billion this works fine
    uint32_t int_part = 0;
    while (val >= 1.0) {
        if (val >= 1000000000.0) { int_part += 1000000000; val -= 1000000000.0; }
        else if (val >= 100000000.0) { int_part += 100000000; val -= 100000000.0; }
        else if (val >= 10000000.0) { int_part += 10000000; val -= 10000000.0; }
        else if (val >= 1000000.0) { int_part += 1000000; val -= 1000000.0; }
        else if (val >= 100000.0) { int_part += 100000; val -= 100000.0; }
        else if (val >= 10000.0) { int_part += 10000; val -= 10000.0; }
        else if (val >= 1000.0) { int_part += 1000; val -= 1000.0; }
        else if (val >= 100.0) { int_part += 100; val -= 100.0; }
        else if (val >= 10.0) { int_part += 10; val -= 10.0; }
        else { int_part += 1; val -= 1.0; }
    }

    // Fractional part: multiply by 10^precision, extract as integer
    // Use repeated multiply-by-10 to avoid __fixunsdfdi on large doubles
    uint32_t frac_part = 0;
    double frac = val;
    for (int i = 0; i < precision; i++) frac *= 10.0;
    // Now frac should be < mult, extract via subtraction
    while (frac >= 1.0) {
        if (frac >= 1000000000.0) { frac_part += 1000000000; frac -= 1000000000.0; }
        else if (frac >= 100000000.0) { frac_part += 100000000; frac -= 100000000.0; }
        else if (frac >= 10000000.0) { frac_part += 10000000; frac -= 10000000.0; }
        else if (frac >= 1000000.0) { frac_part += 1000000; frac -= 1000000.0; }
        else if (frac >= 100000.0) { frac_part += 100000; frac -= 100000.0; }
        else if (frac >= 10000.0) { frac_part += 10000; frac -= 10000.0; }
        else if (frac >= 1000.0) { frac_part += 1000; frac -= 1000.0; }
        else if (frac >= 100.0) { frac_part += 100; frac -= 100.0; }
        else if (frac >= 10.0) { frac_part += 10; frac -= 10.0; }
        else { frac_part += 1; frac -= 1.0; }
    }

    // Print integer part
    count += print_unsigned((uint64_t)int_part, 10, 0, 0, 0);

    if (precision > 0) {
        count += out_ch('.');
        // Print fractional part with leading zeros
        count += print_unsigned((uint64_t)frac_part, 10, 0, precision, 1);
    }

    return count;
}

// ---------------------------------------------------------------------------
// printf() – minimal subset (supports %d %u %x %lx %ld %lu %s %c %f %%)
// ---------------------------------------------------------------------------
int vprintf(const char *fmt, va_list ap) {
    base_address = (uintptr_t)get_current_chip_baseaddress();
    int total = 0;

    while (*fmt) {
        if (*fmt != '%') {
            total += out_ch(*fmt++);
            continue;
        }

        // We are at a '%'
        ++fmt;
        int zero_pad = 0;
        int width = 0;

        // ---- Parse flags ----
        if (*fmt == '0') {
            zero_pad = 1;
            ++fmt;
        }

        // ---- Parse width ----
        while (*fmt >= '0' && *fmt <= '9') {
            width = width * 10 + (*fmt++ - '0');
        }

        // ---- Conversion specifier ----
        char spec = *fmt ? *fmt++ : '\0';
        switch (spec) {
            case 'c': {
                char c = (char)va_arg(ap, int);
                total += out_ch(c);
            } break;

            case 's': {
                const char *s = va_arg(ap, const char *);
                if (!s) s = "(null)";
                int len = 0;
                while (s[len]) len++;
                // Width & padding
                if (width > len) {
                    for (int i = 0; i < width - len; ++i) total += out_ch(' ');
                }
                for (int i = 0; s[i]; ++i) total += out_ch(s[i]);
            } break;

            case 'd':
                total += print_signed(va_arg(ap, int), width, zero_pad);
                break;

            case 'u':
                total += print_unsigned(va_arg(ap, unsigned int), 10, 0, width,
                                        zero_pad);
                break;
            case 'p':
                total += print_unsigned(va_arg(ap, unsigned int), 16, 0, width,
                                        zero_pad);
                break;
            case 'x':
                total += print_unsigned(va_arg(ap, unsigned int), 16, 0, width,
                                        zero_pad);
                break;

            case 'X':
                total += print_unsigned(va_arg(ap, unsigned int), 16, 1, width,
                                        zero_pad);
                break;

            case 'f': {
                // %f or %<width>.<prec>f
                // precision was parsed as width if '.' was present
                // Re-parse: check if we consumed a '.' during width parsing
                // Since our parser doesn't handle '.', we pass width as-is
                // and use default precision 6. For %.Nf, the '.' stops the
                // width parse, so width=0 and precision is lost. Handle '.'
                // by checking if the character before 'f' was a '.'.
                // Simple approach: just use default precision.
                double fval = va_arg(ap, double);
                total += print_float(fval, width, -1);
            } break;

            case '.': {
                // Handle %.<prec>f  (we hit '.' after '%')
                int prec = 0;
                while (*fmt >= '0' && *fmt <= '9') {
                    prec = prec * 10 + (*fmt++ - '0');
                }
                char next_spec = *fmt ? *fmt++ : '\0';
                if (next_spec == 'f') {
                    double fval = va_arg(ap, double);
                    total += print_float(fval, width, prec);
                } else {
                    total += out_ch('%');
                    total += out_ch('.');
                    if (next_spec) total += out_ch(next_spec);
                }
            } break;

            case 'l': {  // Handle %lx, %lX, %ld, %lu
                char next = *fmt ? *fmt++ : '\0';
                if (next == 'x' || next == 'X') {
                    uint64_t val = va_arg(ap, uint64_t);
                    // Use updated print_unsigned which handles 64-bit
                    // Note: original code forced 0-padding for split hi/lo.
                    // New code uses whatever width/zero_pad is set by user, or default behavior.
                    // If user wants %016lx, they should specify it in format string.
                    total += print_unsigned(val, 16, (next == 'X'), width, zero_pad);
                } else if (next == 'd') {
                    int64_t val = va_arg(ap, int64_t);
                    total += print_signed(val, width, zero_pad);
                } else if (next == 'u') {
                    uint64_t val = va_arg(ap, uint64_t);
                    total += print_unsigned(val, 10, 0, width, zero_pad);
                } else {
                    total += out_ch('%');
                    total += out_ch('l');
                    if (next) total += out_ch(next);
                }
            } break;

            case '%':
                total += out_ch('%');
                break;

            default:  // Unknown spec => output verbatim
                total += out_ch('%');
                total += out_ch(spec);
                break;
        }
    }

    return total;
}

int printf(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int ret = vprintf(fmt, ap);
    va_end(ap);
    return ret;
}


int printf_safe(const char *fmt, ...) {
    // This should be used mainly for the debug printing
    // The thread-safe version printf
    // It uses the mutex lock in the comm_buffer to ensure only one thread prints at a time
    // The threads means both the host core and the device cores
    // Notice this function assumes there is one uart in each chiplets, so it is not safe when multiple chiplets are printing to one uart at the same time
    mutex_tas_acquire(get_shared_lock());
    va_list ap;
    va_start(ap, fmt);
    int ret = vprintf(fmt, ap);
    va_end(ap);
    mutex_release(get_shared_lock());
    return ret;
}


// Scanf implementation
static int peek;
static int getc_in(void)  // fetch next char or peeked
{
    int c = peek;
    if (c >= 0) {
        peek = -1;
        return c;
    } else
        return scan_char(base_address);
}
static void ungetc_in(int c)  // push char back
{
    peek = c;
}
static int is_space(int c)  // simple isspace()
{
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
           c == '\v';
}
static int hex_val(int c)  // returns 0‑15 or -1
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}
// Read unsigned integer of given base, max "width" chars (0 = unlimited).
static int read_uint(uint32_t *out, int base, int width) {
    uint32_t val = 0;
    int got = 0;
    while (1) {
        if (width && got >= width) break;
        int c = getc_in();
        int d;
        if (base == 10) {
            if (c < '0' || c > '9') {
                ungetc_in(c);
                break;
            }
            d = c - '0';
        } else {
            d = hex_val(c);
            if (d < 0 || d >= base) {
                ungetc_in(c);
                break;
            }
        }
        val = val * base + (uint32_t)d;
        ++got;
    }
    if (!got) return 0;
    *out = val;
    return 1;
}
// Read signed decimal (optionally width), sets *out, returns 1 on success.
static int read_int(int32_t *out, int width) {
    int c, sign = 1;
    do {
        c = getc_in();
    } while (is_space(c));
    if (c == '+' || c == '-') {
        if (c == '-') sign = -1;
        width = width ? width - 1 : 0;
    } else
        ungetc_in(c);
    uint32_t u;
    if (!read_uint(&u, 10, width)) return 0;
    *out = (int32_t)u * sign;
    return 1;
}
// Read 0x / 0X prefix if present.
static void eat_0x_prefix(int *width) {
    int c1 = getc_in();
    int c2 = getc_in();
    if (c1 == '0' && (c2 == 'x' || c2 == 'X')) {
        if (*width > 2)
            *width -= 2;
        else if (*width)
            *width = 0;
    } else {
        ungetc_in(c2);
        ungetc_in(c1);
    }
}
// Read string: stops at space or width limit.
static int read_str(char *dst, int width) {
    int c;  // skip leading spaces
    do {
        c = getc_in();
    } while (is_space(c));
    if (c < 0) return 0;
    int cnt = 0;
    while (c >= 0 && !is_space(c) && (width == 0 || cnt < width - 1)) {
        dst[cnt++] = (char)c;
        c = getc_in();
    }
    dst[cnt] = '\0';
    if (c >= 0) ungetc_in(c);
    return cnt ? 1 : 0;
}
// Skip whitespace in both format and input.
static void skip_ws_fmt_and_input(void) {
    int c;
    do c = getc_in();
    while (is_space(c));
    ungetc_in(c);
}

int scanf(const char *fmt, ...) {
    base_address = (uintptr_t)get_current_chip_baseaddress();
    peek = -1;                  // simple one‑char push‑back
    va_list ap;
    va_start(ap, fmt);
    int assigned = 0;
    while (*fmt) {
        if (is_space(*fmt)) {  // space in format → eat any amount of input ws
            while (is_space(*fmt)) ++fmt;
            skip_ws_fmt_and_input();
            continue;
        }
        if (*fmt != '%') {  // literal char must match
            int c = getc_in();
            if (c != *fmt++) {
                ungetc_in(c);
                break;
            }
            continue;
        }
        // --- conversion specifier ---
        ++fmt;
        int width = 0;
        while (*fmt >= '0' && *fmt <= '9') width = width * 10 + (*fmt++ - '0');
        char sp = *fmt ? *fmt++ : '\0';
        switch (sp) {
            case 'c': {
                int *p = va_arg(ap, int *);
                int c = getc_in();
                if (c < 0) goto end;
                *p = c;
                ++assigned;
            } break;
            case 's': {
                char *p = va_arg(ap, char *);
                if (!read_str(p, width)) goto end;
                ++assigned;
            } break;
            case 'd': {
                int32_t *p = va_arg(ap, int32_t *);
                if (!read_int(p, width)) goto end;
                ++assigned;
            } break;
            case 'u': {
                skip_ws_fmt_and_input();
                uint32_t v;
                if (!read_uint(&v, 10, width)) goto end;
                uint32_t *p = va_arg(ap, uint32_t *);
                *p = v;
                ++assigned;
            } break;
            case 'x':
            case 'X': {
                skip_ws_fmt_and_input();
                if (width) eat_0x_prefix(&width);
                uint32_t v;
                if (!read_uint(&v, 16, width)) goto end;
                uint32_t *p = va_arg(ap, uint32_t *);
                *p = v;
                ++assigned;
            } break;
            case '%': {
                int c = getc_in();
                if (c != '%') {
                    ungetc_in(c);
                    goto end;
                }
            } break;
            default:  // unknown spec – abort
                goto end;
        }
    }
end:
    va_end(ap);
    return assigned;
}
