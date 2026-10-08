// Copyright 2026 KU Leuven.
// SPDX-License-Identifier: Apache-2.0
#include "host.h"
#include <stdint.h>

// Golden words from gen_scheduler_rtl_distilled_wrapper_vectors.build_case:
// eid 0..7: tokens 8..1; eid 0..15: two tokens each; both caches empty.
static const uint64_t golden8[] = {
    0x00000000a8020004ULL, 0x00804001d0008007ULL,
    0x0002000025040000ULL, 0x0080400150010006ULL,
    0x00824082ad038001ULL, 0x0081c0422d030002ULL,
    0x0080c04350018005ULL, 0x0081c043ad028003ULL,
};
static const uint64_t golden16[] = {
    0x0000000028010000ULL, 0x00000000a8010001ULL,
    0x0000000128010002ULL, 0x00000001a8010003ULL,
    0x0000000228010004ULL, 0x00000002a8010005ULL,
    0x0000000328010006ULL, 0x00000003a8010007ULL,
    0x0000000428010008ULL, 0x00000004a8010009ULL,
    0x000000052801000aULL, 0x00000005a801000bULL,
    0x000000062801000cULL, 0x00000006a801000dULL,
    0x000000072801000eULL, 0x00000007a801000fULL,
};

static uintptr_t base;
static void write_reg(unsigned offset, uint64_t value) {
    *(volatile uint64_t *)(base + offset) = value;
    asm volatile("fence iorw, iorw" ::: "memory");
}
static uint64_t read_reg(unsigned offset) {
    asm volatile("fence iorw, iorw" ::: "memory");
    uint64_t value = *(volatile uint64_t *)(base + offset);
    asm volatile("fence iorw, iorw" ::: "memory");
    return value;
}

static int run_job(unsigned count) {
    const uint64_t *golden = count == 8 ? golden8 : golden16;
    unsigned tasks = 0, refills = 0, events = 0;
    write_reg(0x00, ((uint64_t)count << 16) | 0x8080);
    write_reg(0x38, count == 8 ? 0x0000810204140824ULL : 0x20100020ULL);
    write_reg(0x08, count == 8 ? 0x8605840682078008ULL : 0x8602840282028002ULL);
    write_reg(0x10, count == 8 ? 0x8e018c028a038804ULL : 0x8e028c028a028802ULL);
    if (count == 16) {
        write_reg(0x18, 0x9a029c029e029002ULL);
        write_reg(0x40, 0x0000000096029802ULL);
    }
    // Delay consumption so the scheduler can accumulate queued outputs.
    for (volatile unsigned delay = 0; delay < 2000; ++delay)
        asm volatile("nop");
    for (;;) {
        uint64_t event = read_reg(0x28);
        unsigned batch = (event >> 8) & 15;
        if (++events > 256 || batch > 8) return 1;
        if (event & 2) {
            if (count != 16 || refills || ((event >> 2) & 7) != 2 ||
                ((event >> 5) & 7) != 0) return 2;
            write_reg(0x20, 0x94029202ULL);
            ++refills;
        }
        for (unsigned i = 0; i < batch; ++i) {
            uint64_t got = read_reg(0x30);
            if (tasks >= count || got != golden[tasks]) {
                printf("FAIL count=%u task=%u got=%016lx\r\n",
                       count, tasks, (unsigned long)got);
                return 3;
            }
            ++tasks;
        }
        if (event & 1) break;
    }
    if (tasks != count || refills != (count == 16)) return 4;
    printf("PASS scheduler count=%u tasks=%u refills=%u events=%u\r\n",
           count, tasks, refills, events);
    return 0;
}

int main(void) {
    uintptr_t chip = (uintptr_t)get_current_chip_baseaddress();
    init_uart(chip, 32, 1);
    base = chip + 0x05010000UL;
    printf("Scheduler MMIO smoke start\r\n");
    int result = run_job(8);
    if (!result) result = run_job(16);
    if (result) {
        printf("Scheduler MMIO smoke FAIL code=%d\r\n", result);
        return result;
    }
    printf("Scheduler MMIO smoke PASS: two consecutive jobs, golden task match\r\n");
    return 0;
}
