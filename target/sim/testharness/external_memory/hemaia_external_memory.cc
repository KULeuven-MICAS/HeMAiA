// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "svdpi.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <new>
#include <unordered_map>

namespace {

constexpr std::size_t BeatBytes = 64;
constexpr std::size_t PageBytes = 4096;
constexpr std::size_t DpiWordBytes = sizeof(svBitVecVal);

using Page = std::array<std::uint8_t, PageBytes>;

struct ExternalMemory {
    std::uint64_t size_bytes;
    std::unordered_map<std::uint64_t, Page> pages;
};

bool valid_beat(const ExternalMemory *memory, std::uint64_t offset) {
    return memory && offset % BeatBytes == 0 &&
           offset <= memory->size_bytes - BeatBytes;
}

}  // namespace

extern "C" void *hemaia_extmem_create(unsigned long long size_bytes) {
    if (size_bytes == 0 || size_bytes % BeatBytes != 0) {
        return nullptr;
    }
    try {
        return new ExternalMemory{size_bytes, {}};
    } catch (const std::bad_alloc &) {
        return nullptr;
    }
}

extern "C" void hemaia_extmem_destroy(void *handle) {
    delete static_cast<ExternalMemory *>(handle);
}

extern "C" int hemaia_extmem_read(void *handle, unsigned long long offset,
                                 svBitVecVal *data) {
    const auto *memory = static_cast<const ExternalMemory *>(handle);
    std::fill_n(data, BeatBytes / DpiWordBytes, svBitVecVal{0});
    if (!valid_beat(memory, offset)) {
        return -1;
    }

    const auto page = memory->pages.find(offset / PageBytes);
    if (page == memory->pages.end()) {
        return 0;
    }

    const std::size_t page_offset = offset % PageBytes;
    // The DPI packs the lowest AXI byte lanes into the lowest word.
    for (std::size_t lane = 0; lane < BeatBytes; ++lane) {
        const std::size_t shift = 8 * (lane % DpiWordBytes);
        data[lane / DpiWordBytes] |=
            svBitVecVal{page->second[page_offset + lane]} << shift;
    }
    return 0;
}

extern "C" int hemaia_extmem_write(void *handle, unsigned long long offset,
                                  const svBitVecVal *data,
                                  unsigned long long strb) {
    auto *memory = static_cast<ExternalMemory *>(handle);
    if (!valid_beat(memory, offset)) {
        return -1;
    }
    if (strb == 0) {
        return 0;
    }

    try {
        // A newly allocated page is value-initialized to zero.
        auto &page = memory->pages.try_emplace(offset / PageBytes).first->second;
        const std::size_t page_offset = offset % PageBytes;
        for (std::size_t lane = 0; lane < BeatBytes; ++lane) {
            if ((strb >> lane) & 1ULL) {
                const std::size_t shift = 8 * (lane % DpiWordBytes);
                page[page_offset + lane] =
                    static_cast<std::uint8_t>(data[lane / DpiWordBytes] >> shift);
            }
        }
        return 0;
    } catch (const std::bad_alloc &) {
        return -1;
    }
}
