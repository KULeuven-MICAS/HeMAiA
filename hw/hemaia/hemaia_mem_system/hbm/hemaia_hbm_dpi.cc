// Copyright 2026 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
//
// Storage behind hemaia_hbm_model.sv (simulation only).
//
// An HBM image is tens of GiB. An SV array of that size cannot be elaborated, and
// $readmemh of it would take hours, so the bytes live here instead:
//
//   * FILE SEGMENTS. A loaded file is mmap()ed MAP_PRIVATE, never read or copied.
//     The OS pages in only what the simulation touches, and a write from the RTL
//     lands in a private copy-on-write page, so the file on disk is never
//     modified and several memchips can map the same file independently.
//   * SPARSE PAGES. Everything outside a file segment is 64 KiB pages allocated on
//     first write. A read of a byte that was never written returns zero.
//
// One store per model instance, addressed through the chandle the model gets from
// hemaia_hbm_create(). All offsets are relative to the HBM base.

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "svdpi.h"

namespace {

constexpr uint64_t kPageBits = 16;
constexpr uint64_t kPageSize = 1ull << kPageBits;
// The widest bus the model supports (512 bit); the SV side passes bit [511:0].
constexpr uint32_t kMaxLineBytes = 64;

struct Segment {
    uint64_t off;       // first byte, relative to the HBM base
    uint64_t len;       // bytes backed by the file
    uint8_t *data;      // MAP_PRIVATE mapping of the whole file
    std::string path;
};

struct HbmStore {
    std::string name;
    uint64_t size;
    std::vector<Segment> segs;  // sorted by off, never overlapping
    std::unordered_map<uint64_t, std::unique_ptr<uint8_t[]>> pages;

    ~HbmStore() { clear(); }

    void clear() {
        for (auto &s : segs) munmap(s.data, s.len);
        segs.clear();
        pages.clear();
    }

    // The segment holding `off`, or nullptr. `next` is set to the start of the
    // first segment above `off` (or UINT64_MAX), which bounds a page access.
    const Segment *find(uint64_t off, uint64_t &next) const {
        auto it = std::upper_bound(
            segs.begin(), segs.end(), off,
            [](uint64_t o, const Segment &s) { return o < s.off; });
        next = (it == segs.end()) ? UINT64_MAX : it->off;
        if (it == segs.begin()) return nullptr;
        const Segment &s = *std::prev(it);
        return (off < s.off + s.len) ? &s : nullptr;
    }

    // Length of the run starting at `off` that stays inside one segment or one
    // page, capped at `n`. `seg` is that segment, or nullptr for a page.
    uint64_t run(uint64_t off, uint64_t n, const Segment *&seg) const {
        uint64_t next;
        seg = find(off, next);
        if (seg) return std::min(n, seg->off + seg->len - off);
        uint64_t page_end = ((off >> kPageBits) + 1) << kPageBits;
        return std::min({n, page_end - off, next - off});
    }

    void read(uint64_t off, uint64_t n, uint8_t *dst) const {
        while (n) {
            const Segment *seg;
            uint64_t k = run(off, n, seg);
            if (seg) {
                std::memcpy(dst, seg->data + (off - seg->off), k);
            } else {
                auto p = pages.find(off >> kPageBits);
                if (p == pages.end())
                    std::memset(dst, 0, k);
                else
                    std::memcpy(dst, p->second.get() + (off & (kPageSize - 1)), k);
            }
            off += k, dst += k, n -= k;
        }
    }

    // `strb` holds one bit per byte of `src`, LSB first, as SV packs bit [N-1:0].
    void write(uint64_t off, uint64_t n, const uint8_t *src, const uint8_t *strb) {
        uint64_t i = 0;
        while (i < n) {
            const Segment *seg;
            uint64_t k = run(off + i, n - i, seg);
            uint8_t *dst;
            if (seg) {
                dst = seg->data + (off + i - seg->off);
            } else {
                bool any = false;
                for (uint64_t j = i; j < i + k && !any; j++)
                    any = (strb[j >> 3] >> (j & 7)) & 1;
                if (!any) {
                    i += k;
                    continue;
                }
                auto &page = pages[(off + i) >> kPageBits];
                if (!page) {
                    page.reset(new uint8_t[kPageSize]);
                    std::memset(page.get(), 0, kPageSize);
                }
                dst = page.get() + ((off + i) & (kPageSize - 1));
            }
            for (uint64_t j = 0; j < k; j++)
                if ((strb[(i + j) >> 3] >> ((i + j) & 7)) & 1) dst[j] = src[i + j];
            i += k;
        }
    }

    // Map `path` at `off`. Returns false (and says why) on any error.
    bool load(const std::string &path, uint64_t off) {
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            std::fprintf(stderr, "[HBM %s] cannot open %s: %s\n", name.c_str(),
                         path.c_str(), std::strerror(errno));
            return false;
        }
        struct stat st;
        if (fstat(fd, &st) != 0) {
            std::fprintf(stderr, "[HBM %s] cannot stat %s: %s\n", name.c_str(),
                         path.c_str(), std::strerror(errno));
            close(fd);
            return false;
        }
        uint64_t len = static_cast<uint64_t>(st.st_size);
        if (len == 0) {
            std::fprintf(stderr, "[HBM %s] %s is empty, nothing loaded\n",
                         name.c_str(), path.c_str());
            close(fd);
            return true;
        }
        if (off > size || len > size - off) {
            std::fprintf(stderr,
                         "[HBM %s] %s (0x%lx bytes at offset 0x%lx) does not fit "
                         "the 0x%lx-byte HBM\n",
                         name.c_str(), path.c_str(), (unsigned long)len,
                         (unsigned long)off, (unsigned long)size);
            close(fd);
            return false;
        }
        for (const auto &s : segs) {
            if (off < s.off + s.len && s.off < off + len) {
                std::fprintf(stderr,
                             "[HBM %s] %s [0x%lx, 0x%lx) overlaps %s [0x%lx, "
                             "0x%lx)\n",
                             name.c_str(), path.c_str(), (unsigned long)off,
                             (unsigned long)(off + len), s.path.c_str(),
                             (unsigned long)s.off, (unsigned long)(s.off + s.len));
                close(fd);
                return false;
            }
        }
        void *p = mmap(nullptr, len, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_NORESERVE, fd, 0);
        close(fd);
        if (p == MAP_FAILED) {
            std::fprintf(stderr, "[HBM %s] cannot map %s: %s\n", name.c_str(),
                         path.c_str(), std::strerror(errno));
            return false;
        }
        // Bytes written before the load (there should be none: the harness clears
        // first) would otherwise hide under the segment and resurface after the
        // next clear. Drop every page the segment now covers.
        for (uint64_t pg = off >> kPageBits; pg <= (off + len - 1) >> kPageBits; pg++) {
            uint64_t pb = pg << kPageBits;
            if (pb >= off && pb + kPageSize <= off + len) pages.erase(pg);
        }
        Segment s{off, len, static_cast<uint8_t *>(p), path};
        segs.insert(std::upper_bound(segs.begin(), segs.end(), s,
                                     [](const Segment &a, const Segment &b) {
                                         return a.off < b.off;
                                     }),
                    s);
        std::printf("[HBM %s] mapped %s at offset 0x%lx (0x%lx bytes)\n",
                    name.c_str(), path.c_str(), (unsigned long)off,
                    (unsigned long)len);
        return true;
    }
};

std::string dirname_of(const std::string &path) {
    auto slash = path.find_last_of('/');
    return slash == std::string::npos ? std::string(".") : path.substr(0, slash);
}

// strtoull that also accepts '_' digit separators (0x1_0000_0000).
bool parse_u64(std::string tok, uint64_t &val) {
    tok.erase(std::remove(tok.begin(), tok.end(), '_'), tok.end());
    if (tok.empty()) return false;
    char *end = nullptr;
    errno = 0;
    val = std::strtoull(tok.c_str(), &end, 0);
    return errno == 0 && end && *end == '\0';
}

}  // namespace

extern "C" {

void *hemaia_hbm_create(const char *name, unsigned long long size) {
    auto *h = new HbmStore();
    h->name = name ? name : "hbm";
    h->size = size;
    return h;
}

void hemaia_hbm_clear(void *handle) { static_cast<HbmStore *>(handle)->clear(); }

// 0 on success, -1 on error.
int hemaia_hbm_load_file(void *handle, const char *path, unsigned long long off) {
    return static_cast<HbmStore *>(handle)->load(path, off) ? 0 : -1;
}

// Load every file a manifest lists. One entry per line:
//
//     <offset>  <file>  [chip=<id>]
//
// <offset> is relative to the HBM base (C syntax, '_' separators allowed), <file>
// is relative to the manifest's own directory unless absolute, and an entry with
// chip=<id> is loaded only into the memchip whose chip id is <id>. '#' starts a
// comment. Returns the number of files mapped, -1 when the manifest does not exist
// (not an error: most workloads have no HBM image), -2 on any error.
int hemaia_hbm_load_manifest(void *handle, const char *manifest, int chip_id) {
    auto *h = static_cast<HbmStore *>(handle);
    std::ifstream in(manifest);
    if (!in) return -1;
    std::string dir = dirname_of(manifest);
    std::string line;
    int lineno = 0, loaded = 0;
    while (std::getline(in, line)) {
        lineno++;
        auto hash = line.find('#');
        if (hash != std::string::npos) line.resize(hash);
        std::istringstream ss(line);
        std::string off_tok, file, opt;
        if (!(ss >> off_tok)) continue;
        uint64_t off;
        if (!parse_u64(off_tok, off) || !(ss >> file)) {
            std::fprintf(stderr, "[HBM %s] %s:%d: expected '<offset> <file> [chip=<id>]'\n",
                         h->name.c_str(), manifest, lineno);
            return -2;
        }
        bool mine = true;
        while (ss >> opt) {
            uint64_t id;
            if (opt.rfind("chip=", 0) == 0 && parse_u64(opt.substr(5), id)) {
                mine = mine && (static_cast<int>(id) == chip_id);
            } else {
                std::fprintf(stderr, "[HBM %s] %s:%d: unknown option '%s'\n",
                             h->name.c_str(), manifest, lineno, opt.c_str());
                return -2;
            }
        }
        if (!mine) continue;
        if (file[0] != '/') file = dir + "/" + file;
        if (!h->load(file, off)) return -2;
        loaded++;
    }
    return loaded;
}

void hemaia_hbm_read(void *handle, unsigned long long off, unsigned int len,
                     svBitVecVal *data) {
    uint8_t buf[kMaxLineBytes] = {0};
    static_cast<HbmStore *>(handle)->read(off, std::min(len, kMaxLineBytes), buf);
    std::memcpy(data, buf, kMaxLineBytes);
}

void hemaia_hbm_write(void *handle, unsigned long long off, unsigned int len,
                      const svBitVecVal *data, const svBitVecVal *strb) {
    static_cast<HbmStore *>(handle)->write(
        off, std::min(len, kMaxLineBytes),
        reinterpret_cast<const uint8_t *>(data),
        reinterpret_cast<const uint8_t *>(strb));
}

// Write [off, off+len) to `path`. 0 on success, -1 on error.
int hemaia_hbm_dump(void *handle, const char *path, unsigned long long off,
                    unsigned long long len) {
    auto *h = static_cast<HbmStore *>(handle);
    if (off > h->size || len > h->size - off) {
        std::fprintf(stderr, "[HBM %s] dump [0x%llx, +0x%llx) is outside the HBM\n",
                     h->name.c_str(), off, len);
        return -1;
    }
    FILE *f = std::fopen(path, "wb");
    if (!f) {
        std::fprintf(stderr, "[HBM %s] cannot create %s: %s\n", h->name.c_str(),
                     path, std::strerror(errno));
        return -1;
    }
    std::vector<uint8_t> buf(1 << 20);
    while (len) {
        uint64_t k = std::min<uint64_t>(len, buf.size());
        h->read(off, k, buf.data());
        if (std::fwrite(buf.data(), 1, k, f) != k) {
            std::fclose(f);
            return -1;
        }
        off += k, len -= k;
    }
    return std::fclose(f) == 0 ? 0 : -1;
}

// Host memory held by pages written outside any file segment.
unsigned long long hemaia_hbm_sparse_bytes(void *handle) {
    return static_cast<HbmStore *>(handle)->pages.size() * kPageSize;
}

}  // extern "C"
