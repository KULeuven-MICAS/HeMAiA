// Force 64bit across the code
#include "o1heap64.h"
#include "uart.h"
#define O1HEAP_MAX 18446744073709551615ULL
#define FRAGMENT_SIZE_MIN (O1HEAP_ALIGNMENT * 2U)
#define FRAGMENT_SIZE_MAX ((O1HEAP_MAX >> 1U) + 1U)

#define O1HEAP_LOW32(x)  ((uint32_t)(((uint64_t)(x) >> 0) & 0xFFFFFFFF))

#define O1HEAP_FRAGMENT_PTR(x) \
    ((Fragment*) (void*) ((__riscv_xlen == 64) ? (x) : O1HEAP_LOW32(x)))
#define O1HEAP_HANDLE_PTR(x) \
    ((O1HeapInstance*) (void*) ((__riscv_xlen == 64) ? (x) : O1HEAP_LOW32(x)))


#define INSTANCE_SIZE_PADDED ((sizeof(O1HeapInstance) + O1HEAP_ALIGNMENT - 1U) & ~(O1HEAP_ALIGNMENT - 1U))

static inline uint8_t O1HEAP_CLZ_64(const uint64_t x) {
    // Implementation here
    uint64_t t = ((uint64_t) 1U) << 63U; // 64-bit size_t: MSB is at bit 63.
    uint8_t r = 0;
    while ((x & t) == 0)
    {
        t >>= 1U;  // Shift the mask one bit to the right.
        r++;
    }
    return r;  // Return the count of leading zeros.
}

static inline uint8_t log2Floor_64(const uint64_t x){
    return (uint8_t)(63U - O1HEAP_CLZ_64(x));
}


static inline uint8_t log2Ceil_64(const uint64_t x){
    // Special case: if x is 0 or 1, return 0.
    if (x <= 1U)
    {
        return 0U;
    }
    return (uint8_t)(64U - O1HEAP_CLZ_64(x - 1U));
}

static inline uint64_t pow2_64(const uint8_t power){
    return  ((uint64_t)1U) << power;
}


/// This is equivalent to pow2(log2Ceil(x)). 
static inline uint64_t roundUpToPowerOf2_64(const uint64_t x){
    return pow2_64(log2Ceil_64(x));
}

static inline void interlink(uint64_t const left, uint64_t const right)
{
    if (left != 0U)
    {
        O1HEAP_FRAGMENT_PTR(left)->header.next = right;
    }
    if (right != 0U)
    {

        O1HEAP_FRAGMENT_PTR(right)->header.prev = left;
    }
}

// THe handle_ptr_val and fragment_ptr_val are uint64_t values representing pointers
static inline void rebin(uint64_t const handle_ptr_val, uint64_t const fragment_ptr_val)
{
    const uint8_t idx = log2Floor_64(O1HEAP_FRAGMENT_PTR(fragment_ptr_val)->header.size / FRAGMENT_SIZE_MIN);
    O1HEAP_FRAGMENT_PTR(fragment_ptr_val)->next_free = O1HEAP_HANDLE_PTR(handle_ptr_val)->bins[idx];
    O1HEAP_FRAGMENT_PTR(fragment_ptr_val)->prev_free = 0U;
    if (O1HEAP_HANDLE_PTR(handle_ptr_val)->bins[idx] != 0U)
    {
        uint64_t cur_bin = O1HEAP_HANDLE_PTR(handle_ptr_val)->bins[idx];
        O1HEAP_FRAGMENT_PTR(cur_bin)->prev_free = fragment_ptr_val;
    }

    O1HEAP_HANDLE_PTR(handle_ptr_val)->bins[idx] = fragment_ptr_val;
    O1HEAP_HANDLE_PTR(handle_ptr_val)->nonempty_bin_mask |= pow2_64(idx);
}

// Remove the specified fragment from its bin
static inline void unbin(uint64_t const handle_ptr_val, uint64_t const fragment_ptr_val)
{
    const uint8_t idx = log2Floor_64(O1HEAP_FRAGMENT_PTR(fragment_ptr_val)->header.size / FRAGMENT_SIZE_MIN);

    if (O1HEAP_FRAGMENT_PTR(fragment_ptr_val)->next_free != 0U)
    {
        uint64_t next_free = O1HEAP_FRAGMENT_PTR(fragment_ptr_val)->next_free;
        O1HEAP_FRAGMENT_PTR(next_free)->prev_free = O1HEAP_FRAGMENT_PTR(fragment_ptr_val)->prev_free;
    }

    if (O1HEAP_FRAGMENT_PTR(fragment_ptr_val)->prev_free != 0U)
    {
        uint64_t prev_free = O1HEAP_FRAGMENT_PTR(fragment_ptr_val)->prev_free;
        O1HEAP_FRAGMENT_PTR(prev_free)->next_free = O1HEAP_FRAGMENT_PTR(fragment_ptr_val)->next_free;
    }

    if (O1HEAP_HANDLE_PTR(handle_ptr_val)->bins[idx] == fragment_ptr_val)
    {
        O1HEAP_HANDLE_PTR(handle_ptr_val)->bins[idx] = O1HEAP_FRAGMENT_PTR(fragment_ptr_val)->next_free;
        if (O1HEAP_HANDLE_PTR(handle_ptr_val)->bins[idx] == 0U)
        {
            O1HEAP_HANDLE_PTR(handle_ptr_val)->nonempty_bin_mask &= ~pow2_64(idx);
        }
    }
}
// -------------PUBLIC API IMPLEMENTATION ----------------

uint64_t o1heapInit(uint64_t const heap_base_addr, uint64_t const heap_size)
{
    uint64_t out = 0U;
    if ((heap_base_addr != 0U) && ((heap_base_addr % O1HEAP_ALIGNMENT) == 0U) && (heap_size >= (INSTANCE_SIZE_PADDED + FRAGMENT_SIZE_MIN)))
    {
        out = heap_base_addr;

        O1HEAP_HANDLE_PTR(out)->nonempty_bin_mask = 0UL;

        for (uint64_t i = 0U; i < NUM_BINS_MAX; i++)
        {
            O1HEAP_HANDLE_PTR(out)->bins[i] = 0UL;
        }

        uint64_t capacity = heap_size - INSTANCE_SIZE_PADDED;
        if (capacity > FRAGMENT_SIZE_MAX)
        {
            capacity = FRAGMENT_SIZE_MAX;
        }

        while ((capacity % FRAGMENT_SIZE_MIN) != 0UL)
        {
            capacity--;
        }

        uint64_t const fragment_start_addr = heap_base_addr + INSTANCE_SIZE_PADDED;
        O1HEAP_FRAGMENT_PTR(fragment_start_addr)->header.next = 0UL;
        O1HEAP_FRAGMENT_PTR(fragment_start_addr)->header.prev = 0UL;
        O1HEAP_FRAGMENT_PTR(fragment_start_addr)->header.size = capacity;
        O1HEAP_FRAGMENT_PTR(fragment_start_addr)->header.used = 0UL;
        O1HEAP_FRAGMENT_PTR(fragment_start_addr)->next_free = 0UL;
        O1HEAP_FRAGMENT_PTR(fragment_start_addr)->prev_free = 0UL;
        rebin(out, fragment_start_addr);

        O1HEAP_HANDLE_PTR(out)->diagnostics.capacity = capacity;
        O1HEAP_HANDLE_PTR(out)->diagnostics.allocated = 0UL;
        O1HEAP_HANDLE_PTR(out)->diagnostics.peak_allocated = 0UL;
        O1HEAP_HANDLE_PTR(out)->diagnostics.peak_request_size = 0UL;
        O1HEAP_HANDLE_PTR(out)->diagnostics.oom_count = 0UL;
    }

    return out;
}

uint64_t o1heapAllocate(uint64_t const handle_ptr_val, const uint64_t amount)
{
    uint64_t out = 0UL;
    if ((amount > 0UL) && (amount <= O1HEAP_HANDLE_PTR(handle_ptr_val)->diagnostics.capacity))
    {
        const uint64_t fragment_size = roundUpToPowerOf2_64(amount + O1HEAP_ALIGNMENT);
        const uint8_t optimal_bin_index = log2Ceil_64(fragment_size / FRAGMENT_SIZE_MIN);
        const uint64_t candidate_bin_mask = ~(pow2_64(optimal_bin_index) - 1U);
        const uint64_t suitable_bins = O1HEAP_HANDLE_PTR(handle_ptr_val)->nonempty_bin_mask & candidate_bin_mask;
        const uint64_t smallest_bin_mask = suitable_bins & ~(suitable_bins - 1U);

        if (smallest_bin_mask != 0)
        {
            const uint8_t bin_index = log2Floor_64(smallest_bin_mask);
            uint64_t const frag_ptr_val = O1HEAP_HANDLE_PTR(handle_ptr_val)->bins[bin_index];
            unbin(handle_ptr_val, frag_ptr_val);

            const uint64_t leftover = O1HEAP_FRAGMENT_PTR(frag_ptr_val)->header.size - fragment_size;
            O1HEAP_FRAGMENT_PTR(frag_ptr_val)->header.size = fragment_size;

            if (leftover >= FRAGMENT_SIZE_MIN)
            {
                uint64_t const new_frag_ptr_val = frag_ptr_val + fragment_size;
                O1HEAP_FRAGMENT_PTR(new_frag_ptr_val)->header.size = leftover;
                O1HEAP_FRAGMENT_PTR(new_frag_ptr_val)->header.used = 0UL;
                interlink(new_frag_ptr_val, O1HEAP_FRAGMENT_PTR(frag_ptr_val)->header.next);
                interlink(frag_ptr_val, new_frag_ptr_val);
                rebin(handle_ptr_val, new_frag_ptr_val);
            }

            O1HEAP_HANDLE_PTR(handle_ptr_val)->diagnostics.allocated += fragment_size;
            if (O1HEAP_HANDLE_PTR(handle_ptr_val)->diagnostics.peak_allocated < O1HEAP_HANDLE_PTR(handle_ptr_val)->diagnostics.allocated)
            {
                O1HEAP_HANDLE_PTR(handle_ptr_val)->diagnostics.peak_allocated = O1HEAP_HANDLE_PTR(handle_ptr_val)->diagnostics.allocated;
            }

            O1HEAP_FRAGMENT_PTR(frag_ptr_val)->header.used = 1UL;
            out = frag_ptr_val + O1HEAP_ALIGNMENT;
        }
    }

    if (amount > O1HEAP_HANDLE_PTR(handle_ptr_val)->diagnostics.peak_request_size)
    {
        O1HEAP_HANDLE_PTR(handle_ptr_val)->diagnostics.peak_request_size = amount;
    }

    if ((out == 0UL) && (amount > 0UL))
    {
        O1HEAP_HANDLE_PTR(handle_ptr_val)->diagnostics.oom_count += 1UL;
    }

    return out;
}

void o1heapFree(uint64_t const handle_ptr_val, uint64_t const pointer_val){
    if (pointer_val != 0UL) // NULL pointer is a no-op
    {
        uint64_t const frag_ptr_val = pointer_val - O1HEAP_ALIGNMENT;
        // Even if we're going to drop the fragment later, mark it free anyway to prevent double-free.
        O1HEAP_FRAGMENT_PTR(frag_ptr_val)->header.used = 0UL;
        // Update the diagnostics. It must be done before merging because it invalidates the fragment size information.
        O1HEAP_HANDLE_PTR(handle_ptr_val)->diagnostics.allocated -= O1HEAP_FRAGMENT_PTR(frag_ptr_val)->header.size;
        // Merge with siblings and insert the returned fragment into the appropriate bin and update metadata.
        uint64_t const prev_ptr_val = O1HEAP_FRAGMENT_PTR(frag_ptr_val)->header.prev;
        uint64_t const next_ptr_val = O1HEAP_FRAGMENT_PTR(frag_ptr_val)->header.next;
        const bool      join_left  = (prev_ptr_val != 0UL) && (O1HEAP_FRAGMENT_PTR(prev_ptr_val)->header.used == 0UL);
        const bool      join_right = (next_ptr_val != 0UL) && (O1HEAP_FRAGMENT_PTR(next_ptr_val)->header.used == 0UL);
        if (join_left && join_right)  // [ prev ][ this ][ next ] => [ ------- prev ------- ]
        {
            unbin(handle_ptr_val, prev_ptr_val);
            unbin(handle_ptr_val, next_ptr_val);
            O1HEAP_FRAGMENT_PTR(prev_ptr_val)->header.size += O1HEAP_FRAGMENT_PTR(frag_ptr_val)->header.size + O1HEAP_FRAGMENT_PTR(next_ptr_val)->header.size;
            O1HEAP_FRAGMENT_PTR(frag_ptr_val)->header.size = 0UL;  // Invalidate the dropped fragment headers to prevent double-free.
            O1HEAP_FRAGMENT_PTR(next_ptr_val)->header.size = 0UL;
            interlink(prev_ptr_val, O1HEAP_FRAGMENT_PTR(next_ptr_val)->header.next);
            rebin(handle_ptr_val, prev_ptr_val);
        }
        else if (join_left)  // [ prev ][ this ] => [ ------- prev ------- ]
        {
            unbin(handle_ptr_val, prev_ptr_val);
            O1HEAP_FRAGMENT_PTR(prev_ptr_val)->header.size += O1HEAP_FRAGMENT_PTR(frag_ptr_val)->header.size;
            O1HEAP_FRAGMENT_PTR(frag_ptr_val)->header.size = 0UL;
            interlink(prev_ptr_val, next_ptr_val);
            rebin(handle_ptr_val, prev_ptr_val); 
        }
        else if (join_right)  // [ this ][ next ] => [ ------- this ------- ]
        {
            unbin(handle_ptr_val, next_ptr_val);
            O1HEAP_FRAGMENT_PTR(frag_ptr_val)->header.size += O1HEAP_FRAGMENT_PTR(next_ptr_val)->header.size;
            O1HEAP_FRAGMENT_PTR(next_ptr_val)->header.size = 0UL;
            interlink(frag_ptr_val, O1HEAP_FRAGMENT_PTR(next_ptr_val)->header.next);
            rebin(handle_ptr_val, frag_ptr_val);
        }
        else
        {
            rebin(handle_ptr_val, frag_ptr_val);
        }
    }
}


uint64_t o1heapGetDiagnostics(uint64_t const handle_ptr_val){
    return handle_ptr_val + offsetof(O1HeapInstance, diagnostics);
}