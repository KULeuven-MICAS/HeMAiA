# DeepSeek-V2-Lite on sixteen compute chiplets — planned

No workload yet. The mapping and the platform changes it needs are in
[`docs/dsv2_multichiplet_plan.md`](../../../../../../../../../docs/dsv2_multichiplet_plan.md):
sixteen compute chiplets with four clusters each, every compute chiplet with its own memory-chiplet
link, and every weight — every routed expert included — split over all sixteen links (tensor
parallel), with the attention split by keys and two small all-reduces per layer.

Phase 1 of that plan builds on today's RTL: 16 compute chiplets in one row, each with a memory
chiplet on the south edge. Its first workload here will be a two-chiplet slice of it (2 compute +
2 memory chiplets), reusing the stage structure of [`../two_chiplet`](../two_chiplet): one
directory per stage, each building on the one before.
