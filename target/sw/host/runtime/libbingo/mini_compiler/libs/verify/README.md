# verify/ — host-side verification

| file | |
|---|---|
| `checks.py` | Builds the check NODE: places the kernel on the host core, orders it, and pairs it with the readback that must precede it. |

The comparison itself is a property of the kernel ABI, so the args classes live in
[`../../kernels/kernel_host.py`](../../kernels/kernel_host.py) — one per precision. This
file has no ABI knowledge, which is what keeps it from drifting, and a workload that does
not use `libs` still gets the named args.

## Named by precision, not numbered

`__host_bingo_kernel_check_result` selects its comparison with an integer, and six are
defined. `check_type=5` silently couples three decisions: which comparison runs, that
`num_elements` counts int32s rather than bytes, and that `tolerance` is a **ratio** and not
an absolute. Get any of the three wrong and nothing faults — the wrong element size
compares a prefix of the buffer and passes, and an absolute tolerance where a ratio was
wanted passes everything.

So each wrapper names its tolerance after what the kernel does with it: `tol` where the
comparison is absolute, `rtol` where it is a ratio.

```python
checks.check_int32_rel(ctx, "Check_o_c0", golden=h["o"], got=l3_o,
                       elems=512, rtol=0.02, after=st_o, label="fa_o_c0")
```

`check_out(dtype=…)` picks the comparison from the port's own dtype, so a block that
changes precision cannot leave a stale checker behind.

## Readback and check are two nodes on purpose

The host cannot read L1, so every check is a store followed by a compare. They are returned
separately because they order differently: downstream compute that overwrites the buffer
must wait on the **store**, while anything that only needs the verdict waits on the
**check**. Collapsing them would force the stricter of the two everywhere — and a check
that owned a hidden store could not be ordered against the compute at all, which is exactly
the RAW race that made FA's `Store_o` read the accumulator one PV too early.

`label` is what the UART prints on a mismatch and defaults to the node name. Keep them
separate where they say different things: the node name locates the check in the graph
(`Check_o_c2`), the label names the quantity a reader is looking for (`fa_o_c2`).
