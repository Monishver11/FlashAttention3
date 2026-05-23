### FlashAttention3-worklog

Companion repo for the FlashAttention-3 worklog blog. Incremental kernel implementations in CuTe Python DSL on H100, from a naive baseline through FP8 with incoherent processing.

Blog post: [link](https://monishver11.github.io/blog/2026/fa3-worklog/)

#### Hardware and software

All measurements in the blog use the following stack:

- **GPU:** H100 SXM <!-- TODO: confirm 80GB HBM3, clocks, persistence mode -->
- **CUDA toolkit:** <!-- TODO: `nvcc --version` -->
- **Driver version:** <!-- TODO: `nvidia-smi` -->
- **CUTLASS Python DSL (`nvidia-cutlass-dsl`):** <!-- TODO: `uv pip show nvidia-cutlass-dsl` -->
- **PyTorch:** <!-- TODO -->
- **Python:** <!-- TODO -->
- **OS / kernel:** <!-- TODO -->
- **NCU version:** <!-- TODO -->

<!-- TODO: clock locking and persistence mode. If clocks are locked (e.g. via `nvidia-smi -lgc <freq>` and `nvidia-smi -pm 1`), note the locked frequency. If not, add a reproducibility caveat. -->

#### Repo structure

```
FlashAttention3/
├── pyproject.toml          # uv-managed dependencies
├── uv.lock
├── .gitignore
├── utils.py                # shared helpers: tensor creation, CUTE views
├── bench.py                # benchmark harness, kernel dispatch, sweep loop
├── ref_check.py            # correctness check vs PyTorch SDPA
└── kernels/                # one kernel per file
    ├── __init__.py
    ├── k1.py
    ├── k2.py
    └── ...
```

- **`utils.py`** holds the helpers used by every kernel: dtype constants (`INPUT_DTYPE`, `OUTPUT_DTYPE`, `ACC_DTYPE`), the `(B, N, d)` PyTorch tensor to `(N, d, B)` CUTE view conversion, and `make_qkvo` which produces Q, K, V, and O tensors in the layouts the kernels expect.
- **`bench.py`** is the harness: loads a kernel module by name, sweeps over sequence length, head dim, and causal flag, prints timing and TFLOPS.
- **`ref_check.py`** is the correctness checker: runs PyTorch SDPA on the same inputs and reports max absolute error, max relative error, and count of elements outside tolerance.

Each kernel file in `kernels/` exports two functions:

- `compile_kernel(B, N, d, causal, tensors)`: called once, returns a compiled CuTe object.
- `run_kernel(compiled, tensors)`: called for warmup and benchmark iterations, writes the result into `tensors["o_gpu"]`.

The harness is agnostic to kernel internals; it just calls these two entry points.

#### Workload shape and conventions

The benchmark sweep is parameterized by:

- **Sequence length** $N$: $\{512, 1024, 2048, 4096, 8192, 16384\}$
- **Head dimension** $d$: $\{64, 128\}$
- **Causal flag**: $\{0, 1\}$ (non-causal and causal)

Two workload-size constants pin the total work per sweep configuration:

- `HIDDEN_DIM = 2048`: total feature dimension, split across heads. Number of heads is `HIDDEN_DIM / d`, so 32 for `d = 64` and 16 for `d = 128`.
- `TOTAL_TOKENS = 16384`: total number of tokens across the batch. Logical batch is `TOTAL_TOKENS / N`.

**Heads are folded into the batch dimension** to keep launch geometry simple. The effective batch passed to the kernel is $B = B_{\text{logical}} \times H$, so each (head, batch) pair becomes one independent attention computation. The TFLOPS calculation in the harness unfolds this to report per-attention-call performance.

#### Tensor layout convention

PyTorch tensors are created in `(B, N, d)` row-major. Inside the kernels, we work with the CUTE view of the same memory in `(N, d, B)` layout, where $d$ is the stride-1 axis (K-major from the MMA's perspective) and $B$ is the outermost batch dimension. The view is constructed via `permute(1, 2, 0)` on the PyTorch tensor and wrapped with `from_dlpack` into a CUTE tensor. No data is copied.

All four tensors (Q, K, V, O) share this layout for the BF16 kernels (K1 through K6). K7 introduces FP8, which requires a different V layout to satisfy WGMMA's K-major-only constraint for FP8 operands.

#### Benchmark harness

`bench.py`'s flow:

1. Load the kernel module by short name (e.g., `bench.py k1` imports `kernels.k1`).
2. Sweep over `(causal, headdim, seqlen)` triples, with optional env-var overrides to pin one dimension.
3. For each config: build Q/K/V/O tensors, compile the kernel once, optionally run a correctness check, do warmup iterations, then measure 30 timed iterations using CUDA events.
4. Report average time per iteration and achieved TFLOPS. The TFLOPS calculation uses $4 B N^2 H d$ for non-causal (halved for causal).

Warmup count, bench count, and sweep ranges are set near the top of `bench.py`. Defaults: 10 warmup, 30 bench. Bump these if results are noisy on your hardware.

#### Correctness check

`ref_check.py` runs the kernel against `torch.nn.functional.scaled_dot_product_attention` on the same inputs and reports the diff. The check is opt-in via `CHECK=1`. When enabled, the output line includes:

- **`max_abs`**: maximum element-wise absolute error
- **`max_rel`**: maximum element-wise relative error
- **`bad`**: count of elements outside the absolute tolerance (default `1e-2` for BF16)
- **`total`**: total number of elements compared

For BF16 kernels, a clean run has `bad = 0` and `max_abs` in the $10^{-2}$ range. K7 (FP8) uses a much looser tolerance because per-tensor scaling introduces real quantization error.

#### How to run

##### One-time setup

```bash
cd fa3
uv sync
source .venv/bin/activate
```

##### Run a single config with a correctness check

```bash
# Pin one (seqlen, headdim, causal) triple via env vars.
CHECK=1 SEQLEN=512 HEADDIM=64 CAUSAL=0 python bench.py k1
```

`CHECK=1` enables the SDPA correctness check. `SEQLEN`, `HEADDIM`, and `CAUSAL` are optional pins; omitting any of them sweeps that dimension.

##### Run the full sweep for a kernel

```bash
CHECK=1 python bench.py k1
```

Sweeps causal in `{0, 1}`, headdim in `{64, 128}`, seqlen in `{512, ..., 16384}`. 24 configs per run. Drop `CHECK=1` for timing-only runs.

##### Other kernels

Same pattern, swap the kernel name: `python bench.py k2`, `python bench.py k3`, etc.

##### Clear the compile cache

Only needed after a GPU swap or for a clean rebuild:

```bash
rm -rf kernels/__pycache__
```

<!-- ## License -->

<!-- TODO: pick a license (MIT, Apache 2.0, etc.) -->