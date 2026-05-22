"""Benchmark harness for CuTe DSL FMHA kernels.

Sweeps over (causal, headdim, seqlen). Heads are folded into the batch
dimension to keep launch geometry simple.

Env vars:
  CHECK    if set, run correctness check vs PyTorch SDPA
  SEQLEN   only run this seqlen (e.g. SEQLEN=2048)
  HEADDIM  only run this headdim
  CAUSAL   only run causal (1) or non-causal (0)
  DEVICE   CUDA device index (default 0)

Usage:
  python bench.py k1
  CHECK=1 SEQLEN=2048 python bench.py k1
"""
import argparse
import importlib
import os

import torch

from utils import make_qkvo
from ref_check import run_check


# Workload-size constants. Tune to your hardware/workload if needed.
HIDDEN_DIM = 2048
TOTAL_TOKENS = 16384
SEQLEN_VALS = [512, 1024, 2048, 4096, 8192, 16384]
HEADDIM_VALS = [64, 128]
WARMUP_RUNS = 10
BENCH_RUNS = 30


def attention_flops(batch, seq_len, headdim, nheads, causal):
    """Forward pass: 4 * B * N^2 * H * d, halved for causal."""
    f = 4.0 * batch * seq_len * seq_len * nheads * headdim
    return f / 2.0 if causal else f


def load_kernel_module(name):
    """Import kernels.<name> and return the module. Must export
    compile_kernel(B, N, d, causal, tensors) and run_kernel(compiled, tensors).
    """
    mod = importlib.import_module(f"kernels.{name}")

    if not hasattr(mod, "compile_kernel") or not hasattr(mod, "run_kernel"):
        raise AttributeError(
            f"Module 'kernels.{name}' must export "
            f"compile_kernel(B, N, d, causal, tensors) and run_kernel(compiled, tensors)."
        )
    return mod


def kernel_display_name(name):
    """Pretty name for the kernel."""
    table = {
        "k1": "K1: Naive (materialized S)",
        "k2": "K2: Tiled online softmax",
    }
    return table.get(name, name)


def run_one_config(mod, B, seq_len, d, causal, do_check):
    """Warmup + bench for a single (B, N, d, causal) config.

    Returns:
      avg_ms, tflops, status_string  (status='' on success, otherwise a diagnostic)
    """
    torch.manual_seed(1111)

    try:
        tensors = make_qkvo(B, seq_len, d)
    except torch.cuda.OutOfMemoryError:
        # 4 buffers, BF16 (2 bytes each)
        gb = B * seq_len * d * 4 * 2 / 1e9
        return None, None, f"[OOM: Q/K/V/O alloc, {gb:.2f} GB needed]"

    # Compile.
    try:
        compiled = mod.compile_kernel(B, seq_len, d, causal, tensors)
    except Exception as e:
        return None, None, f"[compile failed: {e}]"

    # Optional correctness check.
    if do_check:
        try:
            run_check(
                lambda t: mod.run_kernel(compiled, t),
                tensors, B, seq_len, d, causal=causal,
            )
        except Exception as e:
            print(f"         [check failed: {e}]")

    # Warmup.
    try:
        for _ in range(WARMUP_RUNS):
            mod.run_kernel(compiled, tensors)
        torch.cuda.synchronize()
    except Exception as e:
        return None, None, f"[SKIP: warmup failed: {e}]"

    # Benchmark.
    try:
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(BENCH_RUNS):
            mod.run_kernel(compiled, tensors)
        stop.record()
        stop.synchronize()
        elapsed_ms = start.elapsed_time(stop)
        avg_ms = elapsed_ms / BENCH_RUNS
    except Exception as e:
        return None, None, f"[SKIP: bench failed: {e}]"

    del tensors
    torch.cuda.empty_cache()

    nheads = HIDDEN_DIM // d
    batch_logical = B // nheads
    flops = attention_flops(batch_logical, seq_len, d, nheads, causal)
    tflops = flops / (avg_ms * 1e-3) / 1e12

    return avg_ms, tflops, ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel", help="Kernel name (e.g. k1)")
    args = parser.parse_args()

    device = int(os.environ.get("DEVICE", "0"))
    torch.cuda.set_device(device)
    prop = torch.cuda.get_device_properties(device)
    print(f"Device: {prop.name} (SM {prop.major}{prop.minor})")
    print(f"Kernel: {kernel_display_name(args.kernel)}")
    print(f"Hidden dim: {HIDDEN_DIM}, Total tokens: {TOTAL_TOKENS}\n")

    mod = load_kernel_module(args.kernel)

    do_check = bool(os.environ.get("CHECK", ""))
    env_seqlen = os.environ.get("SEQLEN", "")
    env_headdim = os.environ.get("HEADDIM", "")
    env_causal = os.environ.get("CAUSAL", "")

    for causal in [0, 1]:
        if env_causal and int(env_causal) != causal:
            continue
        for d in HEADDIM_VALS:
            if env_headdim and int(env_headdim) != d:
                continue
            nheads = HIDDEN_DIM // d
            print(f"### causal={causal}, headdim={d}, nheads={nheads} ###")
            print(f"{'seqlen':>8} {'batch':>6} {'avg_ms':>12} {'TFLOPS':>10}")

            for seq_len in SEQLEN_VALS:
                if env_seqlen and int(env_seqlen) != seq_len:
                    continue
                batch_logical = TOTAL_TOKENS // seq_len
                B = batch_logical * nheads  # fold heads into batch

                avg_ms, tflops, status = run_one_config(
                    mod, B, seq_len, d, bool(causal), do_check,
                )
                if status:
                    print(f"{seq_len:>8} {batch_logical:>6}  {status}")
                else:
                    print(f"{seq_len:>8} {batch_logical:>6}  {avg_ms:>8.3f} ms   {tflops:>8.2f}")

            print()


if __name__ == "__main__":
    main()