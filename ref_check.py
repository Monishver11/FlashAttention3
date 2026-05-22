"""Correctness check for CuTe DSL FMHA kernels.

Compares a kernel's output against PyTorch SDPA reference.
Prints max_abs, max_rel, bad_count, and tolerance.
"""
import torch
import torch.nn.functional as F


def torch_sdpa_reference(q_gpu, k_gpu, v_gpu, B, N, d, causal=False):
    """Run PyTorch SDPA on tensors in CUTE convention (N, d, B).

    Reshapes to (B, 1, N, d) for SDPA, then back to (N, d, B).

    Args:
      q_gpu, k_gpu, v_gpu: (N, d, B) BF16 GPU tensors.
      B, N, d:             shape parameters.
      causal:              causal masking flag.

    Returns:
      o_ref: (N, d, B) BF16 GPU tensor, SDPA's output.
    """
    # (N, d, B) -> (B, N, d) -> (B, 1, N, d)
    def to_sdpa(t):
        return t.permute(2, 0, 1).contiguous().unsqueeze(1)

    q_sdpa = to_sdpa(q_gpu)
    k_sdpa = to_sdpa(k_gpu)
    v_sdpa = to_sdpa(v_gpu)

    with torch.no_grad():
        o_sdpa = F.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa, is_causal=causal,
        )

    # (B, 1, N, d) -> (B, N, d) -> (N, d, B)
    return o_sdpa.squeeze(1).permute(1, 2, 0).contiguous()


def check_kernel_output(o_test_gpu, o_ref_gpu, tol=1e-2, label=""):
    """Compare two (N, d, B) BF16 GPU tensors and print diagnostics.

    Args:
      o_test_gpu: kernel output, (N, d, B) BF16.
      o_ref_gpu:  reference output, (N, d, B) BF16.
      tol:        absolute tolerance for the "bad" count.
      label:      optional prefix for the print line.
    """
    o_test = o_test_gpu.float().cpu()
    o_ref = o_ref_gpu.float().cpu()

    abs_diff = (o_test - o_ref).abs()
    rel_diff = abs_diff / o_ref.abs().clamp(min=1e-6)

    max_abs = abs_diff.max().item()
    max_rel = rel_diff.max().item()
    bad = (abs_diff > tol).sum().item()
    total = o_test.numel()

    prefix = f"[{label}] " if label else ""
    print(f"         {prefix}[check: max_abs={max_abs:.2e} "
          f"max_rel={max_rel:.2e} bad={bad}/{total} (tol={tol:.0e})]")

    return max_abs, max_rel, bad, total


def run_check(kernel_fn, tensors, B, N, d, causal=False, tol=1e-2, label=""):
    """End-to-end correctness check.

    Args:
      kernel_fn: callable(tensors_dict) that runs the kernel and writes into o_gpu.
      tensors:   dict from utils.make_qkvo.
      B, N, d:   shape.
      causal:    causal masking flag.
      tol:       absolute tolerance.
      label:     optional prefix for the print.

    Returns:
      (max_abs, max_rel, bad, total) tuple.
    """
    kernel_fn(tensors)
    torch.cuda.synchronize()

    o_ref = torch_sdpa_reference(
        tensors["q_gpu"], tensors["k_gpu"], tensors["v_gpu"],
        B, N, d, causal=causal,
    )

    return check_kernel_output(tensors["o_gpu"], o_ref, tol=tol, label=label)