"""Shared utilities for CuTe DSL FMHA kernels.

Tensor convention:
  PyTorch:  (B, N, d) row-major BF16 on GPU
  CUTE:     (N, d, B) view of the same memory, with d stride-1 (K-major
            from MMA's perspective) and B as the outermost batch dimension.
"""
import torch
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack


# Default dtypes for BF16 kernels (K1 through K6).
INPUT_DTYPE = cutlass.BFloat16
OUTPUT_DTYPE = cutlass.BFloat16
ACC_DTYPE = cutlass.Float32


def create_and_permute_tensor(B, N, d, dtype, init_random=True):
    """Create a (B, N, d) tensor and its (N, d, B) CUTE view.

    Follows the canonical CUTLASS pattern:
      FP32 CPU master -> typed GPU buffer -> CUTE wraps GPU buffer.

    Returns:
      f32_cpu:   (N, d, B) FP32 PyTorch tensor on CPU, kept as reference.
      cute_t:    CUTE tensor viewing the GPU buffer in BF16 with (N, d, B) layout.
      torch_gpu: (N, d, B) BF16 PyTorch tensor on GPU.
    """
    # Generation shape is (B, N, d); permute to (N, d, B) for the CUTE view.
    shape = (B, N, d)
    permute_order = (1, 2, 0)  # (B, N, d) -> (N, d, B)

    if init_random:
        init_type = cutlass_torch.TensorInitType.RANDOM
        init_config = cutlass_torch.RandomInitConfig(min_val=-2, max_val=2)
    else:
        init_type = cutlass_torch.TensorInitType.SKIP
        init_config = None

    # CPU tensor in target dtype, already permuted to (N, d, B).
    torch_cpu = cutlass_torch.create_and_permute_torch_tensor(
        shape,
        cutlass_torch.dtype(dtype),
        permute_order=permute_order,
        init_type=init_type,
        init_config=init_config,
    )

    # GPU tensor (typed buffer).
    torch_gpu = torch_cpu.cuda()

    # FP32 CPU master copy, kept alive for correctness checks.
    f32_cpu = torch_cpu.to(dtype=torch.float32)

    # CUTE view of the GPU buffer; d-axis is stride-1.
    cute_t = from_dlpack(torch_gpu, assumed_align=16)
    cute_t.element_type = dtype
    cute_t = cute_t.mark_layout_dynamic(leading_dim=1)
    cute_t = cutlass_torch.convert_cute_tensor(
        f32_cpu, cute_t, dtype, is_dynamic_layout=True,
    )

    return f32_cpu, cute_t, torch_gpu


def make_qkvo(B, N, d, dtype=INPUT_DTYPE, out_dtype=OUTPUT_DTYPE):
    """Create Q, K, V (random) and O (zero) for an MHA call.

    Each tensor is (N, d, B) in CUTE-land, BF16 on GPU.
    Returns a dict with f32/cute/gpu views of all four tensors.
    """
    q_f32, q_cute, q_gpu = create_and_permute_tensor(B, N, d, dtype, init_random=True)
    k_f32, k_cute, k_gpu = create_and_permute_tensor(B, N, d, dtype, init_random=True)
    v_f32, v_cute, v_gpu = create_and_permute_tensor(B, N, d, dtype, init_random=True)
    o_f32, o_cute, o_gpu = create_and_permute_tensor(B, N, d, out_dtype, init_random=False)

    return {
        "q_f32": q_f32, "k_f32": k_f32, "v_f32": v_f32, "o_f32": o_f32,
        "q_cute": q_cute, "k_cute": k_cute, "v_cute": v_cute, "o_cute": o_cute,
        "q_gpu": q_gpu, "k_gpu": k_gpu, "v_gpu": v_gpu, "o_gpu": o_gpu,
    }


def get_cuda_stream():
    """Wrap PyTorch's current CUDA stream as a CUstream for CUTE kernel launch."""
    import cuda.bindings.driver as cuda
    torch_stream = torch.cuda.current_stream()
    return cuda.CUstream(torch_stream.cuda_stream)