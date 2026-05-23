"""K1: Naive materialized attention.

The "before" picture: this is what attention looks like without flash. Three
separate kernels, materialized (B, N, N) attention matrix in FP32, no fusion.
OOMs at large N (e.g. B=1024, N=16384 needs ~1 TB).

Step 1: S = Q @ K^T * scale (+ causal mask)  [one thread per element]
Step 2: P = softmax(S)                        [one thread per row, serialized]
Step 3: O = P @ V                             [one thread per element]

Layout:
  Q, K, V, O: (N, d, B) CUTE view, d stride-1 (matches make_qkvo BF16 path).
  S: (N, N, B) materialized FP32, N stride-1.
"""
import math

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from utils import INPUT_DTYPE, OUTPUT_DTYPE, ACC_DTYPE, get_cuda_stream


SCORE_BLOCK_X = 16
SCORE_BLOCK_Y = 16
SOFTMAX_THREADS = 256
OUTPUT_BLOCK_X = 16
OUTPUT_BLOCK_Y = 16


class K1Score:
    """S[b, i, j] = sum_k Q[b, i, k] * K[b, j, k] * scale (with optional causal mask)."""

    def __init__(self, B, N, d, causal):
        self.B = B
        self.N = N
        self.d = d
        self.causal = causal

    @cute.jit
    def __call__(self, mQ, mK, mS, scale: cutlass.Float32, stream):
        grid = (
            (self.N + SCORE_BLOCK_X - 1) // SCORE_BLOCK_X,
            (self.N + SCORE_BLOCK_Y - 1) // SCORE_BLOCK_Y,
            self.B,
        )
        self.kernel(mQ, mK, mS, scale).launch(
            grid=grid,
            block=[SCORE_BLOCK_X, SCORE_BLOCK_Y, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mQ, mK, mS, scale: cutlass.Float32):
        bx, by, bz = cute.arch.block_idx()
        tx, ty, _ = cute.arch.thread_idx()

        j = bx * SCORE_BLOCK_X + tx
        i = by * SCORE_BLOCK_Y + ty
        b = bz

        if i < self.N and j < self.N:
            # Causal mask: kv position > query position => -inf.
            if cutlass.const_expr(self.causal) and j > i:
                mS[i, j, b] = -cutlass.Float32.inf
            else:
                gQ = mQ[(None, None, b)]
                gK = mK[(None, None, b)]
                dot = cutlass.Float32(0.0)
                for k in cutlass.range_constexpr(self.d):
                    dot = dot + gQ[i, k].to(cutlass.Float32) * gK[j, k].to(cutlass.Float32)
                mS[i, j, b] = dot * scale


class K1Softmax:
    """Per-row softmax, one thread per row (intentionally naive)."""

    def __init__(self, B, N):
        self.B = B
        self.N = N

    @cute.jit
    def __call__(self, mS, stream):
        grid = ((self.N + SOFTMAX_THREADS - 1) // SOFTMAX_THREADS, self.B, 1)
        self.kernel(mS).launch(
            grid=grid,
            block=[SOFTMAX_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mS):
        bx, by, _ = cute.arch.block_idx()
        tx, _, _ = cute.arch.thread_idx()

        i = bx * SOFTMAX_THREADS + tx
        b = by

        if i < self.N:
            # Row max.
            m = cutlass.Float32(-cutlass.Float32.inf)
            for j in cutlass.range(self.N, unroll=1):
                m = cute.arch.fmax(m, mS[i, j, b])

            # Exp and sum.
            s = cutlass.Float32(0.0)
            for j in cutlass.range(self.N, unroll=1):
                p = cute.math.exp(mS[i, j, b] - m, fastmath=True)
                mS[i, j, b] = p
                s = s + p

            # Normalize.
            inv_s = cute.arch.rcp_approx(s)
            if s == 0.0 or s != s:
                inv_s = cutlass.Float32(1.0)
            for j in cutlass.range(self.N, unroll=1):
                mS[i, j, b] = mS[i, j, b] * inv_s


class K1Output:
    """O[b, i, j] = sum_k P[b, i, k] * V[b, k, j]."""

    def __init__(self, B, N, d):
        self.B = B
        self.N = N
        self.d = d

    @cute.jit
    def __call__(self, mS, mV, mO, stream):
        grid = (
            (self.d + OUTPUT_BLOCK_X - 1) // OUTPUT_BLOCK_X,
            (self.N + OUTPUT_BLOCK_Y - 1) // OUTPUT_BLOCK_Y,
            self.B,
        )
        self.kernel(mS, mV, mO).launch(
            grid=grid,
            block=[OUTPUT_BLOCK_X, OUTPUT_BLOCK_Y, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mS, mV, mO):
        bx, by, bz = cute.arch.block_idx()
        tx, ty, _ = cute.arch.thread_idx()

        j = bx * OUTPUT_BLOCK_X + tx
        i = by * OUTPUT_BLOCK_Y + ty
        b = bz

        if i < self.N and j < self.d:
            gV = mV[(None, None, b)]
            gO = mO[(None, None, b)]
            s = cutlass.Float32(0.0)
            for k in cutlass.range(self.N, unroll=1):
                s = s + mS[i, k, b] * gV[k, j].to(cutlass.Float32)
            gO[i, j] = s.to(OUTPUT_DTYPE)


def _make_S_tensor(B, N):
    """Allocate the (N, N, B) FP32 S buffer and wrap as CUTE tensor.

    Layout matches Q/K/V/O convention: mode-0 (i-axis) is stride-1.
    Raises torch.cuda.OutOfMemoryError on alloc failure (caught by bench).
    """
    # (B, N, N) row-major storage, strides (N*N, N, 1).
    # Permute (2, 1, 0) -> (N, N, B), strides (1, N, N*N): inner i-axis stride-1.
    s_storage = torch.empty((B, N, N), dtype=torch.float32, device="cuda")
    s_view = s_storage.permute(2, 1, 0)
    s_cute = from_dlpack(s_view, assumed_align=16)
    s_cute.element_type = cutlass.Float32
    s_cute = s_cute.mark_layout_dynamic(leading_dim=0)
    return s_storage, s_cute


def compile_kernel(B, N, d, causal, tensors):
    score = K1Score(B, N, d, causal)
    softmax = K1Softmax(B, N)
    output = K1Output(B, N, d)
    scale = cutlass.Float32(1.0 / math.sqrt(d))
    stream = get_cuda_stream()

    # Allocate S buffer once at compile time (kept alive in compiled handle).
    s_storage, s_cute = _make_S_tensor(B, N)

    score_compiled = cute.compile(
        score, tensors["q_cute"], tensors["k_cute"], s_cute, scale, stream,
    )
    softmax_compiled = cute.compile(softmax, s_cute, stream)
    output_compiled = cute.compile(
        output, s_cute, tensors["v_cute"], tensors["o_cute"], stream,
    )

    return (score_compiled, softmax_compiled, output_compiled,
            scale, s_storage, s_cute)


def run_kernel(compiled_handle, tensors):
    (score_compiled, softmax_compiled, output_compiled,
     scale, s_storage, s_cute) = compiled_handle
    stream = get_cuda_stream()

    score_compiled(tensors["q_cute"], tensors["k_cute"], s_cute, scale, stream)
    softmax_compiled(s_cute, stream)
    output_compiled(s_cute, tensors["v_cute"], tensors["o_cute"], stream)