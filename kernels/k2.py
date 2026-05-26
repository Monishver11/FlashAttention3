"""K2: Tiled online softmax (FA2 algorithm in CuTe Python DSL).

One fused kernel per (Q-tile, batch). Hand-rolled FP32 GEMM in registers,
synchronous GMEM->SMEM copies, online softmax with cross-thread row reduction.

No tensor cores, no TMA, no async pipelining. K3 brings TMA; K4 brings WGMMA.

Threading (THREADS=256, THREADS_PER_ROW=4):
  rows per CTA            = THREADS / THREADS_PER_ROW = 64 = Br
  cols of O per thread    = d  / THREADS_PER_ROW           # O sharded along d
  cols of S row per thread= Bc / THREADS_PER_ROW           # row-scan stride

SMEM (Br=Bc=64, d=128):
  sQ  Br*d   BF16  16 KB
  sKV Bc*d   BF16  16 KB   (reused for K then V each iter)
  sS  Br*Bc  FP32  16 KB
  total                     48 KB
"""
import math

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils

from utils import INPUT_DTYPE, OUTPUT_DTYPE, ACC_DTYPE, get_cuda_stream


Br = 64
Bc = 64
THREADS = 256
THREADS_PER_ROW = 4


class K2FMHA:
    def __init__(self, B, N, d, causal):
        self.B = B
        self.N = N
        self.d = d
        self.causal = causal

        self.Br = Br
        self.Bc = Bc
        self.threads = THREADS
        self.threads_per_row = THREADS_PER_ROW
        self.cols_per_thread_O = d  // THREADS_PER_ROW     # along d
        self.cols_per_thread_S = Bc // THREADS_PER_ROW     # along Bc (row scan)
        self.buffer_align_bytes = 1024

    @cute.jit
    def __call__(self, mQ, mK, mV, mO, scale: cutlass.Float32, stream):
        sQ_layout  = cute.make_layout((self.Br, self.d),  stride=(self.d, 1))
        sKV_layout = cute.make_layout((self.Bc, self.d),  stride=(self.d, 1))
        sS_layout  = cute.make_layout((self.Br, self.Bc), stride=(self.Bc, 1))

        @cute.struct
        class SharedStorage:
            sQ:  cute.struct.Align[
                cute.struct.MemRange[INPUT_DTYPE, cute.cosize(sQ_layout)],
                self.buffer_align_bytes,
            ]
            sKV: cute.struct.Align[
                cute.struct.MemRange[INPUT_DTYPE, cute.cosize(sKV_layout)],
                self.buffer_align_bytes,
            ]
            sS:  cute.struct.Align[
                cute.struct.MemRange[ACC_DTYPE, cute.cosize(sS_layout)],
                self.buffer_align_bytes,
            ]
        self.shared_storage = SharedStorage

        self.kernel(
            mQ, mK, mV, mO, sQ_layout, sKV_layout, sS_layout, scale,
        ).launch(
            grid=(self.N // self.Br, self.B, 1),
            block=[self.threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self, mQ, mK, mV, mO,
        sQ_layout, sKV_layout, sS_layout,
        scale: cutlass.Float32,
    ):
        bidx_m, bidx_b, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        # ---- SMEM tiles ----
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sQ  = storage.sQ.get_tensor(sQ_layout)
        sKV = storage.sKV.get_tensor(sKV_layout)   # K then V
        sS  = storage.sS.get_tensor(sS_layout)

        # ---- Per-batch GMEM views, (N, d) ----
        gQ = mQ[(None, None, bidx_b)]
        gK = mK[(None, None, bidx_b)]
        gV = mV[(None, None, bidx_b)]
        gO = mO[(None, None, bidx_b)]

        # ---- Thread -> (row, col_group) mapping ----
        # 256 threads / 4-per-row = 64 row groups = Br.
        row         = tidx // self.threads_per_row     # which Q-row this thread serves
        col_group   = tidx %  self.threads_per_row     # 0..3, which 1/4 of the row
        d_col_start = col_group * self.cols_per_thread_O

        # ---- Per-row online softmax state (registers, replicated in row group) ----
        m_i = cutlass.Float32(-cutlass.Float32.inf)
        l_i = cutlass.Float32(0.0)
        O_acc = cute.make_rmem_tensor(
            cute.make_layout(self.cols_per_thread_O), ACC_DTYPE,
        )
        for c in cutlass.range_constexpr(self.cols_per_thread_O):
            O_acc[c] = cutlass.Float32(0.0)

        q_row_start = bidx_m * self.Br

        # =====================================================================
        # 1. Load Q tile into SMEM (once per CTA, kept across all KV iters)
        # =====================================================================
        for it in cutlass.range_constexpr(0, self.Br * self.d, self.threads):
            ij = it + tidx
            sQ[ij // self.d, ij % self.d] = gQ[q_row_start + ij // self.d, ij % self.d]
        cute.arch.sync_threads()

        # =====================================================================
        # 2. Mainloop over KV tiles
        # =====================================================================
        n_kv = self.N // self.Bc
        for j in cutlass.range(n_kv, unroll=1):
            kv_row_start = j * self.Bc

            # ---- 2a. Load K_j into sKV ----
            for it in cutlass.range_constexpr(0, self.Bc * self.d, self.threads):
                ij = it + tidx
                sKV[ij // self.d, ij % self.d] = gK[kv_row_start + ij // self.d, ij % self.d]
            cute.arch.sync_threads()

            # ---- 2b. S = Q @ K^T * scale (+ causal mask) ----
            # 256 threads cover Br*Bc=4096 elements: 16 per thread, strided.
            for it in cutlass.range_constexpr(0, self.Br * self.Bc, self.threads):
                ij = it + tidx
                si, sj = ij // self.Bc, ij % self.Bc
                dot = cutlass.Float32(0.0)
                for k in cutlass.range_constexpr(self.d):
                    dot = dot + sQ[si, k].to(cutlass.Float32) \
                              * sKV[sj, k].to(cutlass.Float32)
                val = dot * scale
                if cutlass.const_expr(self.causal):
                    if kv_row_start + sj > q_row_start + si:
                        val = -cutlass.Float32.inf
                sS[si, sj] = val
            cute.arch.sync_threads()

            # ---- 2c. Online softmax row update ----
            # Row max: 4 threads scan 64 cols, 16 cols each, then sub-warp reduce.
            local_max = cutlass.Float32(-cutlass.Float32.inf)
            for c_it in cutlass.range_constexpr(self.cols_per_thread_S):
                c = col_group + c_it * self.threads_per_row
                local_max = cute.arch.fmax(local_max, sS[row, c])
            row_max = cute.arch.warp_reduction_max(
                local_max, threads_in_group=self.threads_per_row,
            )

            m_new = cute.arch.fmax(m_i, row_max)
            alpha = cute.math.exp(m_i - m_new, fastmath=True)

            # Rescale O accumulator before fold-in of this tile.
            for c in cutlass.range_constexpr(self.cols_per_thread_O):
                O_acc[c] = O_acc[c] * alpha

            # P = exp(S - m_new), written back into sS in place; collect row sum.
            local_sum = cutlass.Float32(0.0)
            for c_it in cutlass.range_constexpr(self.cols_per_thread_S):
                c = col_group + c_it * self.threads_per_row
                p = cute.math.exp(sS[row, c] - m_new, fastmath=True)
                sS[row, c] = p
                local_sum = local_sum + p
            row_sum = cute.arch.warp_reduction_sum(
                local_sum, threads_in_group=self.threads_per_row,
            )

            l_i = alpha * l_i + row_sum
            m_i = m_new
            cute.arch.sync_threads()   # sS now holds P; visible before V load / PV

            # ---- 2d. Load V_j into sKV (overwrites K_j) ----
            for it in cutlass.range_constexpr(0, self.Bc * self.d, self.threads):
                ij = it + tidx
                sKV[ij // self.d, ij % self.d] = gV[kv_row_start + ij // self.d, ij % self.d]
            cute.arch.sync_threads()

            # ---- 2e. O_acc += P @ V_j (each thread updates its d slice) ----
            for c in cutlass.range_constexpr(self.cols_per_thread_O):
                col = d_col_start + c
                pv = cutlass.Float32(0.0)
                for k in cutlass.range_constexpr(self.Bc):
                    pv = pv + sS[row, k] * sKV[k, col].to(cutlass.Float32)
                O_acc[c] = O_acc[c] + pv
            cute.arch.sync_threads()   # V reads done before next iter overwrites

        # =====================================================================
        # 3. Finalize: O /= l, write to GMEM
        # =====================================================================
        inv_l = cute.arch.rcp_approx(l_i)
        if l_i == 0.0 or l_i != l_i:           # fully-masked row guard
            inv_l = cutlass.Float32(1.0)
        for c in cutlass.range_constexpr(self.cols_per_thread_O):
            gO[q_row_start + row, d_col_start + c] = \
                (O_acc[c] * inv_l).to(OUTPUT_DTYPE)


def compile_kernel(B, N, d, causal, tensors):
    fmha = K2FMHA(B, N, d, causal)
    scale = cutlass.Float32(1.0 / math.sqrt(d))
    stream = get_cuda_stream()
    compiled = cute.compile(
        fmha,
        tensors["q_cute"], tensors["k_cute"], tensors["v_cute"], tensors["o_cute"],
        scale, stream,
    )
    return (compiled, scale)


def run_kernel(compiled_handle, tensors):
    compiled, scale = compiled_handle
    stream = get_cuda_stream()
    compiled(
        tensors["q_cute"], tensors["k_cute"], tensors["v_cute"], tensors["o_cute"],
        scale, stream,
    )