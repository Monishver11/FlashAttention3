"""K3: TMA single-stage FMHA forward.

This is the K3.1 variant of the K3 progression (K3.1=1 stage, K3.2=2, K3.3=5).
The blog covers K3.1 only; multi-stage variants land in K4 once WGMMA gives the
consumer side something heavy enough to actually overlap with.

What changes vs K2:
  * GMEM->SMEM copies are issued by a single thread (warp 0) as TMA bulk loads,
    rather than 256 threads each issuing per-element loads.
  * Completion is tracked by hardware mbarriers via PipelineTmaAsync, not by
    cute.arch.sync_threads.

What does NOT change vs K2:
  * Threading model: 256 threads, 4-per-row, hand-rolled scalar FP32 GEMMs.
  * SMEM layouts: plain row-major. Swizzle is TODO and lands together with
    WGMMA in K4 (swizzle is only meaningful for tensor-core access patterns).
  * Online softmax math.

Pipelines:
  Q  : 1 stage, producer = 1 thread, consumer = 256 threads.
       Loaded once per CTA, reused for every KV iteration.
  KV : 1 stage in K3.1. K and V share the stage index with a single barrier
       whose tx_count = sK_bytes + sV_bytes; the barrier only fires once both
       TMAs have arrived.
"""
import math

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from utils import INPUT_DTYPE, OUTPUT_DTYPE, ACC_DTYPE, get_cuda_stream


Br = 64
Bc = 64
THREADS = 256
THREADS_PER_ROW = 4


class K3FMHA:
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

        # Pipeline depths. K3 = single-stage everywhere; staging arrives in K4.
        self.q_stage = 1
        self.kv_stage = 1

        # Cluster geometry. Single CTA, no multicast.
        self.cluster_shape_mn = (1, 1)
        self.threads_per_cta = self.threads

    @cute.jit
    def __call__(self, mQ, mK, mV, mO, scale: cutlass.Float32, stream):
        # ---- SMEM layouts (plain row-major; TODO: swizzle in K4) ----
        sQ_layout_staged = cute.make_layout(
            (self.Br, self.d, self.q_stage),
            stride=(self.d, 1, self.Br * self.d),
        )
        sK_layout_staged = cute.make_layout(
            (self.Bc, self.d, self.kv_stage),
            stride=(self.d, 1, self.Bc * self.d),
        )
        sV_layout_staged = cute.make_layout(
            (self.Bc, self.d, self.kv_stage),
            stride=(self.d, 1, self.Bc * self.d),
        )
        # Per-stage layouts for the TMA atoms (TMA descriptor is per-stage).
        sQ_layout = cute.slice_(sQ_layout_staged, (None, None, 0))
        sK_layout = cute.slice_(sK_layout_staged, (None, None, 0))
        sV_layout = cute.slice_(sV_layout_staged, (None, None, 0))

        sS_layout = cute.make_layout((self.Br, self.Bc), stride=(self.Bc, 1))

        # ---- TMA atoms (G2S, no multicast) ----
        tma_atom_q, tma_tensor_q = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            mQ, sQ_layout, (self.Br, self.d), num_multicast=1,
        )
        tma_atom_k, tma_tensor_k = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            mK, sK_layout, (self.Bc, self.d), num_multicast=1,
        )
        tma_atom_v, tma_tensor_v = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            mV, sV_layout, (self.Bc, self.d), num_multicast=1,
        )

        # ---- mbarrier byte counts ----
        # KV barrier covers K + V together, so the barrier only fires once
        # both TMAs have arrived.
        q_bytes  = cute.size_in_bytes(INPUT_DTYPE, sQ_layout)
        kv_bytes = (cute.size_in_bytes(INPUT_DTYPE, sK_layout)
                    + cute.size_in_bytes(INPUT_DTYPE, sV_layout))

        # ---- SMEM storage struct ----
        @cute.struct
        class SharedStorage:
            q_pipeline_array_ptr:  cute.struct.MemRange[cutlass.Int64, self.q_stage  * 2]
            kv_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.kv_stage * 2]
            sQ: cute.struct.Align[
                cute.struct.MemRange[INPUT_DTYPE, cute.cosize(sQ_layout_staged)],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[INPUT_DTYPE, cute.cosize(sK_layout_staged)],
                self.buffer_align_bytes,
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[INPUT_DTYPE, cute.cosize(sV_layout_staged)],
                self.buffer_align_bytes,
            ]
            sS: cute.struct.Align[
                cute.struct.MemRange[ACC_DTYPE, cute.cosize(sS_layout)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage
        self.q_bytes  = q_bytes
        self.kv_bytes = kv_bytes

        self.kernel(
            tma_atom_q, tma_tensor_q,
            tma_atom_k, tma_tensor_k,
            tma_atom_v, tma_tensor_v,
            mO,
            sQ_layout_staged, sK_layout_staged, sV_layout_staged, sS_layout,
            scale,
        ).launch(
            grid=(self.N // self.Br, self.B, 1),
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tma_atom_q, tma_tensor_q,
        tma_atom_k, tma_tensor_k,
        tma_atom_v, tma_tensor_v,
        mO,
        sQ_layout_staged, sK_layout_staged, sV_layout_staged, sS_layout,
        scale: cutlass.Float32,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # ---- Descriptor prefetch (warp 0, once) ----
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)

        bidx_m, bidx_b, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        # ---- SMEM allocation ----
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # ---- Pipelines: Q (one-shot reuse) and KV (per-iter) ----
        num_warps = self.threads_per_cta // 32

        q_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.q_pipeline_array_ptr.data_ptr(),
            num_stages=self.q_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, num_warps),
            tx_count=self.q_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
            defer_sync=True,
        )
        kv_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.kv_pipeline_array_ptr.data_ptr(),
            num_stages=self.kv_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, num_warps),
            tx_count=self.kv_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
            defer_sync=True,
        )

        # No-op for cluster size 1, but the canonical init pattern.
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # ---- SMEM tensors ----
        sQ_full = storage.sQ.get_tensor(sQ_layout_staged)
        sK_full = storage.sK.get_tensor(sK_layout_staged)
        sV_full = storage.sV.get_tensor(sV_layout_staged)
        sS      = storage.sS.get_tensor(sS_layout)

        # Q has q_stage=1, so strip the stage dim once. The Q tile lives at
        # stage 0 for the kernel's lifetime.
        sQ = sQ_full[(None, None, 0)]

        # ---- GMEM tile views ----
        gQ = cute.flat_divide(tma_tensor_q, (self.Br, self.d))
        gK = cute.flat_divide(tma_tensor_k, (self.Bc, self.d))
        gV = cute.flat_divide(tma_tensor_v, (self.Bc, self.d))
        gO = cute.flat_divide(mO,           (self.Br, self.d))

        gQ_tiles = gQ[(None, None, None, 0, bidx_b)]
        gK_tiles = gK[(None, None, None, 0, bidx_b)]
        gV_tiles = gV[(None, None, None, 0, bidx_b)]
        gO_tile  = gO[(None, None, bidx_m, 0, bidx_b)]

        # ---- TMA partitioning ----
        tQsQ, tQgQ = cute.nvgpu.cpasync.tma_partition(
            tma_atom_q, 0, cute.make_layout(1),
            cute.group_modes(sQ_full,  0, 2),
            cute.group_modes(gQ_tiles, 0, 2),
        )
        tKsK, tKgK = cute.nvgpu.cpasync.tma_partition(
            tma_atom_k, 0, cute.make_layout(1),
            cute.group_modes(sK_full,  0, 2),
            cute.group_modes(gK_tiles, 0, 2),
        )
        tVsV, tVgV = cute.nvgpu.cpasync.tma_partition(
            tma_atom_v, 0, cute.make_layout(1),
            cute.group_modes(sV_full,  0, 2),
            cute.group_modes(gV_tiles, 0, 2),
        )

        # ---- Pipeline state objects ----
        q_producer_state  = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.q_stage,
        )
        q_consumer_state  = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.q_stage,
        )
        kv_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.kv_stage,
        )
        kv_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.kv_stage,
        )

        # Make sure pipeline init writes are visible across CTAs before use.
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        # ---- Thread -> (row, col_group) mapping (same as K2) ----
        row         = tidx // self.threads_per_row     # 0..63: which Q-row
        col_group   = tidx %  self.threads_per_row     # 0..3:  which 1/4 of the row
        d_col_start = col_group * self.cols_per_thread_O

        # ---- Per-row softmax state in registers (replicated in row group) ----
        m_i = cutlass.Float32(-cutlass.Float32.inf)
        l_i = cutlass.Float32(0.0)
        O_acc = cute.make_rmem_tensor(
            cute.make_layout(self.cols_per_thread_O), ACC_DTYPE,
        )
        for c in cutlass.range_constexpr(self.cols_per_thread_O):
            O_acc[c] = cutlass.Float32(0.0)

        # =====================================================================
        # 1. Q load (one shot, reused across every KV iter)
        # =====================================================================
        if warp_idx == 0:
            q_pipeline.producer_acquire(q_producer_state)
            cute.copy(
                tma_atom_q,
                tQgQ[(None, bidx_m)],
                tQsQ[(None, q_producer_state.index)],
                tma_bar_ptr=q_pipeline.producer_get_barrier(q_producer_state),
            )
            q_pipeline.producer_commit(q_producer_state)   # no-op for TMA
            q_producer_state.advance()

        q_pipeline.consumer_wait(q_consumer_state)

        # =====================================================================
        # 2. KV mainloop
        # =====================================================================
        n_kv = self.N // self.Bc

        for j in cutlass.range(n_kv, unroll=1):

            # ---- 2a. Producer: issue K + V into the single KV stage ----
            if warp_idx == 0:
                kv_pipeline.producer_acquire(kv_producer_state)
                cute.copy(
                    tma_atom_k,
                    tKgK[(None, j)],
                    tKsK[(None, kv_producer_state.index)],
                    tma_bar_ptr=kv_pipeline.producer_get_barrier(kv_producer_state),
                )
                cute.copy(
                    tma_atom_v,
                    tVgV[(None, j)],
                    tVsV[(None, kv_producer_state.index)],
                    tma_bar_ptr=kv_pipeline.producer_get_barrier(kv_producer_state),
                )
                kv_pipeline.producer_commit(kv_producer_state)   # no-op for TMA
                kv_producer_state.advance()

            # ---- 2b. Consumer: wait for K + V, slice this stage's buffers ----
            kv_pipeline.consumer_wait(kv_consumer_state)
            sK = sK_full[(None, None, kv_consumer_state.index)]
            sV = sV_full[(None, None, kv_consumer_state.index)]

            # ---- 2c. S = Q @ K^T * scale (+ optional causal mask) ----
            for it in cutlass.range_constexpr(0, self.Br * self.Bc, self.threads):
                ij = it + tidx
                si, sj = ij // self.Bc, ij % self.Bc
                dot = cutlass.Float32(0.0)
                for k in cutlass.range_constexpr(self.d):
                    dot = dot + sQ[si, k].to(cutlass.Float32) \
                              * sK[sj, k].to(cutlass.Float32)
                val = dot * scale
                if cutlass.const_expr(self.causal):
                    q_abs  = bidx_m * self.Br + si
                    kv_abs = j      * self.Bc + sj
                    if kv_abs > q_abs:
                        val = -cutlass.Float32.inf
                sS[si, sj] = val
            cute.arch.sync_threads()

            # ---- 2d. Online softmax row update ----
            local_max = cutlass.Float32(-cutlass.Float32.inf)
            for c_it in cutlass.range_constexpr(self.cols_per_thread_S):
                c = col_group + c_it * self.threads_per_row
                local_max = cute.arch.fmax(local_max, sS[row, c])
            row_max = cute.arch.warp_reduction_max(
                local_max, threads_in_group=self.threads_per_row,
            )

            m_new = cute.arch.fmax(m_i, row_max)
            alpha = cute.math.exp(m_i - m_new, fastmath=True)

            for c in cutlass.range_constexpr(self.cols_per_thread_O):
                O_acc[c] = O_acc[c] * alpha

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

            cute.arch.sync_threads()   # sS now holds P; visible to PV

            # ---- 2e. O_acc += P @ V (each thread updates its d slice) ----
            for c in cutlass.range_constexpr(self.cols_per_thread_O):
                col = d_col_start + c
                pv = cutlass.Float32(0.0)
                for k in cutlass.range_constexpr(self.Bc):
                    pv = pv + sS[row, k] * sV[k, col].to(cutlass.Float32)
                O_acc[c] = O_acc[c] + pv

            cute.arch.sync_threads()   # V reads done; safe to release stage

            # ---- 2f. Consumer release: producer may refill this slot ----
            kv_pipeline.consumer_release(kv_consumer_state)
            kv_consumer_state.advance()

        # =====================================================================
        # 3. Finalize: O /= l, write to GMEM
        # =====================================================================
        inv_l = cute.arch.rcp_approx(l_i)
        if l_i == 0.0 or l_i != l_i:           # fully-masked row guard
            inv_l = cutlass.Float32(1.0)

        for c in cutlass.range_constexpr(self.cols_per_thread_O):
            col = d_col_start + c
            gO_tile[row, col] = (O_acc[c] * inv_l).to(OUTPUT_DTYPE)


def compile_kernel(B, N, d, causal, tensors):
    fmha = K3FMHA(B, N, d, causal)
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