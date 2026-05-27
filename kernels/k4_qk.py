"""K4.1: WGMMA for QK, hand-rolled scalar PV.

First half of the K4 progression. The QK matmul becomes one WGMMA call per
k-block on the tensor cores; the result is scattered back to SMEM via the
WGMMA C identity-tensor trick so the existing softmax (from K3) can read it
as a plain (Br, Bc) tile. The PV matmul stays scalar in registers.

What changes vs K3.3:
  * Threads: 256 -> 128 (one warpgroup, the unit WGMMA operates on).
  * threads_per_row: 4 -> 2 (128 / 64 rows).
  * SMEM layouts for Q and K are swizzled (built by sm90_utils.make_smem_layout_*).
    Required for WGMMA to access SMEM without bank conflicts.
  * QK is one WGMMA call per k-block: cute.gemm + ACCUMULATE flag + fence/commit/wait.
  * acc_qk (per-thread WGMMA C-fragment) is scattered to sS using a partitioned
    identity tensor that reports the (M, N) tile coord of each fragment slot.

What does NOT change vs K3.3:
  * TMA loads, pipeline structure, 5 KV stages.
  * Online softmax math.
  * Hand-rolled PV matmul (per-thread scalar FP32 over Bc).
"""
import math

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.hopper_helpers as sm90_utils
import cutlass.cute.nvgpu.warpgroup as warpgroup
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from utils import INPUT_DTYPE, OUTPUT_DTYPE, ACC_DTYPE, get_cuda_stream


Br = 64
Bc = 64
THREADS = 128                    # one warpgroup
THREADS_PER_ROW = 2              # 128 threads / 64 rows


class K4QKFMHA:
    def __init__(self, B, N, d, causal):
        self.B = B
        self.N = N
        self.d = d
        self.causal = causal

        self.Br = Br
        self.Bc = Bc
        self.threads = THREADS
        self.threads_per_row = THREADS_PER_ROW
        self.cols_per_thread_O = d  // THREADS_PER_ROW
        self.cols_per_thread_S = Bc // THREADS_PER_ROW
        self.buffer_align_bytes = 1024

        self.q_stage = 1
        self.kv_stage = 5

        self.cluster_shape_mn = (1, 1)
        self.threads_per_cta = self.threads
        self.atom_layout_mnk = (1, 1, 1)        # one atom per warpgroup

    # -------------------------------------------------------------------------
    # Layout helpers: split a hierarchical mode into M-walking and N-walking
    # parts using stride < M as the test. layout_acc_mn builds an mn-view of
    # the per-thread C fragment so we can sweep its M rows / N cols cleanly.
    # See K4.2 for the deeper write-up; K4.1 only uses these for the scatter.
    # -------------------------------------------------------------------------
    @staticmethod
    def layout_separate(thr, src, ref):
        lt = cute.make_layout(())
        ge = cute.make_layout(())
        for k, v in enumerate(ref):
            if cutlass.const_expr(v < thr):
                lt = cute.append(lt, src[k])
            else:
                ge = cute.append(ge, src[k])
        if cutlass.const_expr(cute.rank(lt) == 1):
            return cute.append(lt, ge)
        else:
            return cute.append(cute.append(cute.make_layout(()), lt), ge)

    @cute.jit
    def layout_acc_mn(self, tiled_mma, acc):
        separated = self.layout_separate(
            tiled_mma.shape_mnk[0], acc[0], tiled_mma.tv_layout_C.stride[1],
        )
        V_M = separated[0]
        V_N = separated[1]
        if cutlass.const_expr(cute.rank(V_M) == 1):
            V_M1 = cute.append(V_M, acc[1])
        else:
            V_M1 = cute.append(cute.append(cute.make_layout(()), V_M), acc[1])
        if cutlass.const_expr(cute.rank(V_N) == 1):
            V_N1 = cute.append(V_N, acc[2])
        else:
            V_N1 = cute.append(cute.append(cute.make_layout(()), V_N), acc[2])
        if cutlass.const_expr(cute.rank(V_M1) == 1):
            return cute.append(V_M1, V_N1)
        else:
            return cute.append(cute.append(cute.make_layout(()), V_M1), V_N1)

    @cute.jit
    def __call__(self, mQ, mK, mV, mO, scale: cutlass.Float32, stream):
        q_layout_enum = utils.LayoutEnum.from_tensor(mQ)
        k_layout_enum = utils.LayoutEnum.from_tensor(mK)

        mma_tile_mnk = (self.Br, self.Bc, self.d)

        # ---- Swizzled SMEM layouts for Q, K (V kept plain for hand-rolled PV) ----
        sQ_layout_staged = sm90_utils.make_smem_layout_a(
            q_layout_enum, mma_tile_mnk, INPUT_DTYPE, self.q_stage,
        )
        sK_layout_staged = sm90_utils.make_smem_layout_b(
            k_layout_enum, mma_tile_mnk, INPUT_DTYPE, self.kv_stage,
        )
        sV_layout_staged = sm90_utils.make_smem_layout_b(
            k_layout_enum, mma_tile_mnk, INPUT_DTYPE, self.kv_stage,
        )

        sQ_layout = cute.slice_(sQ_layout_staged, (None, None, 0))
        sK_layout = cute.slice_(sK_layout_staged, (None, None, 0))
        sV_layout = cute.slice_(sV_layout_staged, (None, None, 0))

        sS_layout = cute.make_layout((self.Br, self.Bc), stride=(self.Bc, 1))

        # ---- QK tiled MMA: A=Q (SMEM, K-major), B=K (SMEM, K-major), C=fp32 acc ----
        qk_tiled_mma = sm90_utils.make_trivial_tiled_mma(
            INPUT_DTYPE, INPUT_DTYPE,
            warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K,
            ACC_DTYPE,
            self.atom_layout_mnk,
            (self.Br, self.Bc),
            warpgroup.OperandSource.SMEM,
        )

        # ---- TMA atoms (same as K3.3) ----
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

        q_bytes  = cute.size_in_bytes(INPUT_DTYPE, sQ_layout)
        kv_bytes = (cute.size_in_bytes(INPUT_DTYPE, sK_layout)
                    + cute.size_in_bytes(INPUT_DTYPE, sV_layout))

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
            qk_tiled_mma,
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
        qk_tiled_mma,
        sQ_layout_staged, sK_layout_staged, sV_layout_staged, sS_layout,
        scale: cutlass.Float32,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)

        bidx_m, bidx_b, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # ---- Pipelines (same as K3.3) ----
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

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # ---- SMEM tensors with swizzle ----
        sQ_full = storage.sQ.get_tensor(
            sQ_layout_staged.outer, swizzle=sQ_layout_staged.inner,
        )
        sK_full = storage.sK.get_tensor(
            sK_layout_staged.outer, swizzle=sK_layout_staged.inner,
        )
        sV_full = storage.sV.get_tensor(
            sV_layout_staged.outer, swizzle=sV_layout_staged.inner,
        )
        sS = storage.sS.get_tensor(sS_layout)

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

        # ---- WGMMA partitioning for QK (both A and B from SMEM) ----
        qk_thr_mma = qk_tiled_mma.get_slice(tidx)
        tSsQ = qk_thr_mma.partition_A(sQ_full)             # (MMA, MMA_M, MMA_K, q_stage)
        tSsK = qk_thr_mma.partition_B(sK_full)             # (MMA, MMA_N, MMA_K, kv_stage)
        tSrQ = qk_tiled_mma.make_fragment_A(tSsQ)          # layout reinterpretation
        tSrK = qk_tiled_mma.make_fragment_B(tSsK)          # layout reinterpretation
        qk_acc_shape = qk_thr_mma.partition_shape_C((self.Br, self.Bc))

        # Identity tensor for the acc_qk -> sS scatter. partition_C(cS) reports
        # the (M, N) tile coord of each per-thread C-fragment slot, indexed the
        # same way as acc_qk's mn-view.
        cS = cute.make_identity_tensor((self.Br, self.Bc))
        tScS = qk_thr_mma.partition_C(cS)

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

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

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
            q_pipeline.producer_commit(q_producer_state)
            q_producer_state.advance()

        q_pipeline.consumer_wait(q_consumer_state)

        # ---- Thread -> (row, col_group) for softmax + PV ----
        row         = tidx // self.threads_per_row     # 0..63
        col_group   = tidx %  self.threads_per_row     # 0..1
        d_col_start = col_group * self.cols_per_thread_O

        # ---- Per-row softmax state in registers ----
        m_i = cutlass.Float32(-cutlass.Float32.inf)
        l_i = cutlass.Float32(0.0)
        O_acc = cute.make_rmem_tensor(
            cute.make_layout(self.cols_per_thread_O), ACC_DTYPE,
        )
        for c in cutlass.range_constexpr(self.cols_per_thread_O):
            O_acc[c] = cutlass.Float32(0.0)

        n_kv = self.N // self.Bc

        # =====================================================================
        # 2. KV prefetch (fill the pipeline)
        # =====================================================================
        prefetch_count = cutlass.min(self.kv_stage, n_kv)
        if warp_idx == 0:
            for _ in cutlass.range(prefetch_count, unroll=1):
                kv_pipeline.producer_acquire(kv_producer_state)
                cute.copy(
                    tma_atom_k,
                    tKgK[(None, kv_producer_state.count)],
                    tKsK[(None, kv_producer_state.index)],
                    tma_bar_ptr=kv_pipeline.producer_get_barrier(kv_producer_state),
                )
                cute.copy(
                    tma_atom_v,
                    tVgV[(None, kv_producer_state.count)],
                    tVsV[(None, kv_producer_state.index)],
                    tma_bar_ptr=kv_pipeline.producer_get_barrier(kv_producer_state),
                )
                kv_pipeline.producer_commit(kv_producer_state)
                kv_producer_state.advance()

        # =====================================================================
        # 3. KV mainloop
        # =====================================================================
        for j in cutlass.range(n_kv, unroll=1):
            kv_pipeline.consumer_wait(kv_consumer_state)
            sV = sV_full[(None, None, kv_consumer_state.index)]

            # ---- 3a. WGMMA QK: S = Q @ K^T (FP32 acc, no scaling yet) ----
            acc_qk = qk_thr_mma.make_fragment_C(qk_acc_shape)
            cute.nvgpu.warpgroup.fence()

            tSrQ_k = tSrQ[(None, None, None, q_consumer_state.index)]
            tSrK_k = tSrK[(None, None, None, kv_consumer_state.index)]
            num_k_blocks = cute.size(tSrQ_k, mode=[2])
            for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                qk_tiled_mma.set(
                    cute.nvgpu.warpgroup.Field.ACCUMULATE, k_block_idx != 0,
                )
                cute.gemm(
                    qk_tiled_mma, acc_qk,
                    tSrQ_k[(None, None, k_block_idx)],
                    tSrK_k[(None, None, k_block_idx)],
                    acc_qk,
                )
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)

            # ---- 3b. Scatter acc_qk -> sS (with scale + causal) ----
            acc_qk_mn = cute.make_tensor(
                acc_qk.iterator,
                self.layout_acc_mn(qk_tiled_mma, acc_qk.layout),
            )
            tScS_mn = cute.make_tensor(
                tScS.iterator,
                self.layout_acc_mn(qk_tiled_mma, tScS.layout),
            )

            n_rows_qk = cute.size(acc_qk_mn, mode=[0])
            n_cols_qk = cute.size(acc_qk_mn, mode=[1])
            for i in cutlass.range_constexpr(n_rows_qk):
                for jj in cutlass.range_constexpr(n_cols_qk):
                    m_idx = tScS_mn[i, jj][0]
                    n_idx = tScS_mn[i, jj][1]
                    val = acc_qk_mn[i, jj] * scale
                    if cutlass.const_expr(self.causal):
                        q_abs  = bidx_m * self.Br + m_idx
                        kv_abs = j      * self.Bc + n_idx
                        if kv_abs > q_abs:
                            val = -cutlass.Float32.inf
                    sS[m_idx, n_idx] = val
            cute.arch.sync_threads()

            # ---- 3c. Online softmax row update (same as K3, threads_per_row=2) ----
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

            cute.arch.sync_threads()

            # ---- 3d. Hand-rolled O_acc += P @ V (same as K3) ----
            for c in cutlass.range_constexpr(self.cols_per_thread_O):
                col = d_col_start + c
                pv = cutlass.Float32(0.0)
                for k in cutlass.range_constexpr(self.Bc):
                    pv = pv + sS[row, k] * sV[k, col].to(cutlass.Float32)
                O_acc[c] = O_acc[c] + pv

            cute.arch.sync_threads()

            kv_pipeline.consumer_release(kv_consumer_state)
            kv_consumer_state.advance()

            # ---- 3e. Issue next KV prefetch ----
            if warp_idx == 0 and kv_producer_state.count < n_kv:
                kv_pipeline.producer_acquire(kv_producer_state)
                cute.copy(
                    tma_atom_k,
                    tKgK[(None, kv_producer_state.count)],
                    tKsK[(None, kv_producer_state.index)],
                    tma_bar_ptr=kv_pipeline.producer_get_barrier(kv_producer_state),
                )
                cute.copy(
                    tma_atom_v,
                    tVgV[(None, kv_producer_state.count)],
                    tVsV[(None, kv_producer_state.index)],
                    tma_bar_ptr=kv_pipeline.producer_get_barrier(kv_producer_state),
                )
                kv_pipeline.producer_commit(kv_producer_state)
                kv_producer_state.advance()

        # =====================================================================
        # 4. Finalize: O /= l, write to GMEM
        # =====================================================================
        inv_l = cute.arch.rcp_approx(l_i)
        if l_i == 0.0 or l_i != l_i:
            inv_l = cutlass.Float32(1.0)

        for c in cutlass.range_constexpr(self.cols_per_thread_O):
            col = d_col_start + c
            gO_tile[row, col] = (O_acc[c] * inv_l).to(OUTPUT_DTYPE)


def compile_kernel(B, N, d, causal, tensors):
    fmha = K4QKFMHA(B, N, d, causal)
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