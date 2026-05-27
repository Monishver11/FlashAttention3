"""K4.2: Full WGMMA (QK + PV) with register-resident P handoff.

Both matmuls run on tensor cores. P (softmax output) stays in registers under
the WGMMA C-fragment layout and is reinterpreted as PV's A-operand register
fragment in place (no SMEM round-trip). The online softmax runs directly on
acc_qk's per-thread fragment, with cross-thread row reductions over the
WGMMA C TV layout's row-sharing sibling threads.

V is re-viewed at kernel entry with mode order (d, k, l) instead of (k, d, l).
The bytes don't move; CuTe now sees V as COL_MAJOR with d as the leading axis,
which makes V MN-major for PV's B operand (its contiguous axis d is PV's N
axis, not K). See the V-mode-order callout in the blog.

What changes vs K4.1:
  * PV is a WGMMA call too (cute.gemm with A=RMEM, B=SMEM).
  * P flows through registers: acc_qk -> make_acc_into_op -> PV's A operand.
  * No sS tile in SMEM; the entire softmax runs on the per-thread fragment.
  * Per-row state arrays sized to cute.size(acc_pv_mn, mode=[0]) = n_rows
    (typically 2 for SM90 BF16 64x64 atom), not a scalar.
  * Cross-thread row reductions use reduction_target_n(tiled_mma).
  * acc_pv is allocated once before the mainloop, zero-initialized, and
    accumulates across all KV iterations.
  * Output scatter uses a coord-tensor probe via partition_C on an identity.
  * V's GMEM view is transposed; V's SMEM layout is MN-major; V's TMA tile
    is (d, Bc); V's flat_divide slice indexing matches the new mode order.

What does NOT change vs K4.1:
  * Threading: 128 threads (one warpgroup).
  * Pipeline: 5 KV stages, TMA loads, Q one-shot.
  * QK tiled MMA shape and major modes.
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
THREADS = 128


class K4FMHA:
    def __init__(self, B, N, d, causal):
        self.B = B
        self.N = N
        self.d = d
        self.causal = causal

        self.Br = Br
        self.Bc = Bc
        self.threads = THREADS
        self.buffer_align_bytes = 1024

        self.q_stage = 1
        self.kv_stage = 5

        self.cluster_shape_mn = (1, 1)
        self.threads_per_cta = self.threads
        self.atom_layout_mnk = (1, 1, 1)

    # -------------------------------------------------------------------------
    # Layout helpers.
    #
    # layout_separate: peel a hierarchical mode into (M-walking, N-walking)
    # parts using stride < M as the test. Used by both layout_acc_mn (value
    # mode) and reduction_target_n (thread mode).
    #
    # layout_acc_mn: produce an mn-view of the per-thread C fragment where
    # mode 0 walks the thread's distinct M rows and mode 1 walks its distinct
    # N columns.
    #
    # reduction_target_n: return the part of the thread mode whose stride is
    # >= M, i.e. the threads that share a row in the WGMMA C-partition.
    # warp_reduction over these threads correctly reduces a row across all
    # columns of that row.
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
    def reduction_target_n(self, tiled_mma):
        """Threads that share a row in the WGMMA C-partition: the N-walking
        part of the thread mode."""
        separated = self.layout_separate(
            tiled_mma.shape_mnk[0],
            cute.make_layout(tiled_mma.tv_layout_C.shape[0]),
            tiled_mma.tv_layout_C.stride[0],
        )
        return separated[1]

    # -------------------------------------------------------------------------
    # Register-handoff helpers: reinterpret the WGMMA C-fragment of acc_qk as
    # the WGMMA A-fragment for the PV matmul, in place. BF16 only; FP8 (K7)
    # needs additional cross-thread shuffles on top of this.
    # -------------------------------------------------------------------------
    @staticmethod
    def convert_c_layout_to_a_layout(c, a):
        return cute.make_layout(
            (a, c.shape[1], (c.shape[2], cute.size(c, mode=[0]) // cute.size(a))),
            stride=(
                c.stride[0],
                c.stride[1],
                (c.stride[2], cute.size(a, mode=[2]) * c.stride[0][2]),
            ),
        )

    @cute.jit
    def make_acc_into_op(self, acc, operand_layout_tv, Element):
        operand = cute.make_rmem_tensor_like(
            self.convert_c_layout_to_a_layout(acc.layout, operand_layout_tv.shape[1]),
            Element,
        )
        operand_as_acc = cute.make_tensor(operand.iterator, acc.layout)
        acc_vec = acc.load()
        operand_as_acc.store(acc_vec.to(Element))
        return operand

    @cute.jit
    def __call__(self, mQ, mK, mV, mO, scale: cutlass.Float32, stream):
        # V re-view: (k, d, l) -> (d, k, l). Same bytes, leading axis is now d.
        # CuTe now reports V as COL_MAJOR; sm90_mma_major_mode() returns MN.
        mV = cute.make_tensor(mV.iterator, cute.select(mV.layout, [1, 0, 2]))

        q_layout_enum = utils.LayoutEnum.from_tensor(mQ)
        k_layout_enum = utils.LayoutEnum.from_tensor(mK)
        v_layout_enum = utils.LayoutEnum.from_tensor(mV)

        mma_tile_mnk    = (self.Br, self.Bc, self.d)
        pv_mma_tile_mnk = (self.Br, self.d,  self.Bc)

        # ---- Swizzled SMEM layouts ----
        sQ_layout_staged = sm90_utils.make_smem_layout_a(
            q_layout_enum, mma_tile_mnk, INPUT_DTYPE, self.q_stage,
        )
        sK_layout_staged = sm90_utils.make_smem_layout_b(
            k_layout_enum, mma_tile_mnk, INPUT_DTYPE, self.kv_stage,
        )
        # V uses pv_mma_tile_mnk = (Br, d, Bc); make_smem_layout_b takes B's
        # (N, K) which is (d, Bc), matching the re-viewed mV's (d, k, l) shape.
        sV_layout_staged = sm90_utils.make_smem_layout_b(
            v_layout_enum, pv_mma_tile_mnk, INPUT_DTYPE, self.kv_stage,
        )

        sQ_layout = cute.slice_(sQ_layout_staged, (None, None, 0))
        sK_layout = cute.slice_(sK_layout_staged, (None, None, 0))
        sV_layout = cute.slice_(sV_layout_staged, (None, None, 0))

        # ---- Tiled MMAs ----
        # QK: A=Q (SMEM, K-major), B=K (SMEM, K-major), acc=FP32.
        qk_tiled_mma = sm90_utils.make_trivial_tiled_mma(
            INPUT_DTYPE, INPUT_DTYPE,
            warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K,
            ACC_DTYPE,
            self.atom_layout_mnk,
            (self.Br, self.Bc),
            warpgroup.OperandSource.SMEM,
        )
        # PV: A=P (RMEM, K-major), B=V (SMEM, MN-major from the re-view),
        # acc=FP32. acc_pv persists across KV iters.
        pv_tiled_mma = sm90_utils.make_trivial_tiled_mma(
            INPUT_DTYPE, INPUT_DTYPE,
            warpgroup.OperandMajorMode.K,
            v_layout_enum.sm90_mma_major_mode(),
            ACC_DTYPE,
            self.atom_layout_mnk,
            (self.Br, self.d),
            warpgroup.OperandSource.RMEM,
        )

        # ---- TMA atoms ----
        tma_atom_q, tma_tensor_q = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            mQ, sQ_layout, (self.Br, self.d), num_multicast=1,
        )
        tma_atom_k, tma_tensor_k = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            mK, sK_layout, (self.Bc, self.d), num_multicast=1,
        )
        # V's TMA tile is (d, Bc) to match the re-viewed (d, k, l) shape.
        tma_atom_v, tma_tensor_v = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            mV, sV_layout, (self.d, self.Bc), num_multicast=1,
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
        self.shared_storage = SharedStorage
        self.q_bytes  = q_bytes
        self.kv_bytes = kv_bytes

        self.kernel(
            tma_atom_q, tma_tensor_q,
            tma_atom_k, tma_tensor_k,
            tma_atom_v, tma_tensor_v,
            mO,
            qk_tiled_mma, pv_tiled_mma,
            sQ_layout_staged, sK_layout_staged, sV_layout_staged,
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
        qk_tiled_mma, pv_tiled_mma,
        sQ_layout_staged, sK_layout_staged, sV_layout_staged,
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

        # ---- Pipelines ----
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

        # ---- SMEM tensors (with swizzle) ----
        sQ_full = storage.sQ.get_tensor(
            sQ_layout_staged.outer, swizzle=sQ_layout_staged.inner,
        )
        sK_full = storage.sK.get_tensor(
            sK_layout_staged.outer, swizzle=sK_layout_staged.inner,
        )
        sV_full = storage.sV.get_tensor(
            sV_layout_staged.outer, swizzle=sV_layout_staged.inner,
        )

        # ---- GMEM tile views ----
        gQ = cute.flat_divide(tma_tensor_q, (self.Br, self.d))
        gK = cute.flat_divide(tma_tensor_k, (self.Bc, self.d))
        # V is (d, k, l) divided by (d, Bc); the new modes are
        # (d_inner, Bc_inner, d_tile, k_tile, l). d_tile=0 (only one).
        gV = cute.flat_divide(tma_tensor_v, (self.d, self.Bc))
        gO = cute.flat_divide(mO,           (self.Br, self.d))

        gQ_tiles = gQ[(None, None, None, 0, bidx_b)]
        gK_tiles = gK[(None, None, None, 0, bidx_b)]
        gV_tiles = gV[(None, None, 0, None, bidx_b)]   # d_tile=0, iterate k_tile
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

        # ---- WGMMA QK partitioning (A and B from SMEM) ----
        qk_thr_mma = qk_tiled_mma.get_slice(tidx)
        tSsQ = qk_thr_mma.partition_A(sQ_full)
        tSsK = qk_thr_mma.partition_B(sK_full)
        tSrQ = qk_tiled_mma.make_fragment_A(tSsQ)
        tSrK = qk_tiled_mma.make_fragment_B(tSsK)
        qk_acc_shape = qk_thr_mma.partition_shape_C((self.Br, self.Bc))

        # ---- WGMMA PV partitioning (only B = sV is from SMEM; A from RMEM) ----
        pv_thr_mma = pv_tiled_mma.get_slice(tidx)
        tOsV = pv_thr_mma.partition_B(sV_full)
        tOrV = pv_tiled_mma.make_fragment_B(tOsV)
        pv_acc_shape = pv_thr_mma.partition_shape_C((self.Br, self.d))

        # ---- Persistent acc_pv + zero-init ----
        # Allocated once before the mainloop. Zero-init makes the unified
        # rescale benign on iter 0: scale_pv = exp(-inf - x) = 0, 0*0 = 0.
        acc_pv = pv_thr_mma.make_fragment_C(pv_acc_shape)
        acc_pv_mn = cute.make_tensor(
            acc_pv.iterator,
            self.layout_acc_mn(pv_tiled_mma, acc_pv.layout),
        )
        n_rows    = cute.size(acc_pv_mn, mode=[0])
        n_cols_pv = cute.size(acc_pv_mn, mode=[1])
        for i in cutlass.range_constexpr(n_rows):
            for jj in cutlass.range_constexpr(n_cols_pv):
                acc_pv_mn[i, jj] = cutlass.Float32(0.0)

        # ---- Per-row softmax state in registers ----
        s_max_layout = cute.make_layout(n_rows)
        s_max      = cute.make_rmem_tensor_like(s_max_layout, cutlass.Float32)
        a_sum      = cute.make_rmem_tensor_like(s_max, cutlass.Float32)
        s_max_prev = cute.make_rmem_tensor_like(s_max, cutlass.Float32)
        for i in cutlass.range_constexpr(n_rows):
            s_max[i] = -cutlass.Float32.inf
            a_sum[i] = cutlass.Float32(0.0)

        reduction_target_qk = self.reduction_target_n(qk_tiled_mma)
        red_rank_qk = cute.rank(reduction_target_qk)

        # ---- QK identity tensor for absolute (M, N) coords (causal mask) ----
        cS = cute.make_identity_tensor((self.Br, self.Bc))
        tScS = qk_thr_mma.partition_C(cS)
        tScS_mn = cute.make_tensor(
            tScS.iterator,
            self.layout_acc_mn(qk_tiled_mma, tScS.layout),
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

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        # =====================================================================
        # 1. Q load (one-shot)
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

        n_kv = self.N // self.Bc

        # =====================================================================
        # 2. KV prefetch
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

            # ---- 3a. WGMMA QK ----
            acc_qk = qk_thr_mma.make_fragment_C(qk_acc_shape)
            cute.nvgpu.warpgroup.fence()
            tSrQ_k = tSrQ[(None, None, None, q_consumer_state.index)]
            tSrK_k = tSrK[(None, None, None, kv_consumer_state.index)]
            num_k_blocks_qk = cute.size(tSrQ_k, mode=[2])
            for k_block_idx in cutlass.range_constexpr(num_k_blocks_qk):
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

            # ---- 3b. Softmax in registers, on the WGMMA C fragment ----
            acc_qk_mn = cute.make_tensor(
                acc_qk.iterator,
                self.layout_acc_mn(qk_tiled_mma, acc_qk.layout),
            )
            n_cols_qk = cute.size(acc_qk_mn, mode=[1])

            # Scale.
            for i in cutlass.range_constexpr(n_rows):
                for jj in cutlass.range_constexpr(n_cols_qk):
                    acc_qk_mn[i, jj] = acc_qk_mn[i, jj] * scale

            # Causal mask.
            if cutlass.const_expr(self.causal):
                q_block_offset  = bidx_m * self.Br
                kv_block_offset = j      * self.Bc
                for i in cutlass.range_constexpr(n_rows):
                    for jj in cutlass.range_constexpr(n_cols_qk):
                        q_abs  = q_block_offset  + tScS_mn[i, jj][0]
                        kv_abs = kv_block_offset + tScS_mn[i, jj][1]
                        if kv_abs > q_abs:
                            acc_qk_mn[i, jj] = -cutlass.Float32.inf

            # Save prev row max, take new local row max, cross-thread reduce.
            for i in cutlass.range_constexpr(n_rows):
                s_max_prev[i] = s_max[i]
                for jj in cutlass.range_constexpr(n_cols_qk):
                    s_max[i] = cute.arch.fmax(s_max[i], acc_qk_mn[i, jj])
            for r in cutlass.range_constexpr(red_rank_qk):
                for i in cutlass.range_constexpr(n_rows):
                    s_max[i] = cute.arch.warp_reduction_max(
                        s_max[i],
                        threads_in_group=reduction_target_qk.shape[r],
                    )

            # Per-row rescale of acc_pv and a_sum.
            for i in cutlass.range_constexpr(n_rows):
                scale_pv = cute.math.exp(
                    s_max_prev[i] - s_max[i], fastmath=True,
                )
                a_sum[i] = a_sum[i] * scale_pv
                for jj in cutlass.range_constexpr(n_cols_pv):
                    acc_pv_mn[i, jj] = acc_pv_mn[i, jj] * scale_pv

            # P = exp(S - s_max) in-place into acc_qk, accumulate local row sum.
            for i in cutlass.range_constexpr(n_rows):
                for jj in cutlass.range_constexpr(n_cols_qk):
                    p = cute.math.exp(
                        acc_qk_mn[i, jj] - s_max[i], fastmath=True,
                    )
                    acc_qk_mn[i, jj] = p
                    a_sum[i] = a_sum[i] + p

            # ---- 3c. QK -> PV register handoff: acc_qk (FP32, C-layout)
            #          -> tOrP (BF16, A-layout) ----
            tOrP = self.make_acc_into_op(
                acc_qk, pv_tiled_mma.tv_layout_A, INPUT_DTYPE,
            )

            # ---- 3d. WGMMA PV: A from RMEM (tOrP), B from SMEM (tOrV) ----
            cute.nvgpu.warpgroup.fence()
            tOrV_k = tOrV[(None, None, None, kv_consumer_state.index)]
            num_k_blocks_pv = cute.size(tOrP, mode=[2])
            for k_block_idx in cutlass.range_constexpr(num_k_blocks_pv):
                pv_tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                cute.gemm(
                    pv_tiled_mma, acc_pv,
                    tOrP[(None, None, k_block_idx)],
                    tOrV_k[(None, None, k_block_idx)],
                    acc_pv,
                )
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)

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
        # 4. Finalize: cross-thread reduce a_sum, divide acc_pv, scatter to gO
        # =====================================================================
        reduction_target_pv = self.reduction_target_n(pv_tiled_mma)
        red_rank_pv = cute.rank(reduction_target_pv)
        for r in cutlass.range_constexpr(red_rank_pv):
            for i in cutlass.range_constexpr(n_rows):
                a_sum[i] = cute.arch.warp_reduction_sum(
                    a_sum[i],
                    threads_in_group=reduction_target_pv.shape[r],
                )

        # PV identity tensor for the output scatter.
        cO = cute.make_identity_tensor((self.Br, self.d))
        tOcO = pv_thr_mma.partition_C(cO)
        tOcO_mn = cute.make_tensor(
            tOcO.iterator,
            self.layout_acc_mn(pv_tiled_mma, tOcO.layout),
        )
        for i in cutlass.range_constexpr(n_rows):
            inv_l = cute.arch.rcp_approx(a_sum[i])
            if a_sum[i] == 0.0 or a_sum[i] != a_sum[i]:    # fully-masked row guard
                inv_l = cutlass.Float32(1.0)
            for jj in cutlass.range_constexpr(n_cols_pv):
                m_idx = tOcO_mn[i, jj][0]
                n_idx = tOcO_mn[i, jj][1]
                gO_tile[m_idx, n_idx] = (acc_pv_mn[i, jj] * inv_l).to(OUTPUT_DTYPE)


def compile_kernel(B, N, d, causal, tensors):
    fmha = K4FMHA(B, N, d, causal)
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