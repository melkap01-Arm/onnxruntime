// Copyright (c) 2026 Arm Limited. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#include "mlasi_kleidiai.h"

#if !defined(DISABLE_FLOAT8_TYPES) && defined(MLAS_USE_KLEIDIAI_FP8)

#include <arm_sme.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "core/common/fp8_common.h"
#include "kai/kai_common.h"
#include "kai/ukernels/matmul/kai_matmul.h"
#include "kai/ukernels/matmul/kai_matmul_pack_lhs.h"
#include "kai/ukernels/matmul/kai_matmul_pack_rhs.h"

namespace {

constexpr size_t kr = 4;

bool IsSupportedFp8Mode(mlas_fp8_mode fp8_type) {
    switch (fp8_type) {
        case MLAS_FP8_MODE_E4M3_INF:
        case MLAS_FP8_MODE_E5M2_INF:
            return true;
        case MLAS_FP8_MODE_E4M3_SAT:
        case MLAS_FP8_MODE_E5M2_SAT:
            // ORT maps SAT modes to ONNX FNUZ types. KleidiAI SAT modes are overflow policies for
            // the non-FNUZ byte formats, so pre-quantized ONNX FNUZ bytes would be reinterpreted
            // incorrectly by the SME2 kernel. Use the generic MLAS fallback for FNUZ correctness.
        default:
            return false;
    }
}

bool IsFp8RhsPackSupported(
    size_t N,
    size_t K,
    size_t BlockSizeK,
    size_t BlockSizeN,
    mlas_fp8_mode Fp8Type
    )
{
    if (!ArmKleidiAI::UseSME2FP8 || !IsSupportedFp8Mode(Fp8Type) || N == 0 || K == 0 ||
        BlockSizeK == 0 || BlockSizeN == 0 || BlockSizeK != BlockSizeN) {
        return false;
    }

    const size_t vector_length_u32 = static_cast<size_t>(kai_get_sme_vector_length_u8() / sizeof(uint32_t));
    if (vector_length_u32 == 0) {
        return false;
    }

    const size_t n_block = 4 * vector_length_u32;
    if (K % kr != 0 ||
        K % BlockSizeK != 0 ||
        N % BlockSizeN != 0 ||
        N % n_block != 0 ||
        BlockSizeK % 32 != 0 ||
        BlockSizeK % n_block != 0) {
        return false;
    }

    return true;
}

enum kai_f8_mode ToKaiFp8Mode(mlas_fp8_mode fp8_type) {
    switch (fp8_type) {
        case MLAS_FP8_MODE_E4M3_INF:
            return KAI_F8_MODE_E4M3_INF;
        case MLAS_FP8_MODE_E4M3_SAT:
            return KAI_F8_MODE_E4M3_SAT;
        case MLAS_FP8_MODE_E5M2_INF:
            return KAI_F8_MODE_E5M2_INF;
        case MLAS_FP8_MODE_E5M2_SAT:
            return KAI_F8_MODE_E5M2_SAT;
        default:
            return KAI_F8_MODE_END;
    }
}

mlas_fp8_mode ToMlasFp8Mode(kai_f8_mode fp8_type) {
    switch (fp8_type) {
        case KAI_F8_MODE_E4M3_INF:
            return MLAS_FP8_MODE_E4M3_INF;
        case KAI_F8_MODE_E4M3_SAT:
            return MLAS_FP8_MODE_E4M3_SAT;
        case KAI_F8_MODE_E5M2_INF:
            return MLAS_FP8_MODE_E5M2_INF;
        case KAI_F8_MODE_E5M2_SAT:
            return MLAS_FP8_MODE_E5M2_SAT;
        default:
            return MLAS_FP8_MODE_E4M3_INF;
    }
}

void kai_quant_f8_f32(
    const float* src,
    size_t rows,
    size_t cols,
    size_t src_stride,
    uint8_t* dst,
    size_t dst_stride,
    float* dst_scales,
    size_t scales_block_row_stride,
    size_t scales_block_col_stride,
    size_t block_size_rows,
    size_t block_size_cols,
    kai_f8_mode f8_mode
    )
{
    const float fp8_max_abs = kai_get_abs_max_f8(f8_mode);
    const mlas_fp8_mode mlas_mode = ToMlasFp8Mode(f8_mode);

    for (size_t block_row = 0; block_row * block_size_rows < rows; ++block_row) {
        const size_t row_begin = block_row * block_size_rows;
        const size_t row_end = std::min(rows, row_begin + block_size_rows);
        for (size_t block_col = 0; block_col * block_size_cols < cols; ++block_col) {
            const size_t col_begin = block_col * block_size_cols;
            const size_t col_end = std::min(cols, col_begin + block_size_cols);

            float max_abs = 0.0f;
            for (size_t row = row_begin; row < row_end; ++row) {
                for (size_t col = col_begin; col < col_end; ++col) {
                    max_abs = std::max(max_abs, std::fabs(src[row * src_stride + col]));
                }
            }

            const float scale = max_abs == 0.0f ? 1.0f : max_abs / fp8_max_abs;
            dst_scales[block_row * scales_block_row_stride + block_col * scales_block_col_stride] = scale;

            for (size_t row = row_begin; row < row_end; ++row) {
                for (size_t col = col_begin; col < col_end; ++col) {
                    dst[row * dst_stride + col] =
                        onnxruntime::FloatToFp8Byte(src[row * src_stride + col] / scale, mlas_mode);
                }
            }
        }
    }
}

bool IsFp8GemmSupported(
    const MLAS_FP8_GEMM_SHAPE_PARAMS& Shape,
    const MLAS_FP8_GEMM_DATA_PARAMS* DataParams,
    size_t BatchSize
    )
{
    if (!ArmKleidiAI::UseSME2FP8 || DataParams == nullptr || BatchSize == 0 ||
        Shape.M == 0 || Shape.N == 0 || Shape.K == 0) {
        return false;
    }

    const size_t vector_length_u32 = static_cast<size_t>(kai_get_sme_vector_length_u8() / sizeof(uint32_t));
    if (vector_length_u32 == 0 || Shape.M % vector_length_u32 != 0) {
        return false;
    }

    for (size_t batch = 0; batch < BatchSize; ++batch) {
        const auto& params = DataParams[batch];
        if (params.BlockSizeK != DataParams[0].BlockSizeK ||
            params.BlockSizeN != DataParams[0].BlockSizeN ||
            params.Fp8Type != DataParams[0].Fp8Type) {
            return false;
        }

        size_t scratch_size = 0;
        if (MlasMultiplyOverflowsSizeT(Shape.M, Shape.K, &scratch_size) ||
            MlasMultiplyOverflowsSizeT(Shape.M, params.BlocksK, &scratch_size)) {
            return false;
        }

        if (params.AFloat == nullptr || params.PackedB == nullptr || params.C == nullptr ||
            params.PackedScaleB == nullptr || params.BlockSizeM != 1 ||
            params.lda != Shape.K || params.ldc < Shape.N ||
            params.ldc > std::numeric_limits<size_t>::max() / sizeof(float) ||
            !IsFp8RhsPackSupported(Shape.N, Shape.K, params.BlockSizeK, params.BlockSizeN, params.Fp8Type)) {
            return false;
        }

        size_t output_offset = 0;
        if (MlasMultiplyOverflowsSizeT(Shape.M - 1, params.ldc, &output_offset) ||
            Shape.N - 1 > std::numeric_limits<size_t>::max() - output_offset) {
            return false;
        }

        const size_t blocks_m = ((Shape.M - 1) / params.BlockSizeM) + 1;
        const size_t blocks_k = Shape.K / params.BlockSizeK;
        const size_t blocks_n = Shape.N / params.BlockSizeN;
        if (params.BlocksM != blocks_m || params.BlocksK != blocks_k || params.BlocksN != blocks_n) {
            return false;
        }
    }

    return true;
}

size_t GetRhsPackedSize(size_t N, size_t K) {
    const auto rhs_pack = kai_matmul_pack_rhs_kxn_x8p4vsx4_x8_sme();
    kai_matmul_pack_rhs_uker_rhs_packed_dim_args rhs_pack_dims = {};
    rhs_pack_dims.n = N;
    rhs_pack_dims.k = K;

    const auto rhs_pack_stride = rhs_pack.get_rhs_packed_stride(nullptr, &rhs_pack_dims);
    return rhs_pack.get_rhs_packed_size(nullptr, &rhs_pack_dims, &rhs_pack_stride);
}

void PackRhsFp8(const uint8_t* src, uint8_t* dst, size_t N, size_t K) {
    const auto rhs_pack = kai_matmul_pack_rhs_kxn_x8p4vsx4_x8_sme();

    kai_matmul_pack_rhs_uker_rhs_dim_args rhs_dims = {};
    rhs_dims.n = N;
    rhs_dims.k = K;
    kai_matmul_pack_rhs_uker_rhs_packed_dim_args rhs_pack_dims = {};
    rhs_pack_dims.n = N;
    rhs_pack_dims.k = K;

    const auto rhs_stride = rhs_pack.get_rhs_stride(nullptr, &rhs_dims);
    const auto rhs_pack_stride = rhs_pack.get_rhs_packed_stride(nullptr, &rhs_pack_dims);

    kai_matmul_pack_rhs_uker_args rhs_args = {};
    rhs_args.flags = 0;
    rhs_args.shape.n = N;
    rhs_args.shape.k = K;
    rhs_args.operand.rhs.ptr = src;
    rhs_args.operand.rhs.stride = rhs_stride;
    rhs_args.operand.rhs_packed.ptr = dst;
    rhs_args.operand.rhs_packed.stride = rhs_pack_stride;

    rhs_pack.run(nullptr, &rhs_args);
}

}  // namespace

size_t
MLASCALL
ArmKleidiAI::MlasFp8GemmPackBSize(
    size_t N,
    size_t K,
    size_t BlockSizeK,
    size_t BlockSizeN,
    mlas_fp8_mode Fp8Type
    )
{
    if (!IsFp8RhsPackSupported(N, K, BlockSizeK, BlockSizeN, Fp8Type)) {
        return 0;
    }

    return GetRhsPackedSize(N, K);
}

size_t
MLASCALL
ArmKleidiAI::MlasFp8GemmPackBScaleSize(
    size_t N,
    size_t K,
    size_t BlockSizeK,
    size_t BlockSizeN,
    mlas_fp8_mode Fp8Type
    )
{
    if (!IsFp8RhsPackSupported(N, K, BlockSizeK, BlockSizeN, Fp8Type)) {
        return 0;
    }

    size_t scale_count = 0;
    if (MlasMultiplyOverflowsSizeT(K / BlockSizeK, N / BlockSizeN, &scale_count) ||
        scale_count > std::numeric_limits<size_t>::max() / sizeof(float)) {
        return 0;
    }
    return scale_count * sizeof(float);
}

bool
MLASCALL
ArmKleidiAI::MlasFp8GemmPackB(
    size_t N,
    size_t K,
    const void* B,
    size_t ldb,
    const float* ScaleB,
    size_t BlockSizeK,
    size_t BlockSizeN,
    mlas_fp8_mode Fp8Type,
    void* PackedB,
    float* PackedScaleB
    )
{
    if (B == nullptr || PackedB == nullptr || PackedScaleB == nullptr || ldb != N ||
        !IsFp8RhsPackSupported(N, K, BlockSizeK, BlockSizeN, Fp8Type)) {
        return false;
    }

    const size_t blocks_k = K / BlockSizeK;
    const size_t blocks_n = N / BlockSizeN;

    size_t packed_size = 0;
    size_t scale_count = 0;
    if (MlasMultiplyOverflowsSizeT(K, N, &packed_size) ||
        MlasMultiplyOverflowsSizeT(blocks_k, blocks_n, &scale_count) ||
        scale_count > std::numeric_limits<size_t>::max() / sizeof(float)) {
        return false;
    }
    const size_t scale_bytes = scale_count * sizeof(float);

    if (ScaleB == nullptr) {
        std::vector<uint8_t> b_fp8(packed_size);
        kai_quant_f8_f32(
            static_cast<const float*>(B),
            K,
            N,
            ldb,
            b_fp8.data(),
            N,
            PackedScaleB,
            1,
            blocks_k,
            BlockSizeK,
            BlockSizeN,
            ToKaiFp8Mode(Fp8Type));
        PackRhsFp8(b_fp8.data(), static_cast<uint8_t*>(PackedB), N, K);
        return true;
    }

    PackRhsFp8(static_cast<const uint8_t*>(B), static_cast<uint8_t*>(PackedB), N, K);
    // Keep a backend-owned scale buffer alongside the packed RHS. ORT now stores B scales as [BlocksN, BlocksK],
    // which matches the SME2 RHS scale layout, so no transpose is needed here.
    std::memcpy(PackedScaleB, ScaleB, scale_bytes);
    return true;
}

bool
MLASCALL
ArmKleidiAI::MlasFp8GemmPackedBIsSupported(
    const MLAS_FP8_GEMM_SHAPE_PARAMS& Shape,
    const MLAS_FP8_GEMM_DATA_PARAMS* DataParams,
    const size_t BatchSize
    )
{
    return IsFp8GemmSupported(Shape, DataParams, BatchSize);
}

bool
MLASCALL
ArmKleidiAI::MlasFp8GemmBatch(
    const MLAS_FP8_GEMM_SHAPE_PARAMS& Shape,
    const MLAS_FP8_GEMM_DATA_PARAMS* DataParams,
    const size_t BatchSize,
    MLAS_THREADPOOL* ThreadPool
    )
{
    if (!IsFp8GemmSupported(Shape, DataParams, BatchSize)) {
        return false;
    }

    const size_t M = Shape.M;
    const size_t N = Shape.N;
    const size_t K = Shape.K;
    const auto ukernel =
        kai_matmul_clamp_f32_qsf8dblp4vsx4_qsf8c2blp4vsx4_f32_f32_4vsx16vs_sme2_mopa();
    const auto lhs_pack = kai_matmul_pack_lhs_qsf8d32p4vsx4sf32_f32_sme();

    kai_matmul_pack_lhs_uker_config lhs_pack_config = {};
    lhs_pack_config.format.bl = DataParams[0].BlockSizeK;
    lhs_pack_config.format.f8_mode = ToKaiFp8Mode(DataParams[0].Fp8Type);

    kai_matmul_pack_lhs_uker_lhs_dim_args lhs_pack_lhs_shape = {};
    lhs_pack_lhs_shape.m = M;
    lhs_pack_lhs_shape.k = K;
    kai_matmul_pack_lhs_uker_lhs_packed_dim_args lhs_pack_lhs_packed_shape = {};
    lhs_pack_lhs_packed_shape.m = M;
    lhs_pack_lhs_packed_shape.k = K;

    const auto lhs_pack_lhs_stride = lhs_pack.get_lhs_stride(&lhs_pack_config, &lhs_pack_lhs_shape);
    const auto lhs_pack_lhs_packed_stride =
        lhs_pack.get_lhs_packed_stride(&lhs_pack_config, &lhs_pack_lhs_packed_shape);
    const size_t lhs_packed_size =
        lhs_pack.get_lhs_packed_size(&lhs_pack_config, &lhs_pack_lhs_packed_shape, &lhs_pack_lhs_packed_stride);

    size_t lhs_buffer_size = 0;
    if (MlasMultiplyOverflowsSizeT(lhs_packed_size, BatchSize, &lhs_buffer_size)) {
        return false;
    }
    std::vector<uint8_t> lhs_packed(lhs_buffer_size);

    MlasTrySimpleParallel(ThreadPool, static_cast<ptrdiff_t>(BatchSize), [&](ptrdiff_t batch_idx) {
        const auto& params = DataParams[static_cast<size_t>(batch_idx)];
        kai_matmul_pack_lhs_uker_args lhs_pack_args = {};
        lhs_pack_args.flags = 0;
        lhs_pack_args.shape.m = M;
        lhs_pack_args.shape.k = K;
        lhs_pack_args.operand.lhs.ptr = params.AFloat;
        lhs_pack_args.operand.lhs.stride = lhs_pack_lhs_stride;
        lhs_pack_args.operand.lhs_packed.ptr = lhs_packed.data() + lhs_packed_size * static_cast<size_t>(batch_idx);
        lhs_pack_args.operand.lhs_packed.stride = lhs_pack_lhs_packed_stride;

        lhs_pack.run(&lhs_pack_config, &lhs_pack_args);
    });

    kai_matmul_uker_config config = {};
    config.format.bl = DataParams[0].BlockSizeK;
    config.format.f8_mode = ToKaiFp8Mode(DataParams[0].Fp8Type);

    const auto step = ukernel.get_step(&config);
    size_t m_step = step.m;
    size_t n_step = step.n;
    if (m_step == 0 || n_step == 0) {
        return false;
    }

    std::array<size_t, 3> dim;
    dim[0] = BatchSize;
    dim[1] = MlasDivRoundup(M, m_step);
    dim[2] = MlasDivRoundup(N, n_step);

    const size_t thread_count = static_cast<size_t>(MlasGetMaximumThreadCount(ThreadPool));
    const size_t max_tiles = dim[0] * dim[1] * dim[2];
    const size_t required_tiles = std::min(thread_count, max_tiles);
    if (required_tiles == 0) {
        return false;
    }

    dim[1] = MlasDivRoundup(required_tiles * dim[1], dim[1] * dim[2]);
    dim[2] = MlasDivRoundup(required_tiles * dim[2], dim[1] * dim[2]);

    m_step *= MlasDivRoundup(MlasDivRoundup(M, dim[1]), m_step);
    n_step *= MlasDivRoundup(MlasDivRoundup(N, dim[2]), n_step);

    dim[1] = MlasDivRoundup(M, m_step);
    dim[2] = MlasDivRoundup(N, n_step);

    kai_matmul_uker_lhs_dim_args lhs_shape = {};
    lhs_shape.m = M;
    lhs_shape.k = K;
    const auto lhs_stride = ukernel.get_lhs_stride(&config, &lhs_shape);
    kai_matmul_uker_rhs_dim_args rhs_shape = {};
    rhs_shape.n = N;
    rhs_shape.k = K;
    const auto rhs_stride = ukernel.get_rhs_stride(&config, &rhs_shape);

    const size_t work_items = dim[0] * dim[1] * dim[2];
    if (work_items > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max())) {
        return false;
    }

    KLEIDIAI_KERNEL_LOG(
        "kai_matmul_clamp_f32_qsf8dblp4vsx4_qsf8c2blp4vsx4_f32_f32_4vsx16vs_sme2_mopa"
        << " M=" << M << " N=" << N << " K=" << K << " BatchSize=" << BatchSize);

    MlasTrySimpleParallel(ThreadPool, static_cast<ptrdiff_t>(work_items), [&](ptrdiff_t tid) {
        const size_t batch = static_cast<size_t>(tid) / (dim[1] * dim[2]);
        const size_t m_idx = (static_cast<size_t>(tid) % (dim[1] * dim[2])) / dim[2];
        const size_t n_idx = static_cast<size_t>(tid) % dim[2];
        const auto& params = DataParams[batch];
        const size_t blocks_k = params.BlocksK;
        const size_t m_start = m_idx * m_step;
        const size_t n_start = n_idx * n_step;
        const size_t tile_m = std::min(m_step, M - m_start);
        const size_t tile_n = std::min(n_step, N - n_start);

        kai_matmul_uker_lhs_dim_args lhs_index = {};
        lhs_index.m = m_start;
        lhs_index.k = 0;
        kai_matmul_uker_rhs_dim_args rhs_index = {};
        rhs_index.n = n_start;
        rhs_index.k = 0;

        auto* c = static_cast<float*>(params.C);
        kai_matmul_uker_args args = {};
        args.shape.m = tile_m;
        args.shape.n = tile_n;
        args.shape.k = K;
        args.operand.lhs.ptr = lhs_packed.data() + lhs_packed_size * batch +
                               ukernel.get_lhs_offset(&config, &lhs_index, &lhs_stride);
        args.operand.lhs.stride = lhs_stride;
        args.operand.rhs.ptr = static_cast<const uint8_t*>(params.PackedB) +
                               ukernel.get_rhs_offset(&config, &rhs_index, &rhs_stride);
        args.operand.rhs.stride = rhs_stride;
        args.operand.dst.ptr = c + m_start * params.ldc + n_start;
        args.operand.dst.stride.m = params.ldc * sizeof(float);
        args.operand.lhs_scale.ptr = nullptr;
        args.operand.lhs_scale.stride.m = 0;
        args.operand.rhs_scale.ptr = params.PackedScaleB + (n_start / params.BlockSizeN) * blocks_k;
        args.operand.rhs_scale.stride.n = blocks_k * sizeof(float);
        args.operand.rhs_bias.ptr = nullptr;
        args.operand.rhs_bias.stride.n = sizeof(float);

        ukernel.run(&config, &args);
    });

    MlasTrySimpleParallel(ThreadPool, static_cast<ptrdiff_t>(BatchSize), [&](ptrdiff_t batch_idx) {
        const auto& params = DataParams[static_cast<size_t>(batch_idx)];
        if (params.ScaleY != nullptr && params.ScaleY[0] != 1.0f) {
            auto* c = static_cast<float*>(params.C);
            for (size_t m = 0; m < M; ++m) {
                for (size_t n = 0; n < N; ++n) {
                    c[m * params.ldc + n] *= params.ScaleY[0];
                }
            }
        }
    });

    return true;
}

#endif  // !defined(DISABLE_FLOAT8_TYPES) && defined(MLAS_USE_KLEIDIAI_FP8)
