// Copyright (c) 2026 Arm Limited. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#include "mlas.h"
#include "test_util.h"
#include "core/common/float8.h"
#include "core/mlas/inc/mlas.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#if !defined(DISABLE_FLOAT8_TYPES)

namespace {

uint8_t EncodeFp8(float value, mlas_fp8_mode mode) {
  using onnxruntime::Float8E4M3FN;
  using onnxruntime::Float8E4M3FNUZ;
  using onnxruntime::Float8E5M2;
  using onnxruntime::Float8E5M2FNUZ;

  switch (mode) {
    case MLAS_FP8_MODE_E4M3_INF:
      return Float8E4M3FN(value).val;
    case MLAS_FP8_MODE_E4M3_SAT:
      return Float8E4M3FNUZ(value).val;
    case MLAS_FP8_MODE_E5M2_INF:
      return Float8E5M2(value).val;
    case MLAS_FP8_MODE_E5M2_SAT:
      return Float8E5M2FNUZ(value).val;
    default:
      ORT_THROW("Unsupported FP8 GEMM test mode.");
  }
}

const char* Fp8ModeName(mlas_fp8_mode mode) {
  switch (mode) {
    case MLAS_FP8_MODE_E4M3_INF:
      return "E4M3_INF";
    case MLAS_FP8_MODE_E4M3_SAT:
      return "E4M3_SAT";
    case MLAS_FP8_MODE_E5M2_INF:
      return "E5M2_INF";
    case MLAS_FP8_MODE_E5M2_SAT:
      return "E5M2_SAT";
    default:
      return "unknown";
  }
}

float Fp8MaxAbs(mlas_fp8_mode mode) {
  switch (mode) {
    case MLAS_FP8_MODE_E4M3_INF:
      return 448.0f;
    case MLAS_FP8_MODE_E4M3_SAT:
      return 240.0f;
    case MLAS_FP8_MODE_E5M2_INF:
    case MLAS_FP8_MODE_E5M2_SAT:
      return 57344.0f;
    default:
      ORT_THROW("Unsupported FP8 GEMM test mode.");
  }
}

void QuantizeBlockwiseFp8ForTest(const float* src,
                                 size_t rows,
                                 size_t cols,
                                 size_t src_row_stride,
                                 size_t block_size_rows,
                                 size_t block_size_cols,
                                 uint8_t* dst,
                                 size_t dst_row_stride,
                                 float* scales,
                                 size_t scales_block_row_stride,
                                 size_t scales_block_col_stride,
                                 mlas_fp8_mode mode) {
  const size_t block_rows = rows / block_size_rows;
  const size_t block_cols = cols / block_size_cols;
  const float fp8_max_abs = Fp8MaxAbs(mode);

  for (size_t br = 0; br < block_rows; ++br) {
    for (size_t bc = 0; bc < block_cols; ++bc) {
      const size_t row_start = br * block_size_rows;
      const size_t col_start = bc * block_size_cols;
      float max_abs = 0.0f;

      for (size_t r = 0; r < block_size_rows; ++r) {
        for (size_t c = 0; c < block_size_cols; ++c) {
          max_abs = std::max(max_abs, std::fabs(src[(row_start + r) * src_row_stride + col_start + c]));
        }
      }

      const float scale = max_abs == 0.0f ? 1.0f : max_abs / fp8_max_abs;
      scales[br * scales_block_row_stride + bc * scales_block_col_stride] = scale;

      for (size_t r = 0; r < block_size_rows; ++r) {
        for (size_t c = 0; c < block_size_cols; ++c) {
          const float value = max_abs == 0.0f ? 0.0f : src[(row_start + r) * src_row_stride + col_start + c] / scale;
          dst[(row_start + r) * dst_row_stride + col_start + c] = EncodeFp8(value, mode);
        }
      }
    }
  }
}

void RunFp8GemmBatchThreaded(mlas_fp8_mode mode) {
  constexpr size_t BatchN = 2;
  constexpr size_t M = 3;
  constexpr size_t N = 2;
  constexpr size_t K = 4;
  constexpr size_t BlockSizeM = 2;
  constexpr size_t BlockSizeK = 2;
  constexpr size_t BlockSizeN = 1;
  constexpr size_t BlocksM = 2;
  constexpr size_t BlocksK = 2;
  constexpr size_t BlocksN = 2;
  constexpr size_t ScaleElements = BlocksM * BlocksK;
  constexpr size_t BScaleElements = BlocksK * BlocksN;

  const std::array<float, BatchN * M * K> a_values{
      1.0f, 2.0f, -1.0f, 0.5f,
      -2.0f, 1.5f, 0.0f, 4.0f,
      0.25f, -0.5f, 3.0f, -4.0f,
      -1.0f, 0.5f, 2.0f, -2.0f,
      4.0f, -0.25f, -1.5f, 1.0f,
      0.0f, 3.0f, -4.0f, 0.5f};
  const std::array<float, BatchN * K * N> b_values{
      1.0f, -1.0f,
      0.5f, 2.0f,
      -2.0f, 0.25f,
      1.5f, -0.5f,
      -0.5f, 1.0f,
      2.0f, -2.0f,
      0.25f, 1.5f,
      -1.0f, 0.5f};
  const std::array<float, BatchN * ScaleElements> scale_a{
      1.0f, 0.5f,
      2.0f, 1.5f,
      0.25f, 1.0f,
      0.5f, 2.0f};
  const std::array<float, BatchN * BScaleElements> scale_b{
      1.0f, 2.0f,
      0.25f, 1.25f,
      0.5f, 1.0f,
      2.0f, 0.25f};
  const std::array<float, BatchN> y_scale{0.5f, 2.0f};

  std::vector<uint8_t> a_fp8(a_values.size());
  std::vector<uint8_t> b_fp8(b_values.size());
  for (size_t i = 0; i < a_values.size(); ++i) {
    a_fp8[i] = EncodeFp8(a_values[i], mode);
  }
  for (size_t i = 0; i < b_values.size(); ++i) {
    b_fp8[i] = EncodeFp8(b_values[i], mode);
  }

  std::array<float, BatchN * M * N> output{};
  std::array<float, BatchN * M * N> expected{};
  std::array<MLAS_FP8_GEMM_DATA_PARAMS, BatchN> params{};

  for (size_t batch = 0; batch < BatchN; ++batch) {
    params[batch].A = a_fp8.data() + batch * M * K;
    params[batch].lda = K;
    params[batch].B = b_fp8.data() + batch * K * N;
    params[batch].ldb = N;
    params[batch].C = output.data() + batch * M * N;
    params[batch].ldc = N;
    params[batch].ScaleA = scale_a.data() + batch * ScaleElements;
    params[batch].ScaleB = scale_b.data() + batch * BScaleElements;
    params[batch].ScaleY = y_scale.data() + batch;
    params[batch].Fp8Type = mode;
    params[batch].BlockSizeM = BlockSizeM;
    params[batch].BlockSizeK = BlockSizeK;
    params[batch].BlockSizeN = BlockSizeN;
    params[batch].BlocksM = BlocksM;
    params[batch].BlocksK = BlocksK;
    params[batch].BlocksN = BlocksN;
    params[batch].ScaleAStrideK = 1;
    params[batch].ScaleAStrideM = BlocksK;
    params[batch].ScaleBStrideN = 1;
    params[batch].ScaleBStrideK = BlocksN;

    for (size_t m = 0; m < M; ++m) {
      const size_t block_m = m / BlockSizeM;
      for (size_t n = 0; n < N; ++n) {
        const size_t block_n = n / BlockSizeN;
        float acc = 0.0f;
        for (size_t k = 0; k < K; ++k) {
          const size_t block_k = k / BlockSizeK;
          const size_t a_scale_idx = batch * ScaleElements + block_m * BlocksK + block_k;
          const size_t b_scale_idx = batch * BScaleElements + block_k * BlocksN + block_n;
          const float a_deq = a_values[batch * M * K + m * K + k] * scale_a[a_scale_idx];
          const float b_deq = b_values[batch * K * N + k * N + n] * scale_b[b_scale_idx];
          acc += a_deq * b_deq;
        }
        expected[batch * M * N + m * N + n] = acc * y_scale[batch];
      }
    }
  }

  MLAS_FP8_GEMM_SHAPE_PARAMS shape{M, N, K};
  MLAS_THREADPOOL* threadpool = GetMlasThreadPool();
  if (threadpool == nullptr) {
    GTEST_SKIP() << "MlasFp8GemmBatch threaded test requires an MLAS thread pool.";
  }

  MlasFp8GemmBatch(shape, params.data(), BatchN, threadpool);

  // Inputs are exactly representable test values, so the scalar fallback should match closely.
  for (size_t i = 0; i < output.size(); ++i) {
    EXPECT_NEAR(output[i], expected[i], 1e-5f);
  }
}

void RunKleidiaiPackedPathMatchesScalarFallback(mlas_fp8_mode mode, bool pack_prequantized_b) {
  SCOPED_TRACE(::testing::Message()
               << "mode=" << Fp8ModeName(mode)
               << ", pack_prequantized_b=" << pack_prequantized_b);

  constexpr size_t BatchN = 2;
  constexpr size_t M = 64;
  constexpr size_t N = 256;
  constexpr size_t K = 256;
  constexpr size_t BlockSizeM = 1;
  constexpr size_t BlockSizeK = 256;
  constexpr size_t BlockSizeN = 256;
  constexpr size_t BlocksM = M / BlockSizeM;
  constexpr size_t BlocksK = K / BlockSizeK;
  constexpr size_t BlocksN = N / BlockSizeN;
  constexpr size_t AScaleElements = BlocksM * BlocksK;
  constexpr size_t BScaleElements = BlocksN * BlocksK;

  const size_t packed_b_size = MlasFp8GemmPackBSize(N, K, BlockSizeK, BlockSizeN, mode);
  if (packed_b_size == 0) {
    GTEST_SKIP() << "KleidiAI FP8 packed-B path is unavailable for this target or shape.";
  }

  const size_t packed_b_scale_size = MlasFp8GemmPackBScaleSize(N, K, BlockSizeK, BlockSizeN, mode);
  ASSERT_NE(packed_b_scale_size, 0u);
  ASSERT_EQ(packed_b_scale_size % sizeof(float), 0u);
  const size_t packed_b_scale_count = packed_b_scale_size / sizeof(float);
  ASSERT_EQ(packed_b_scale_count, BScaleElements);

  MLAS_THREADPOOL* threadpool = GetMlasThreadPool();
  if (threadpool == nullptr) {
    GTEST_SKIP() << "KleidiAI FP8 threaded test requires an MLAS thread pool.";
  }

  std::vector<float> a_float(BatchN * M * K);
  std::vector<float> b_float(BatchN * K * N);
  for (size_t batch = 0; batch < BatchN; ++batch) {
    for (size_t i = 0; i < M * K; ++i) {
      const int pattern = static_cast<int>((i + batch * 5) % 17) - 8;
      a_float[batch * M * K + i] = static_cast<float>(pattern) * 0.03125f;
    }
    for (size_t i = 0; i < K * N; ++i) {
      const int pattern = static_cast<int>((i * 3 + batch * 7) % 19) - 9;
      b_float[batch * K * N + i] = static_cast<float>(pattern) * 0.015625f;
    }
  }

  std::vector<uint8_t> a_fp8(BatchN * M * K);
  std::vector<uint8_t> b_fp8(BatchN * K * N);
  std::vector<float> scale_a(BatchN * AScaleElements);
  std::vector<float> scale_b(BatchN * BScaleElements);
  std::vector<uint8_t> packed_b(BatchN * packed_b_size);
  std::vector<float> packed_b_scales(BatchN * packed_b_scale_count);
  const std::array<float, BatchN> y_scale{1.0f, 0.5f};

  for (size_t batch = 0; batch < BatchN; ++batch) {
    QuantizeBlockwiseFp8ForTest(a_float.data() + batch * M * K,
                                M,
                                K,
                                K,
                                BlockSizeM,
                                BlockSizeK,
                                a_fp8.data() + batch * M * K,
                                K,
                                scale_a.data() + batch * AScaleElements,
                                BlocksK,
                                1,
                                mode);
    QuantizeBlockwiseFp8ForTest(b_float.data() + batch * K * N,
                                K,
                                N,
                                N,
                                BlockSizeK,
                                BlockSizeN,
                                b_fp8.data() + batch * K * N,
                                N,
                                scale_b.data() + batch * BScaleElements,
                                1,
                                BlocksK,
                                mode);

    ASSERT_TRUE(MlasFp8GemmPackB(N,
                                 K,
                                 pack_prequantized_b ? static_cast<const void*>(b_fp8.data() + batch * K * N)
                                                     : static_cast<const void*>(b_float.data() + batch * K * N),
                                 N,
                                 pack_prequantized_b ? scale_b.data() + batch * BScaleElements : nullptr,
                                 BlockSizeK,
                                 BlockSizeN,
                                 mode,
                                 packed_b.data() + batch * packed_b_size,
                                 packed_b_scales.data() + batch * packed_b_scale_count));
  }

  std::vector<float> packed_output(BatchN * M * N);
  std::vector<float> scalar_output(BatchN * M * N);
  std::array<MLAS_FP8_GEMM_DATA_PARAMS, BatchN> packed_params{};
  std::array<MLAS_FP8_GEMM_DATA_PARAMS, BatchN> scalar_params{};

  for (size_t batch = 0; batch < BatchN; ++batch) {
    packed_params[batch].AFloat = a_float.data() + batch * M * K;
    packed_params[batch].lda = K;
    packed_params[batch].PackedB = packed_b.data() + batch * packed_b_size;
    packed_params[batch].PackedScaleB = packed_b_scales.data() + batch * packed_b_scale_count;
    packed_params[batch].C = packed_output.data() + batch * M * N;
    packed_params[batch].ldc = N;
    packed_params[batch].ScaleY = y_scale.data() + batch;
    packed_params[batch].Fp8Type = mode;
    packed_params[batch].BlockSizeM = BlockSizeM;
    packed_params[batch].BlockSizeK = BlockSizeK;
    packed_params[batch].BlockSizeN = BlockSizeN;
    packed_params[batch].BlocksM = BlocksM;
    packed_params[batch].BlocksK = BlocksK;
    packed_params[batch].BlocksN = BlocksN;

    scalar_params[batch].A = a_fp8.data() + batch * M * K;
    scalar_params[batch].lda = K;
    scalar_params[batch].B = b_fp8.data() + batch * K * N;
    scalar_params[batch].ldb = N;
    scalar_params[batch].C = scalar_output.data() + batch * M * N;
    scalar_params[batch].ldc = N;
    scalar_params[batch].ScaleA = scale_a.data() + batch * AScaleElements;
    scalar_params[batch].ScaleB = scale_b.data() + batch * BScaleElements;
    scalar_params[batch].ScaleY = y_scale.data() + batch;
    scalar_params[batch].Fp8Type = mode;
    scalar_params[batch].BlockSizeM = BlockSizeM;
    scalar_params[batch].BlockSizeK = BlockSizeK;
    scalar_params[batch].BlockSizeN = BlockSizeN;
    scalar_params[batch].BlocksM = BlocksM;
    scalar_params[batch].BlocksK = BlocksK;
    scalar_params[batch].BlocksN = BlocksN;
    scalar_params[batch].ScaleAStrideK = 1;
    scalar_params[batch].ScaleAStrideM = BlocksK;
    scalar_params[batch].ScaleBStrideK = 1;
    scalar_params[batch].ScaleBStrideN = BlocksK;
  }

  MLAS_FP8_GEMM_SHAPE_PARAMS shape{M, N, K};
  MlasFp8GemmBatch(shape, packed_params.data(), BatchN, threadpool);
  MlasFp8GemmBatch(shape, scalar_params.data(), BatchN, nullptr);

  for (size_t i = 0; i < packed_output.size(); ++i) {
    ASSERT_FALSE(std::isnan(packed_output[i])) << "packed output contains NaN at index " << i;
    EXPECT_NEAR(packed_output[i], scalar_output[i], 5e-1f);
  }
}

}  // namespace

TEST(Fp8Gemm, BatchedModesSymmetricThreaded) {
  RunFp8GemmBatchThreaded(MLAS_FP8_MODE_E4M3_INF);
  RunFp8GemmBatchThreaded(MLAS_FP8_MODE_E4M3_SAT);
  RunFp8GemmBatchThreaded(MLAS_FP8_MODE_E5M2_INF);
  RunFp8GemmBatchThreaded(MLAS_FP8_MODE_E5M2_SAT);
}

TEST(Fp8Gemm, KleidiaiPackedFloatBMatchesScalarFallbackThreadedE4M3Inf) {
  RunKleidiaiPackedPathMatchesScalarFallback(MLAS_FP8_MODE_E4M3_INF, false);
}

TEST(Fp8Gemm, KleidiaiPackedFloatBMatchesScalarFallbackThreadedE4M3Sat) {
  RunKleidiaiPackedPathMatchesScalarFallback(MLAS_FP8_MODE_E4M3_SAT, false);
}

TEST(Fp8Gemm, KleidiaiPackedFloatBMatchesScalarFallbackThreadedE5M2Inf) {
  RunKleidiaiPackedPathMatchesScalarFallback(MLAS_FP8_MODE_E5M2_INF, false);
}

TEST(Fp8Gemm, KleidiaiPackedFloatBMatchesScalarFallbackThreadedE5M2Sat) {
  RunKleidiaiPackedPathMatchesScalarFallback(MLAS_FP8_MODE_E5M2_SAT, false);
}

TEST(Fp8Gemm, KleidiaiPackedFp8BMatchesScalarFallbackThreadedE4M3Inf) {
  RunKleidiaiPackedPathMatchesScalarFallback(MLAS_FP8_MODE_E4M3_INF, true);
}

TEST(Fp8Gemm, KleidiaiPackedFp8BMatchesScalarFallbackThreadedE4M3Sat) {
  RunKleidiaiPackedPathMatchesScalarFallback(MLAS_FP8_MODE_E4M3_SAT, true);
}

TEST(Fp8Gemm, KleidiaiPackedFp8BMatchesScalarFallbackThreadedE5M2Inf) {
  RunKleidiaiPackedPathMatchesScalarFallback(MLAS_FP8_MODE_E5M2_INF, true);
}

TEST(Fp8Gemm, KleidiaiPackedFp8BMatchesScalarFallbackThreadedE5M2Sat) {
  RunKleidiaiPackedPathMatchesScalarFallback(MLAS_FP8_MODE_E5M2_SAT, true);
}

TEST(Fp8Gemm, KleidiaiPackedFp8BKeepsScaleLayout) {
  constexpr size_t N = 512;
  constexpr size_t K = 512;
  constexpr size_t BlockSizeK = 256;
  constexpr size_t BlockSizeN = 256;
  constexpr size_t BlocksK = K / BlockSizeK;
  constexpr size_t BlocksN = N / BlockSizeN;
  constexpr mlas_fp8_mode Mode = MLAS_FP8_MODE_E4M3_INF;

  const size_t packed_b_size = MlasFp8GemmPackBSize(N, K, BlockSizeK, BlockSizeN, Mode);
  if (packed_b_size == 0) {
    GTEST_SKIP() << "KleidiAI FP8 packed-B path is unavailable for this target or shape.";
  }

  const size_t packed_b_scale_size = MlasFp8GemmPackBScaleSize(N, K, BlockSizeK, BlockSizeN, Mode);
  ASSERT_EQ(packed_b_scale_size, BlocksK * BlocksN * sizeof(float));

  std::vector<uint8_t> b_fp8(K * N, EncodeFp8(0.0f, Mode));
  std::vector<uint8_t> packed_b(packed_b_size);
  std::array<float, BlocksK * BlocksN> scale_b{
      1.0f, 2.0f,
      3.0f, 4.0f};
  std::array<float, BlocksK * BlocksN> packed_scales{};

  ASSERT_TRUE(MlasFp8GemmPackB(N,
                               K,
                               b_fp8.data(),
                               N,
                               scale_b.data(),
                               BlockSizeK,
                               BlockSizeN,
                               Mode,
                               packed_b.data(),
                               packed_scales.data()));

  const std::array<float, BlocksK * BlocksN> expected_packed_scales{
      1.0f, 2.0f,
      3.0f, 4.0f};
  EXPECT_EQ(packed_scales, expected_packed_scales);
}

TEST(Fp8Gemm, EmptyDimensionsSkipUnusedBufferValidation) {
  MLAS_FP8_GEMM_DATA_PARAMS params{};
  params.Fp8Type = MLAS_FP8_MODE_E4M3_INF;
  params.BlockSizeM = 2;
  params.BlockSizeK = 2;
  params.BlockSizeN = 2;

  MLAS_FP8_GEMM_SHAPE_PARAMS empty_output_shape{3, 0, 4};
  MlasFp8GemmBatch(empty_output_shape, &params, 1, nullptr);

  std::array<float, 6> output;
  output.fill(-1.0f);

  MLAS_FP8_GEMM_SHAPE_PARAMS empty_reduction_shape{3, 2, 0};
  params.C = output.data();
  params.ldc = 2;
  MlasFp8GemmBatch(empty_reduction_shape, &params, 1, nullptr);

  for (float value : output) {
    EXPECT_EQ(value, 0.0f);
  }
}

TEST(Fp8Gemm, ZeroColumnReturnsBeforeWorkItemOverflow) {
  MLAS_FP8_GEMM_SHAPE_PARAMS shape{std::numeric_limits<size_t>::max(), 0, 4};
  MlasFp8GemmBatch(shape, nullptr, 2, nullptr);
  SUCCEED();
}

#endif  // !defined(DISABLE_FLOAT8_TYPES)
