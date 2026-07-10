//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: MIT
//

#if defined(USE_KLEIDIAI) && !defined(_MSC_VER)

#include "test_util.h"
#include "core/mlas/lib/kleidiai/mlasi_kleidiai.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace {

void ConvertNchwToNhwc(const float* src,
                       float* dst,
                       size_t batches,
                       size_t channels,
                       size_t height,
                       size_t width) {
  const size_t src_stride_n = channels * height * width;
  const size_t src_stride_c = height * width;
  const size_t src_stride_h = width;
  const size_t dst_stride_n = height * width * channels;
  const size_t dst_stride_h = width * channels;
  const size_t dst_stride_w = channels;

  for (size_t n = 0; n < batches; ++n) {
    for (size_t h = 0; h < height; ++h) {
      for (size_t w = 0; w < width; ++w) {
        for (size_t c = 0; c < channels; ++c) {
          const size_t src_index = n * src_stride_n + c * src_stride_c + h * src_stride_h + w;
          const size_t dst_index = n * dst_stride_n + h * dst_stride_h + w * dst_stride_w + c;
          dst[dst_index] = src[src_index];
        }
      }
    }
  }
}

void DepthwiseReferenceNchw(const float* input,
                            const float* weights,
                            const float* bias,
                            size_t batches,
                            size_t channels,
                            size_t in_height,
                            size_t in_width,
                            size_t filter_height,
                            size_t filter_width,
                            size_t pad_top,
                            size_t pad_left,
                            size_t pad_bottom,
                            size_t pad_right,
                            float clamp_min,
                            float clamp_max,
                            float* output) {
  const size_t out_height = in_height + pad_top + pad_bottom + 1 - filter_height;
  const size_t out_width = in_width + pad_left + pad_right + 1 - filter_width;

  for (size_t b = 0; b < batches; ++b) {
    for (size_t c = 0; c < channels; ++c) {
      for (size_t oh = 0; oh < out_height; ++oh) {
        for (size_t ow = 0; ow < out_width; ++ow) {
          float acc = bias != nullptr ? bias[c] : 0.0f;
          for (size_t kh = 0; kh < filter_height; ++kh) {
            const int in_y = static_cast<int>(oh) + static_cast<int>(kh) - static_cast<int>(pad_top);
            if (in_y < 0 || in_y >= static_cast<int>(in_height)) {
              continue;
            }
            for (size_t kw = 0; kw < filter_width; ++kw) {
              const int in_x = static_cast<int>(ow) + static_cast<int>(kw) - static_cast<int>(pad_left);
              if (in_x < 0 || in_x >= static_cast<int>(in_width)) {
                continue;
              }
              const size_t input_idx =
                  (((b * channels) + c) * in_height + static_cast<size_t>(in_y)) * in_width +
                  static_cast<size_t>(in_x);
              const size_t weight_idx = (c * filter_height + kh) * filter_width + kw;
              acc += input[input_idx] * weights[weight_idx];
            }
          }
          const size_t output_idx = (((b * channels) + c) * out_height + oh) * out_width + ow;
          output[output_idx] = std::clamp(acc, clamp_min, clamp_max);
        }
      }
    }
  }
}

void RunDepthwiseConvCase(size_t channels,
                          size_t in_height,
                          size_t in_width,
                          size_t padding,
                          bool channels_last,
                          bool use_threadpool = false) {
  if (!MLAS_CPUIDINFO::GetCPUIDInfo().HasArm_SME2()) {
    GTEST_SKIP() << "DepthwiseConvKleidiAI requires ARM64 SME2. Skipping test.";
  }

  constexpr size_t batches = 1;
  constexpr size_t filter_height = 3;
  constexpr size_t filter_width = 3;

  const size_t pad_top = padding;
  const size_t pad_left = padding;
  const size_t pad_bottom = padding;
  const size_t pad_right = padding;

  const size_t input_size = batches * channels * in_height * in_width;
  const size_t weights_size = channels * filter_height * filter_width;
  const size_t out_height = in_height + pad_top + pad_bottom + 1 - filter_height;
  const size_t out_width = in_width + pad_left + pad_right + 1 - filter_width;
  const size_t output_size = batches * channels * out_height * out_width;

  std::vector<float> input(input_size);
  std::vector<float> weights(weights_size);
  std::vector<float> bias(channels);
  std::vector<float> input_nhwc(input_size);
  std::vector<float> expected(output_size);
  std::vector<float> expected_nhwc(output_size);
  std::vector<float> output(output_size, std::numeric_limits<float>::quiet_NaN());

  std::mt19937 rng(static_cast<uint32_t>(channels * 131 + in_height * 17 + padding));
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  auto fill_buffer = [&](std::vector<float>& buffer) {
    for (float& v : buffer) {
      v = dist(rng);
    }
  };

  fill_buffer(input);
  fill_buffer(weights);
  fill_buffer(bias);

  const float clamp_min = -std::numeric_limits<float>::max();
  const float clamp_max = std::numeric_limits<float>::max();

  DepthwiseReferenceNchw(input.data(),
                         weights.data(),
                         bias.data(),
                         batches,
                         channels,
                         in_height,
                         in_width,
                         filter_height,
                         filter_width,
                         pad_top,
                         pad_left,
                         pad_bottom,
                         pad_right,
                         clamp_min,
                         clamp_max,
                         expected.data());

  ConvertNchwToNhwc(input.data(), input_nhwc.data(), batches, channels, in_height, in_width);
  ConvertNchwToNhwc(expected.data(), expected_nhwc.data(), batches, channels, out_height, out_width);

  const float* input_data = channels_last ? input_nhwc.data() : input.data();
  const float* expected_data = channels_last ? expected_nhwc.data() : expected.data();

  const bool status = ArmKleidiAI::DepthwiseConvKleidiAI(batches,
                                                         in_height,
                                                         in_width,
                                                         channels,
                                                         filter_height,
                                                         filter_width,
                                                         1,
                                                         1,
                                                         1,
                                                         1,
                                                         pad_top,
                                                         pad_left,
                                                         pad_bottom,
                                                         pad_right,
                                                         channels_last,
                                                         input_data,
                                                         weights.data(),
                                                         bias.data(),
                                                         nullptr,
                                                         output.data(),
                                                         clamp_min,
                                                         clamp_max,
                                                         use_threadpool ? GetMlasThreadPool() : nullptr);
  ASSERT_TRUE(status);

  for (size_t i = 0; i < output_size; ++i) {
    EXPECT_NEAR(expected_data[i], output[i], 1e-4f) << "Mismatch at element " << i;
  }
}

void RunMlasConvCase(size_t channels,
                     size_t in_height,
                     size_t in_width,
                     size_t padding,
                     size_t batches,
                     bool channels_last,
                     bool use_kleidiai,
                     bool prepacked) {
  if (!MLAS_CPUIDINFO::GetCPUIDInfo().HasArm_SME2()) {
    GTEST_SKIP() << "DepthwiseConvKleidiAI requires ARM64 SME2. Skipping test.";
  }

  constexpr size_t filter_height = 3;
  constexpr size_t filter_width = 3;
  constexpr size_t filter_count = 1;

  const size_t out_height = in_height + padding + padding + 1 - filter_height;
  const size_t out_width = in_width + padding + padding + 1 - filter_width;
  const size_t input_size = batches * channels * in_height * in_width;
  const size_t output_size = batches * channels * out_height * out_width;

  std::vector<float> input(input_size);
  std::vector<float> input_nhwc(input_size);
  std::vector<float> weights(channels * filter_height * filter_width);
  std::vector<float> bias(channels);
  std::vector<float> expected(output_size);
  std::vector<float> expected_nhwc(output_size);
  std::vector<float> output(output_size, std::numeric_limits<float>::quiet_NaN());

  std::mt19937 rng(static_cast<uint32_t>(channels * 197 + in_width * 23 + padding));
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (float& v : input) {
    v = dist(rng);
  }
  for (float& v : weights) {
    v = dist(rng);
  }
  for (float& v : bias) {
    v = dist(rng);
  }

  DepthwiseReferenceNchw(input.data(),
                         weights.data(),
                         bias.data(),
                         batches,
                         channels,
                         in_height,
                         in_width,
                         filter_height,
                         filter_width,
                         padding,
                         padding,
                         padding,
                         padding,
                         -std::numeric_limits<float>::max(),
                         std::numeric_limits<float>::max(),
                         expected.data());
  ConvertNchwToNhwc(input.data(), input_nhwc.data(), batches, channels, in_height, in_width);
  ConvertNchwToNhwc(expected.data(), expected_nhwc.data(), batches, channels, out_height, out_width);
  const float* input_data = channels_last ? input_nhwc.data() : input.data();
  const float* expected_data = channels_last ? expected_nhwc.data() : expected.data();

  const int64_t input_shape[] = {static_cast<int64_t>(in_height), static_cast<int64_t>(in_width)};
  const int64_t kernel_shape[] = {static_cast<int64_t>(filter_height), static_cast<int64_t>(filter_width)};
  const int64_t dilation_shape[] = {1, 1};
  const int64_t pads[] = {static_cast<int64_t>(padding), static_cast<int64_t>(padding),
                          static_cast<int64_t>(padding), static_cast<int64_t>(padding)};
  const int64_t strides[] = {1, 1};
  const int64_t output_shape[] = {static_cast<int64_t>(out_height), static_cast<int64_t>(out_width)};
  MLAS_ACTIVATION activation;
  activation.ActivationKind = MlasIdentityActivation;

  MLAS_CONV_PARAMETERS parameters{};
  MLAS_BACKEND_KERNEL_SELECTOR_CONFIG selector_config{};
  selector_config.use_kleidiai = use_kleidiai;
  parameters.BackendKernelSelectorConfig = &selector_config;
  size_t working_buffer_size = 0;
  MlasConvPrepare(&parameters,
                  2,
                  batches,
                  channels,
                  1,
                  input_shape,
                  kernel_shape,
                  dilation_shape,
                  pads,
                  strides,
                  output_shape,
                  filter_count,
                  &activation,
                  &working_buffer_size,
                  channels_last,
                  0.0f,
                  nullptr);

  if (batches != 1 || !use_kleidiai) {
    EXPECT_EQ(ArmKleidiAI::SelectConvRoute(&parameters).route, ArmKleidiAI::ConvRoute::NoKleidiAi);
  }

  const float* filter_data = weights.data();
  std::vector<std::byte> packed_filter;
  if (prepacked) {
    packed_filter.resize(ArmKleidiAI::MlasConvDepthwiseFloatPackWSize(channels));
    ArmKleidiAI::MlasConvDepthwiseFloatPackW(
        channels, weights.data(), bias.data(), packed_filter.data());
    parameters.FilterIsPacked = true;
    parameters.PackedFilter = packed_filter.data();
    parameters.PackedFilterGroupStride = packed_filter.size();
    filter_data = nullptr;
  }

  std::vector<float> working_buffer(working_buffer_size);
  MlasConv(&parameters,
           input_data,
           filter_data,
           bias.data(),
           working_buffer.data(),
           output.data(),
           nullptr);

  for (size_t i = 0; i < output_size; ++i) {
    EXPECT_NEAR(expected_data[i], output[i], 1e-4f) << "Mismatch at element " << i;
  }
}

}  // namespace

TEST(MlasKleidiDepthwiseTest, ZeroPaddingNchw) {
  RunDepthwiseConvCase(/*channels=*/32, /*in_height=*/8, /*in_width=*/8, /*padding=*/0, /*channels_last=*/false);
}

TEST(MlasKleidiDepthwiseTest, UnitPaddingNchw) {
  RunDepthwiseConvCase(/*channels=*/32, /*in_height=*/8, /*in_width=*/8, /*padding=*/1, /*channels_last=*/false);
}

TEST(MlasKleidiDepthwiseTest, ZeroPaddingNhwc) {
  RunDepthwiseConvCase(/*channels=*/32, /*in_height=*/8, /*in_width=*/8, /*padding=*/0, /*channels_last=*/true);
}

TEST(MlasKleidiDepthwiseTest, UnitPaddingNhwc) {
  RunDepthwiseConvCase(/*channels=*/32, /*in_height=*/8, /*in_width=*/8, /*padding=*/1, /*channels_last=*/true);
}

TEST(MlasKleidiDepthwiseTest, MlasConvChannelsLast) {
  RunMlasConvCase(/*channels=*/32, /*in_height=*/8, /*in_width=*/8, /*padding=*/1,
                  /*batches=*/1, /*channels_last=*/true, /*use_kleidiai=*/true, /*prepacked=*/false);
}

TEST(MlasKleidiDepthwiseTest, LargeChannelCount) {
  RunDepthwiseConvCase(/*channels=*/128, /*in_height=*/8, /*in_width=*/8, /*padding=*/1,
                       /*channels_last=*/true);
}

TEST(MlasKleidiDepthwiseTest, NonVectorAlignedChannelCount) {
  RunDepthwiseConvCase(/*channels=*/130, /*in_height=*/8, /*in_width=*/8, /*padding=*/1,
                       /*channels_last=*/true);
}

TEST(MlasKleidiDepthwiseTest, ThreadedRows) {
  RunDepthwiseConvCase(/*channels=*/130, /*in_height=*/32, /*in_width=*/8, /*padding=*/1,
                       /*channels_last=*/true, /*use_threadpool=*/true);
}

TEST(MlasKleidiDepthwiseTest, MlasConvChannelsLastPrepacked) {
  RunMlasConvCase(/*channels=*/32, /*in_height=*/8, /*in_width=*/8, /*padding=*/1,
                  /*batches=*/1, /*channels_last=*/true, /*use_kleidiai=*/true, /*prepacked=*/true);
}

TEST(MlasKleidiDepthwiseTest, BatchFallback) {
  RunMlasConvCase(/*channels=*/32, /*in_height=*/8, /*in_width=*/8, /*padding=*/1,
                  /*batches=*/2, /*channels_last=*/false, /*use_kleidiai=*/true, /*prepacked=*/false);
}

TEST(MlasKleidiDepthwiseTest, SelectorOptOutFallback) {
  RunMlasConvCase(/*channels=*/32, /*in_height=*/8, /*in_width=*/8, /*padding=*/1,
                  /*batches=*/1, /*channels_last=*/false, /*use_kleidiai=*/false, /*prepacked=*/false);
}

TEST(MlasKleidiDepthwiseTest, RejectsOutputSmallerThanTile) {
  if (!MLAS_CPUIDINFO::GetCPUIDInfo().HasArm_SME2()) {
    GTEST_SKIP() << "DepthwiseConvKleidiAI requires ARM64 SME2. Skipping test.";
  }

  MLAS_CONV_PARAMETERS parameters{};
  parameters.Dimensions = 2;
  parameters.BatchCount = 1;
  parameters.GroupCount = 32;
  parameters.InputChannels = 1;
  parameters.FilterCount = 1;
  parameters.InputShape[0] = 5;
  parameters.InputShape[1] = 8;
  parameters.KernelShape[0] = 3;
  parameters.KernelShape[1] = 3;
  parameters.DilationShape[0] = 1;
  parameters.DilationShape[1] = 1;
  parameters.StrideShape[0] = 1;
  parameters.StrideShape[1] = 1;

  EXPECT_FALSE(ArmKleidiAI::DepthwiseConvKleidiAISupported(&parameters));
}

TEST(MlasKleidiDepthwiseTest, StaticSupportHonorsSelectorOptOut) {
  if (!MLAS_CPUIDINFO::GetCPUIDInfo().HasArm_SME2()) {
    GTEST_SKIP() << "DepthwiseConvKleidiAI requires ARM64 SME2. Skipping test.";
  }

  const size_t input_shape[] = {8, 8};
  const size_t kernel_shape[] = {3, 3};
  const size_t dilation_shape[] = {1, 1};
  const size_t padding[] = {1, 1, 1, 1};
  const size_t stride_shape[] = {1, 1};
  MLAS_BACKEND_KERNEL_SELECTOR_CONFIG selector_config{};

  const auto is_supported = [&] {
    return MlasConvSupportsDepthwiseChannelsLast2DFloatKernel(
        /*Dimensions=*/2,
        /*BatchCount=*/1,
        /*GroupCount=*/32,
        /*InputChannelsPerGroup=*/1,
        input_shape,
        kernel_shape,
        dilation_shape,
        padding,
        stride_shape,
        /*FilterCount=*/1,
        /*Beta=*/0.0f,
        &selector_config);
  };

  ASSERT_TRUE(is_supported());
  selector_config.use_kleidiai = false;
  EXPECT_FALSE(is_supported());
}

TEST(MlasKleidiDepthwiseTest, RejectsUnsupportedDirectConfigurations) {
  if (!MLAS_CPUIDINFO::GetCPUIDInfo().HasArm_SME2()) {
    GTEST_SKIP() << "DepthwiseConvKleidiAI requires ARM64 SME2. Skipping test.";
  }

  constexpr size_t channels = 32;
  constexpr size_t in_height = 8;
  constexpr size_t in_width = 8;
  std::vector<float> input(channels * in_height * in_width);
  std::vector<float> weights(channels * 5 * 5);
  std::vector<float> output(channels * in_height * in_width);

  struct Configuration {
    size_t filter_height;
    size_t filter_width;
    size_t stride_height;
    size_t stride_width;
    size_t dilation_height;
    size_t dilation_width;
    size_t pad_top;
    size_t pad_left;
    size_t pad_bottom;
    size_t pad_right;
  };

  constexpr Configuration unsupported_configurations[] = {
      {5, 3, 1, 1, 1, 1, 0, 0, 0, 0},
      {3, 3, 2, 1, 1, 1, 0, 0, 0, 0},
      {3, 3, 1, 1, 2, 1, 0, 0, 0, 0},
      {3, 3, 1, 1, 1, 1, 1, 0, 1, 1},
      {3, 3, 1, 1, 1, 1, 2, 2, 2, 2},
  };

  for (const auto& config : unsupported_configurations) {
    EXPECT_FALSE(ArmKleidiAI::DepthwiseConvKleidiAI(
        1, in_height, in_width, channels,
        config.filter_height, config.filter_width,
        config.stride_height, config.stride_width,
        config.dilation_height, config.dilation_width,
        config.pad_top, config.pad_left, config.pad_bottom, config.pad_right,
        true, input.data(), weights.data(), nullptr, nullptr, output.data(),
        -std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
        nullptr));
  }
}

#endif  // defined(USE_KLEIDIAI) && !defined(_MSC_VER)
