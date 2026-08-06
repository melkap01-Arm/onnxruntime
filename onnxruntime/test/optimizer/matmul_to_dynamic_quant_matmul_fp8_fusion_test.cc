// Copyright (c) 2026 Arm Limited. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#include "core/graph/constants.h"
#include "core/optimizer/matmul_to_dynamic_quant_matmul_fp8_fusion.h"

#include "test/optimizer/graph_transform_test_fixture.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/asserts.h"

#include "gtest/gtest.h"

#if !defined(DISABLE_CONTRIB_OPS) && !defined(DISABLE_FLOAT8_TYPES)

namespace onnxruntime::test {

class MatMulToDynamicQuantMatMulFp8FusionTest : public GraphTransformationTests {};

TEST_F(MatMulToDynamicQuantMatMulFp8FusionTest, RewritesEligibleCpuMatMul) {
  auto build = [](ModelTestBuilder& builder) {
    constexpr int64_t kM = 16;
    constexpr int64_t kK = 128;
    constexpr int64_t kN = 128;
    auto* activation = builder.MakeInput<float>({kM, kK}, -1.0f, 1.0f);
    auto* weight = builder.MakeInitializer<float>({kK, kN}, -1.0f, 1.0f);
    auto* output = builder.MakeOutput();
    builder.AddNode("MatMul", {activation, weight}, {output})
        .SetExecutionProviderType(kCpuExecutionProvider);
  };

  auto pre_check = [](Graph& graph) -> Status {
    EXPECT_EQ(CountOpsInGraph(graph)["MatMul"], 1);
    return Status::OK();
  };

  auto post_check = [](Graph& graph) -> Status {
    const auto op_counts = CountOpsInGraph(graph);
    EXPECT_EQ(op_counts.count("MatMul"), 0);
    EXPECT_EQ(op_counts.at("com.microsoft.DynamicQuantMatMulFp8"), 1);

    for (const auto& node : graph.Nodes()) {
      if (node.OpType() == "DynamicQuantMatMulFp8") {
        EXPECT_EQ(node.Domain(), kMSDomain);
        EXPECT_EQ(node.GetExecutionProviderType(), kCpuExecutionProvider);
        EXPECT_EQ(node.InputDefs().size(), 2u);
      }
    }
    return Status::OK();
  };

  auto transformer = std::make_unique<MatMulToDynamicQuantMatMulFp8Fusion>(
      InlinedHashSet<std::string_view>{kCpuExecutionProvider});
  ASSERT_STATUS_OK(TestGraphTransformer(build, 21, *logger_, std::move(transformer),
                                        TransformerLevel::Level2, 1, pre_check, post_check));
}

}  // namespace onnxruntime::test

#endif
