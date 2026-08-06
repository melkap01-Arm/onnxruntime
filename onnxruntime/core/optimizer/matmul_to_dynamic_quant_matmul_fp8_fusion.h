// Copyright (c) 2026 Arm Limited. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

class MatMulToDynamicQuantMatMulFp8Fusion final : public GraphTransformer {
 public:
  explicit MatMulToDynamicQuantMatMulFp8Fusion(
      const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("MatMulToDynamicQuantMatMulFp8Fusion", compatible_execution_providers) {}

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level,
                   const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
