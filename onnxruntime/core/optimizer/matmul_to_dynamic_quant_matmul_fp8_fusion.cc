// Copyright (c) 2026 Arm Limited. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#include "core/optimizer/matmul_to_dynamic_quant_matmul_fp8_fusion.h"

#include "core/graph/constants.h"
#include "core/graph/graph_utils.h"
#include "core/graph/graph_viewer.h"

namespace onnxruntime {
namespace {

constexpr int64_t kFp8BlockSize = 128;

bool IsFp32Tensor(const NodeArg* node_arg) {
  const auto* type = node_arg->Type();
  return type != nullptr && *type == "tensor(float)";
}

bool HasSupportedActivationShape(const NodeArg& activation, int64_t weight_k) {
  const auto* shape = activation.Shape();
  if (shape == nullptr || shape->dim_size() == 0) {
    return true;
  }

  const auto& activation_k = shape->dim(shape->dim_size() - 1);
  return !activation_k.has_dim_value() || activation_k.dim_value() == weight_k;
}

bool IsSupportedMatMul(const Graph& graph, const Node& node,
                       const InlinedHashSet<std::string_view>& compatible_execution_providers) {
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "MatMul", {1, 9, 13}, kOnnxDomain) ||
      !graph_utils::IsSupportedProvider(node, compatible_execution_providers) ||
      node.InputDefs().size() != 2 || node.OutputDefs().size() != 1 ||
      !IsFp32Tensor(node.InputDefs()[0]) || !IsFp32Tensor(node.InputDefs()[1]) ||
      !IsFp32Tensor(node.OutputDefs()[0])) {
    return false;
  }

  const auto* weight = graph.GetConstantInitializer(node.InputDefs()[1]->Name(), true);
  if (weight == nullptr || weight->data_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT ||
      weight->dims_size() != 2) {
    return false;
  }

  const int64_t weight_k = weight->dims(0);
  const int64_t weight_n = weight->dims(1);
  return weight_k > 0 && weight_n > 0 &&
         weight_k % kFp8BlockSize == 0 && weight_n % kFp8BlockSize == 0 &&
         HasSupportedActivationShape(*node.InputDefs()[0], weight_k);
}

}  // namespace

Status MatMulToDynamicQuantMatMulFp8Fusion::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                                      const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  size_t transformed_node_count = 0;

  for (const auto node_index : node_topology_list) {
    auto* node = graph.GetNode(node_index);
    if (node == nullptr) {
      continue;
    }

    ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level, logger));

    if (!IsSupportedMatMul(graph, *node, GetCompatibleExecutionProviders())) {
      continue;
    }

    Node& fp8_node = graph.AddNode(
        graph.GenerateNodeName(node->Name() + "_DynamicQuantMatMulFp8"),
        "DynamicQuantMatMulFp8",
        node->Description(),
        node->MutableInputDefs(),
        node->MutableOutputDefs(),
        nullptr,
        kMSDomain);
    fp8_node.SetExecutionProviderType(node->GetExecutionProviderType());

    graph_utils::RemoveNodeOutputEdges(graph, *node);
    graph.RemoveNode(node->Index());
    ++transformed_node_count;
    modified = true;
  }

  LOGS(logger, INFO) << "MatMulToDynamicQuantMatMulFp8Fusion transformed node count: "
                     << transformed_node_count;
  return Status::OK();
}

}  // namespace onnxruntime
