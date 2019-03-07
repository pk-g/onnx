// Adapter for Flatten in default domain from version 9 to 8

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

struct Flatten_9_8 final : public Adapter {
  explicit Flatten_9_8()
    : Adapter("Flatten", OpSetID(9), OpSetID(8)) {
    }

  void adapt_flatten_9_8(std::shared_ptr<Graph> graph, Node* node) const {
    std::cout << TensorProto::DataType_Name((TensorProto_DataType)node->inputs()[0]->elemType()) << std::endl;
    
    const ArrayRef<Value*>& inputs = node->inputs();
    const ArrayRef<Value*>& outputs = node->outputs();
    std::cout << graph->outputs()[0]->uniqueName() << std::endl;

    
    Node *precast = graph->create(kCast, inputs[0]);
    precast->i_(kto, TensorProto_DataType::TensorProto_DataType_FLOAT);
    precast->insertBefore(node);
    precast->output()->setUniqueName("precast_output");
    precast->output()->setElemType(TensorProto_DataType::TensorProto_DataType_FLOAT);
    precast->output()->setSizes(inputs[0]->sizes());
    
    node->replaceInput(0, precast->output());
    node->output()->setElemType(TensorProto_DataType_FLOAT);
    node->output()->setUniqueName("flatten_output");
    
    Node *postcast = graph->create(kCast, node->outputs()[0]);
    postcast->i_(kto, TensorProto_DataType::TensorProto_DataType_UINT8);
    postcast->insertAfter(node);
    postcast->output()->setUniqueName("postcast_output");
    postcast->output()->setElemType(TensorProto_DataType_UINT8);
    postcast->output()->setSizes(outputs[0]->sizes());

    graph->registerOutput(postcast->output());
    std::cout << "GRAPH OUTPUTS SIZE: " << graph->outputs().size() << std::endl;
    std::cout << "return_node OUTPUTS SIZE: " << graph->return_node()->inputs().size() << std::endl;
    graph->return_node()->removeInput(0);
    //graph->outputs().vec().push_back(postcast->output());
    std::cout << graph->outputs()[0]->uniqueName() << std::endl;
  }

  void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_flatten_9_8(graph, node);
  }
};

}} // namespace ONNX_NAMESPACE::version_conversion
