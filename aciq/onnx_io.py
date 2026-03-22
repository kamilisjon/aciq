from dataclasses import dataclass

import onnx


@dataclass
class OnnxLayer:
  tensor: onnx.TensorProto
  op_type: str


def extract_layers(onnx_model: onnx.ModelProto) -> list[OnnxLayer]:
  weight_to_op: dict[str, str] = {}
  for node in onnx_model.graph.node:
    for inp in node.input:
      weight_to_op[inp] = node.op_type

  result = []
  for tensor in onnx_model.graph.initializer:
    assert tensor.name in weight_to_op, f"Initializer '{tensor.name}' has no consuming node"
    result.append(OnnxLayer(tensor=tensor, op_type=weight_to_op[tensor.name]))
  return result
