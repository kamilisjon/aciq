import onnx


def extract_tensors(onnx_model: onnx.ModelProto) -> dict[str, onnx.TensorProto]:
  return {t.name: t for t in onnx_model.graph.initializer}
