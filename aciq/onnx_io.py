import numpy as np
import onnx


def extract_tensors(onnx_model: onnx.ModelProto) -> dict[str, onnx.TensorProto]:
  return {t.name: t for t in onnx_model.graph.initializer}


def replace_weight(model: onnx.ModelProto, name: str, new_data: np.ndarray) -> None:
  for i, init in enumerate(model.graph.initializer):
    if init.name == name:
      model.graph.initializer[i].CopyFrom(onnx.numpy_helper.from_array(new_data, name=name))
      return
