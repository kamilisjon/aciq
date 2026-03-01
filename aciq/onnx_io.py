
from pathlib import Path

import onnx
import numpy as np

def extract_layers(onnx_model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {tensor.name: onnx.numpy_helper.to_array(tensor) for tensor in onnx_model.graph.initializer}

def load_onnx(model_path: Path) -> onnx.ModelProto:
    return onnx.load(str(model_path))
