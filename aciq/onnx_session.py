import copy
from pathlib import Path

import onnx
import onnxruntime as ort


def get_block_output_names(model: onnx.ModelProto) -> list[str]:
  """Return block-final ReLU output names for ResNet models.

  BasicBlock (ResNet18/34): act2 is the final ReLU per block.
  Bottleneck (ResNet50/101/152): act3 is the final ReLU per block.
  Stem act1 is always included.
  """
  relu_outputs = [node.output[0] for node in model.graph.node if node.op_type == "Relu"]

  has_act3 = any("/act3/" in name for name in relu_outputs)
  block_act = "/act3/" if has_act3 else "/act2/"

  # Stem: /act1/Relu (not inside /layer*/)
  stem = [name for name in relu_outputs if "/act1/" in name and "/layer" not in name]
  # Block-final ReLUs
  block_final = [name for name in relu_outputs if block_act in name]

  return stem + block_final


def add_intermediate_outputs(model: onnx.ModelProto, output_names: list[str]) -> onnx.ModelProto:
  """Return a copy of the model with additional intermediate outputs exposed."""
  model = copy.deepcopy(model)
  for name in output_names:
    model.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
  return model


def create_session_with_intermediates(
  model_path: Path, output_names: list[str], cuda: bool = False,
) -> tuple[ort.InferenceSession, list[str]]:
  """Create an ONNX Runtime session that returns intermediate layer outputs.

  Returns (session, all_output_names) where all_output_names = [original_output] + output_names.
  """
  model = onnx.load(str(model_path))
  original_outputs = [o.name for o in model.graph.output]
  model = add_intermediate_outputs(model, output_names)

  opts = ort.SessionOptions()
  opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
  providers = ["CUDAExecutionProvider"] if cuda else ["CPUExecutionProvider"]
  session = ort.InferenceSession(model.SerializeToString(), opts, providers=providers)

  all_output_names = original_outputs + output_names
  return session, all_output_names
