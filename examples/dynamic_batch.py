import argparse

import onnx

BATCH_DIM_NAME = "batch"

def convert_model_batch_to_dynamic(model_path):
    model = onnx.load(model_path)
    initializers = [node.name for node in model.graph.initializer]
    inputs = [node for node in model.graph.input if node.name not in initializers]

    # Set dynamic batch for all inputs
    for inp in inputs:
        shape = inp.type.tensor_type.shape
        if shape.dim and not shape.dim[0].dim_param:
            shape.dim[0].dim_param = BATCH_DIM_NAME

    # Infer shapes
    model = onnx.shape_inference.infer_shapes(model, check_type=True, strict_mode=True, data_prop=True)

    # Set dynamic batch for all outputs
    for out in model.graph.output:
        out_shape = out.type.tensor_type.shape
        if out_shape.dim and out_shape.dim[0].HasField('dim_value'):
            out_shape.dim[0].ClearField('dim_value')
            out_shape.dim[0].dim_param = BATCH_DIM_NAME

    try:
        onnx.checker.check_model(model)
    except Exception as e:
        print(f"Model validation failed. The model will NOT be saved. Error: {e}")
        return None

    new_model_path = model_path.rsplit('.', 1)[0] + "_dynamic.onnx"
    onnx.save(model, new_model_path)
    print(f"Models batch size dim (first dim) converted to dynamic. Model saved at {new_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Path to ONNX model')
    args = parser.parse_args()
    convert_model_batch_to_dynamic(args.model_path) 