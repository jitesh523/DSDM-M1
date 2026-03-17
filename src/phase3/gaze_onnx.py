import os
import sys
import torch
import onnx
import onnxruntime as ort
import numpy as np

# Add src to python path so phase2 import works
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from phase2.models import GazeMLP

def export_gaze_model(pytorch_model_path, onnx_model_path):
    print(f"Loading PyTorch model from {pytorch_model_path}...")
    
    # Initialize model (GazeMLP has default input_dim=19, num_classes=11)
    model = GazeMLP()
    
    # Try to load weights if they exist, otherwise use random for demo/infra setup
    if os.path.exists(pytorch_model_path):
        try:
            model.load_state_dict(torch.load(pytorch_model_path, map_location=torch.device('cpu')))
            print("Successfully loaded trained weights.")
        except Exception as e:
            print(f"Failed to load weights: {e}. Exporting with random weights.")
    else:
        print("No trained weights found. Exporting random weight model for testing.")

    model.eval()

    # Input: (batch, 19 dims)
    dummy_input = torch.randn(1, 19, device='cpu')

    print(f"Exporting to ONNX at {onnx_model_path}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_model_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    print("Checking ONNX model...")
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid.")

    print("Verifying outputs between PyTorch and ONNX Runtime...")
    with torch.no_grad():
        torch_out = model(dummy_input)

    ort_session = ort.InferenceSession(onnx_model_path)
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outs = ort_session.run(None, ort_inputs)

    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("ONNX Runtime output matches PyTorch output. Verification successful!")

if __name__ == "__main__":
    pytorch_path = os.path.join(os.path.dirname(__file__), '../../models/gaze_best.pth')
    onnx_path = os.path.join(os.path.dirname(__file__), '../../models/gaze_best.onnx')
    
    export_gaze_model(pytorch_path, onnx_path)
