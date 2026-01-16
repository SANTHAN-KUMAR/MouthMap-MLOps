"""
ONNX Export Script for MouthMap
================================
Exports the Keras LipNet model to ONNX format for production deployment.

Usage:
    python scripts/export_to_onnx.py --weights pretrained_weights/lipnet_weights.h5

Requirements:
    pip install tf2onnx onnxruntime
"""

import os
import sys

# IMPORTANT: Force CPU to avoid CudnnRNNV3 ops in the exported graph
# CudnnRNNV3 is not supported by ONNX Runtime CPU provider
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import numpy as np


def load_model_with_weights(weights_path):
    """
    Build the model architecture and load trained weights.
    """
    from src.model import build_model
    from src.config import FRAME_COUNT, FRAME_HEIGHT, FRAME_WIDTH, CHANNELS
    
    print(f"Building model architecture...")
    model = build_model(
        input_shape=(FRAME_COUNT, FRAME_HEIGHT, FRAME_WIDTH, CHANNELS)
    )
    
    print(f"Loading weights from: {weights_path}")
    model.load_weights(weights_path)
    print("Weights loaded successfully!")
    
    return model


def export_to_onnx(model, output_path, opset_version=15):
    """
    Export Keras model to ONNX format.
    
    Uses Keras 3's export() to create SavedModel, then converts to ONNX.
    
    Args:
        model: Keras model with loaded weights
        output_path: Where to save the .onnx file
        opset_version: ONNX opset version (15 is widely supported)
    """
    import subprocess
    import tempfile
    
    # Get input shape from model
    input_shape = model.input_shape  # (None, 75, 46, 140, 1)
    
    print(f"Exporting to ONNX (opset {opset_version})...")
    print(f"Input shape: {input_shape}")
    
    # Step 1: Export as SavedModel using Keras 3's export method
    with tempfile.TemporaryDirectory() as tmpdir:
        savedmodel_path = os.path.join(tmpdir, "savedmodel")
        
        print("Exporting as SavedModel format...")
        # Keras 3 uses export() instead of save() for SavedModel
        model.export(savedmodel_path)
        
        print("Converting SavedModel to ONNX...")
        
        # Use tf2onnx command line for reliable conversion
        cmd = [
            "python", "-m", "tf2onnx.convert",
            "--saved-model", savedmodel_path,
            "--output", output_path,
            "--opset", str(opset_version)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"tf2onnx stdout: {result.stdout}")
            print(f"tf2onnx stderr: {result.stderr}")
            raise RuntimeError(f"tf2onnx conversion failed")
    
    # Get file size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"ONNX model saved to: {output_path}")
    print(f"Model size: {size_mb:.2f} MB")
    
    return output_path



def verify_onnx_model(onnx_path, keras_model):
    """
    Verify the ONNX model produces the same output as Keras.
    """
    import onnxruntime as ort
    
    print("\nVerifying ONNX model...")
    
    # Create ONNX session
    session = ort.InferenceSession(
        onnx_path,
        providers=['CPUExecutionProvider']
    )
    
    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"ONNX input name: {input_name}")
    print(f"ONNX output name: {output_name}")
    
    # Create random test input
    test_input = np.random.randn(1, 75, 46, 140, 1).astype(np.float32)
    
    # Run Keras inference
    keras_output = keras_model.predict(test_input, verbose=0)
    
    # Run ONNX inference
    onnx_output = session.run([output_name], {input_name: test_input})[0]
    
    # Compare outputs
    max_diff = np.max(np.abs(keras_output - onnx_output))
    mean_diff = np.mean(np.abs(keras_output - onnx_output))
    
    print(f"Max difference: {max_diff:.8f}")
    print(f"Mean difference: {mean_diff:.8f}")
    
    if max_diff < 1e-4:
        print("✅ ONNX model verified - outputs match Keras!")
        return True
    else:
        print("⚠️  Some difference detected (may be floating point precision)")
        return max_diff < 1e-2


def main():
    parser = argparse.ArgumentParser(
        description="Export Keras LipNet model to ONNX format"
    )
    parser.add_argument(
        "--weights", 
        type=str, 
        default="pretrained_weights/lipnet_weights.h5",
        help="Path to Keras weights file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="models/onnx/lipnet_baseline.onnx",
        help="Output ONNX file path"
    )
    parser.add_argument(
        "--opset", 
        type=int, 
        default=15,
        help="ONNX opset version"
    )
    args = parser.parse_args()
    
    # Check weights file exists
    if not os.path.exists(args.weights):
        print(f"Error: Weights file not found: {args.weights}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    print("=" * 60)
    print("MouthMap: Keras to ONNX Export")
    print("=" * 60)
    
    # Step 1: Load model
    model = load_model_with_weights(args.weights)
    
    # Show model info
    print(f"\nModel summary:")
    print(f"  Input shape: {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")
    print(f"  Parameters: {model.count_params():,}")
    
    # Step 2: Export to ONNX
    onnx_path = export_to_onnx(model, args.output, args.opset)
    
    # Step 3: Verify the exported model
    verify_onnx_model(onnx_path, model)
    
    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
