"""
Quantization Script for MouthMap
=================================
Applies post-training quantization to the ONNX model.
NO RETRAINING REQUIRED - works on frozen models.

Modes:
    - fp16: Half precision (safest, 50% size reduction)
    - dynamic: INT8 weights, FP32 activations (75% reduction, 0.5-2% loss)
    - static: Full INT8 (requires calibration data)

Usage:
    python scripts/quantize_model.py --input models/onnx/lipnet_baseline.onnx --mode dynamic
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def quantize_fp16(input_path, output_path):
    """
    Convert FP32 model to FP16.
    
    - Size reduction: ~50%
    - Accuracy loss: <0.1% (negligible)
    - Best for: GPU inference, maximum accuracy
    """
    from onnxruntime.transformers import float16
    import onnx
    
    print("Loading model for FP16 conversion...")
    model = onnx.load(input_path)
    
    print("Converting to FP16...")
    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=True  # Keep inputs/outputs as FP32 for compatibility
    )
    
    print(f"Saving to {output_path}...")
    onnx.save(model_fp16, output_path)
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"FP16 model size: {size_mb:.2f} MB")
    
    return output_path


def quantize_dynamic_int8(input_path, output_path):
    """
    Dynamic INT8 quantization.
    
    - Weights: Quantized to INT8 (stored as INT8)
    - Activations: Computed in FP32 at runtime
    - Size reduction: ~75%
    - Accuracy loss: 0.5-2%
    - NO calibration data needed!
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType
    
    print("Applying dynamic INT8 quantization...")
    
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QUInt8,
        extra_options={
            'ActivationSymmetric': False,
            'WeightSymmetric': True,
        }
    )
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Dynamic INT8 model size: {size_mb:.2f} MB")
    
    return output_path


def quantize_static_int8(input_path, output_path, calibration_data_path=None):
    """
    Static INT8 quantization (requires calibration data).
    
    - Everything quantized to INT8
    - Fastest inference
    - Accuracy loss: 2-5% (depends on model)
    """
    from onnxruntime.quantization import quantize_static, QuantType, QuantFormat
    from onnxruntime.quantization import CalibrationDataReader
    import numpy as np
    import glob
    
    class LipNetCalibrationReader(CalibrationDataReader):
        """Provides calibration samples for quantization."""
        
        def __init__(self, data_path, model_path, num_samples=100):
            import onnxruntime as ort
            
            # Get actual input name from the model
            sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            self.input_name = sess.get_inputs()[0].name
            print(f"Using input name from model: {self.input_name}")
            
            if data_path and os.path.isdir(data_path):
                # Load preprocessed .npy files if available
                self.data_files = glob.glob(f"{data_path}/*.npy")[:num_samples]
            else:
                # Generate random calibration data as fallback
                print("Warning: No calibration data found. Using random samples.")
                self.data_files = None
                self.num_samples = num_samples
            
            self.current_idx = 0
            
        def get_next(self):
            if self.data_files:
                if self.current_idx >= len(self.data_files):
                    return None
                data = np.load(self.data_files[self.current_idx])
                self.current_idx += 1
            else:
                if self.current_idx >= self.num_samples:
                    return None
                # Random calibration data
                data = np.random.randn(1, 75, 46, 140, 1).astype(np.float32)
                self.current_idx += 1
            
            # Reshape if needed
            if len(data.shape) == 4:
                data = np.expand_dims(data, 0)
            
            return {self.input_name: data.astype(np.float32)}
        
        def rewind(self):
            self.current_idx = 0
    
    print("Applying static INT8 quantization...")
    
    calibration_reader = LipNetCalibrationReader(calibration_data_path, input_path)
    
    quantize_static(
        model_input=input_path,
        model_output=output_path,
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        extra_options={
            'CalibMovingAverage': True,
            'CalibMaxIntermediateOutputs': 100,
        }
    )
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Static INT8 model size: {size_mb:.2f} MB")
    
    return output_path


def verify_quantized_model(original_path, quantized_path):
    """
    Compare outputs between original and quantized models.
    """
    import onnxruntime as ort
    import numpy as np
    
    print("\nVerifying quantized model...")
    
    # Load both models
    original_session = ort.InferenceSession(
        original_path,
        providers=['CPUExecutionProvider']
    )
    quantized_session = ort.InferenceSession(
        quantized_path,
        providers=['CPUExecutionProvider']
    )
    
    # Get input name
    input_name = original_session.get_inputs()[0].name
    
    # Test with random input
    test_input = np.random.randn(1, 75, 46, 140, 1).astype(np.float32)
    
    # Run inference
    original_output = original_session.run(None, {input_name: test_input})[0]
    
    # Quantized model might have different input name
    q_input_name = quantized_session.get_inputs()[0].name
    quantized_output = quantized_session.run(None, {q_input_name: test_input})[0]
    
    # Compare
    max_diff = np.max(np.abs(original_output - quantized_output))
    mean_diff = np.mean(np.abs(original_output - quantized_output))
    
    print(f"Max output difference: {max_diff:.6f}")
    print(f"Mean output difference: {mean_diff:.6f}")
    
    # Check if predictions would be the same (CTC decoding)
    original_pred = np.argmax(original_output, axis=-1)
    quantized_pred = np.argmax(quantized_output, axis=-1)
    prediction_match = np.array_equal(original_pred, quantized_pred)
    
    print(f"Prediction match: {'✅ Yes' if prediction_match else '⚠️ No'}")
    
    return prediction_match


def main():
    parser = argparse.ArgumentParser(
        description="Apply post-training quantization to ONNX model"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="models/onnx/lipnet_baseline.onnx",
        help="Input ONNX model path"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["fp16", "dynamic", "static", "all"],
        default="dynamic",
        help="Quantization mode"
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default=None,
        help="Path to calibration data (for static mode)"
    )
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Get input size for comparison
    input_size = os.path.getsize(args.input) / (1024 * 1024)
    
    print("=" * 60)
    print("MouthMap: Post-Training Quantization")
    print("=" * 60)
    print(f"Input model: {args.input}")
    print(f"Input size: {input_size:.2f} MB")
    print(f"Mode: {args.mode}")
    print("=" * 60)
    
    output_dir = os.path.dirname(args.input)
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    
    outputs = []
    
    if args.mode in ["fp16", "all"]:
        output = os.path.join(output_dir, f"{base_name}_fp16.onnx")
        quantize_fp16(args.input, output)
        outputs.append(("FP16", output))
    
    if args.mode in ["dynamic", "all"]:
        output = os.path.join(output_dir, f"{base_name}_int8_dynamic.onnx")
        quantize_dynamic_int8(args.input, output)
        outputs.append(("Dynamic INT8", output))
    
    if args.mode in ["static", "all"]:
        output = os.path.join(output_dir, f"{base_name}_int8_static.onnx")
        quantize_static_int8(args.input, output, args.calibration)
        outputs.append(("Static INT8", output))
    
    # Verify each quantized model
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)
    
    for name, output_path in outputs:
        print(f"\n{name}:")
        verify_quantized_model(args.input, output_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Variant':<20} {'Size (MB)':<12} {'Reduction':<12}")
    print("-" * 44)
    print(f"{'Original':<20} {input_size:<12.2f} {'--':<12}")
    
    for name, output_path in outputs:
        size = os.path.getsize(output_path) / (1024 * 1024)
        reduction = (1 - size / input_size) * 100
        print(f"{name:<20} {size:<12.2f} {reduction:>10.1f}%")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
