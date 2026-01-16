"""
Benchmark Script for MouthMap
==============================
Benchmarks all ONNX model variants to compare size, latency, and throughput.

Usage:
    python scripts/benchmark.py --models-dir models/onnx

Output:
    - Console report
    - reports/benchmark_results.json
    - reports/benchmark_report.md
"""

import os
import sys
import time
import json
import argparse
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def benchmark_model(model_path, num_warmup=5, num_runs=50):
    """
    Benchmark a single ONNX model.
    
    Returns dict with size, latency stats, and throughput.
    """
    import onnxruntime as ort
    
    # Load model
    session = ort.InferenceSession(
        model_path,
        providers=['CPUExecutionProvider']
    )
    
    # Get input info
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    input_shape = input_info.shape
    
    # Create test input (matching expected shape)
    # Handle dynamic batch dimension
    shape = list(input_shape)
    if shape[0] is None or shape[0] == 'batch_size':
        shape[0] = 1
    
    test_input = np.random.randn(*shape).astype(np.float32)
    
    # Warmup runs
    for _ in range(num_warmup):
        session.run(None, {input_name: test_input})
    
    # Benchmark runs
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        session.run(None, {input_name: test_input})
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms
    
    latencies = np.array(latencies)
    
    return {
        "model_path": model_path,
        "model_name": os.path.basename(model_path),
        "size_mb": os.path.getsize(model_path) / (1024 * 1024),
        "size_bytes": os.path.getsize(model_path),
        "input_shape": str(input_shape),
        "latency_mean_ms": float(np.mean(latencies)),
        "latency_std_ms": float(np.std(latencies)),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
        "latency_min_ms": float(np.min(latencies)),
        "latency_max_ms": float(np.max(latencies)),
        "throughput_fps": float(1000 / np.mean(latencies)),
    }


def generate_report_markdown(results, output_path):
    """
    Generate a markdown benchmark report.
    """
    # Find baseline (first model or one without quantization suffix)
    baseline = results[0]
    for r in results:
        if "baseline" in r["model_name"].lower():
            baseline = r
            break
    
    report = f"""# MouthMap Model Benchmark Report

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Summary

| Model | Size (MB) | Size Reduction | Latency p50 (ms) | Speedup | Throughput (FPS) |
|-------|-----------|----------------|------------------|---------|------------------|
"""
    
    for r in results:
        size_reduction = (1 - r["size_mb"] / baseline["size_mb"]) * 100 if baseline["size_mb"] > 0 else 0
        speedup = baseline["latency_p50_ms"] / r["latency_p50_ms"] if r["latency_p50_ms"] > 0 else 0
        
        size_str = f"{size_reduction:.1f}%" if size_reduction > 0 else "baseline"
        speedup_str = f"{speedup:.2f}x" if speedup != 1 else "baseline"
        
        report += f"| {r['model_name']} | {r['size_mb']:.2f} | {size_str} | {r['latency_p50_ms']:.1f} | {speedup_str} | {r['throughput_fps']:.1f} |\n"
    
    report += """

## Detailed Latency Statistics

| Model | Mean (ms) | Std (ms) | p50 (ms) | p95 (ms) | p99 (ms) |
|-------|-----------|----------|----------|----------|----------|
"""
    
    for r in results:
        report += f"| {r['model_name']} | {r['latency_mean_ms']:.1f} | {r['latency_std_ms']:.1f} | {r['latency_p50_ms']:.1f} | {r['latency_p95_ms']:.1f} | {r['latency_p99_ms']:.1f} |\n"
    
    report += """

## Recommendations

Based on the benchmarks:

"""
    
    # Find best for each category
    smallest = min(results, key=lambda x: x["size_mb"])
    fastest = min(results, key=lambda x: x["latency_p50_ms"])
    
    report += f"- **Smallest model**: `{smallest['model_name']}` ({smallest['size_mb']:.2f} MB)\n"
    report += f"- **Fastest inference**: `{fastest['model_name']}` ({fastest['latency_p50_ms']:.1f} ms)\n"
    
    # Find best balanced (if we have quantized versions)
    if len(results) > 1:
        # Score based on combined metrics (lower is better)
        def score(r):
            size_norm = r["size_mb"] / baseline["size_mb"]
            latency_norm = r["latency_p50_ms"] / baseline["latency_p50_ms"]
            return size_norm + latency_norm
        
        balanced = min(results, key=score)
        report += f"- **Best balanced**: `{balanced['model_name']}` (good size/speed trade-off)\n"
    
    report += """

## Notes

- All benchmarks run on CPU with ONNX Runtime
- Latency measured over 50 runs after 5 warmup runs
- Input shape: (1, 75, 46, 140, 1) - single video sample
"""
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ONNX model variants"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models/onnx",
        help="Directory containing ONNX models"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=50,
        help="Number of benchmark runs per model"
    )
    args = parser.parse_args()
    
    # Find all ONNX models
    model_files = glob.glob(os.path.join(args.models_dir, "*.onnx"))
    
    if not model_files:
        print(f"No ONNX models found in {args.models_dir}")
        sys.exit(1)
    
    print("=" * 60)
    print("MouthMap: Model Benchmark")
    print("=" * 60)
    print(f"Found {len(model_files)} models to benchmark")
    print(f"Runs per model: {args.num_runs}")
    print("=" * 60)
    
    results = []
    
    for model_path in sorted(model_files):
        model_name = os.path.basename(model_path)
        print(f"\nBenchmarking: {model_name}")
        print("-" * 40)
        
        try:
            result = benchmark_model(model_path, num_runs=args.num_runs)
            results.append(result)
            
            print(f"  Size: {result['size_mb']:.2f} MB")
            print(f"  Latency (p50): {result['latency_p50_ms']:.1f} ms")
            print(f"  Throughput: {result['throughput_fps']:.1f} FPS")
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Create reports directory
    os.makedirs("reports", exist_ok=True)
    
    # Save JSON results
    json_path = "reports/benchmark_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON results saved to: {json_path}")
    
    # Generate markdown report
    generate_report_markdown(results, "reports/benchmark_report.md")
    
    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
