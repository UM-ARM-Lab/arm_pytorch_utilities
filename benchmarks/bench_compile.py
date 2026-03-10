#!/usr/bin/env python
"""Benchmark script for arm_pytorch_utilities torch.compile optimization.

Measures eager-mode and torch.compile performance for refactor-target functions.
Outputs a printed table and JSON file for before/after comparison.

Usage:
    python benchmarks/bench_compile.py --device cpu
    python benchmarks/bench_compile.py --device cuda
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch

from arm_pytorch_utilities import math_utils, linalg, tensor_utils, preprocess, softknn


def bench(fn, *args, warmup=5, repeats=20, device='cpu'):
    """Time a function with warmup and repeats. Returns median time in ms."""
    for _ in range(warmup):
        fn(*args)
    if device == 'cuda':
        torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args)
        if device == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times.sort()
    return times[len(times) // 2]


def try_compile_bench(fn, *args, device='cpu', warmup=5, repeats=20):
    """Try to compile and benchmark a function. Returns (time_ms, success) or (None, False)."""
    try:
        compiled_fn = torch.compile(fn, fullgraph=True)
        # Warmup includes compilation
        for _ in range(warmup):
            compiled_fn(*args)
        if device == 'cuda':
            torch.cuda.synchronize()

        times = []
        for _ in range(repeats):
            if device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            compiled_fn(*args)
            if device == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        times.sort()
        return times[len(times) // 2], True
    except Exception as e:
        return None, False, str(e)


def make_psd(n, device):
    R = torch.randn(n, n, device=device)
    return R.t() @ R + torch.eye(n, device=device) * 0.1


def run_benchmarks(device_str):
    device = torch.device(device_str)
    results = {}

    benchmarks = {}

    # --- Setup inputs ---
    # replace_nan_and_inf
    x_nan = torch.randn(10000, 100, device=device)
    mask = torch.rand_like(x_nan) < 0.1
    x_nan[mask] = float('nan')
    benchmarks['replace_nan_and_inf'] = (math_utils.replace_nan_and_inf, (x_nan.clone(), 0))

    # angular_diff_batch
    a_ang = torch.randn(100000, device=device)
    b_ang = torch.randn(100000, device=device)
    benchmarks['angular_diff_batch'] = (math_utils.angular_diff_batch, (a_ang, b_ang))

    # angle_between_stable
    u_abs = torch.randn(200, 50, device=device)
    v_abs = torch.randn(150, 50, device=device)
    benchmarks['angle_between_stable'] = (math_utils.angle_between_stable, (u_abs, v_abs))

    # cos_sim_pairwise
    x1_cos = torch.randn(500, 50, device=device)
    x2_cos = torch.randn(300, 50, device=device)
    benchmarks['cos_sim_pairwise'] = (math_utils.cos_sim_pairwise, (x1_cos, x2_cos))

    # batch_batch_product
    X_bbp = torch.randn(10000, 20, device=device)
    A_bbp = torch.randn(10000, 20, 20, device=device)
    benchmarks['batch_batch_product'] = (linalg.batch_batch_product, (X_bbp, A_bbp))

    # batch_quadratic_product
    X_bqp = torch.randn(10000, 20, device=device)
    A_bqp = make_psd(20, device)
    benchmarks['batch_quadratic_product'] = (linalg.batch_quadratic_product, (X_bqp, A_bqp))

    # batch_outer_product
    u_bop = torch.randn(10000, 20, device=device)
    v_bop = torch.randn(10000, 20, device=device)
    benchmarks['batch_outer_product'] = (linalg.batch_outer_product, (u_bop, v_bop))

    # squeeze_n
    x_sq = torch.randn(1, 1, 1, 1000, 50, device=device)
    benchmarks['squeeze_n'] = (lambda x: tensor_utils.squeeze_n(x, 3), (x_sq,))

    # MinMaxScaler.transform
    x_mm = torch.randn(10000, 50, device=device)
    scaler = preprocess.MinMaxScaler()
    scaler.fit(x_mm)
    benchmarks['MinMaxScaler.transform'] = (scaler.transform, (x_mm,))

    # SoftKNN.forward
    x_knn = torch.randn(200, 10, device=device)
    knn = softknn.SoftKNN(min_k=20)
    benchmarks['SoftKNN.forward'] = (knn, (x_knn,))

    # sqrtm (CPU only due to .numpy())
    if device_str == 'cpu':
        A_sqrtm = make_psd(50, device)
        benchmarks['sqrtm'] = (linalg.sqrtm, (A_sqrtm,))

    # --- Run benchmarks ---
    print(f"\n{'Function':<30} {'Eager (ms)':>12} {'Compile (ms)':>14} {'Speedup':>10} {'Compile OK':>12}")
    print("-" * 80)

    for name, (fn, args) in benchmarks.items():
        # Eager benchmark
        # For replace_nan_and_inf, need fresh clone each call
        if name == 'replace_nan_and_inf':
            def eager_fn(x_template=x_nan):
                return math_utils.replace_nan_and_inf(x_template.clone(), 0)
            eager_ms = bench(eager_fn, warmup=5, repeats=20, device=device_str)
        else:
            try:
                eager_ms = bench(fn, *args, device=device_str)
            except Exception as e:
                print(f"{name:<30} {'ERROR':>12} {'N/A':>14} {'N/A':>10} {'N/A':>12}  ({e})")
                results[name] = {'eager_ms': None, 'compile_ms': None, 'compile_ok': False,
                                 'compile_error': None, 'eager_error': str(e)}
                continue

        # Compile benchmark
        if name == 'replace_nan_and_inf':
            compile_result = try_compile_bench(eager_fn, device=device_str)
        else:
            compile_result = try_compile_bench(fn, *args, device=device_str)

        if len(compile_result) == 2:
            compile_ms, compile_ok = compile_result
            compile_err = None
        else:
            compile_ms, compile_ok, compile_err = compile_result

        speedup = f"{eager_ms / compile_ms:.2f}x" if compile_ms else "N/A"
        compile_str = f"{compile_ms:.3f}" if compile_ms else "FAIL"

        print(f"{name:<30} {eager_ms:>12.3f} {compile_str:>14} {speedup:>10} {'yes' if compile_ok else 'no':>12}")

        results[name] = {
            'eager_ms': eager_ms,
            'compile_ms': compile_ms,
            'compile_ok': compile_ok,
            'compile_error': compile_err,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark arm_pytorch_utilities functions')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    print(f"Running benchmarks on {args.device}")
    print(f"PyTorch version: {torch.__version__}")

    results = run_benchmarks(args.device)

    # Save JSON
    output_dir = Path(__file__).parent
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f"results_{args.device}_{timestamp}.json"
    output = {
        'device': args.device,
        'torch_version': torch.__version__,
        'timestamp': timestamp,
        'results': results,
    }
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    main()
