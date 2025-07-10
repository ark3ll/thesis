#!/usr/bin/env python3
# warning: full tuning takes forever to run

import subprocess
import json
import itertools
import os
import time
import numpy as np
from data_loader import load_dataset

# full params to test, based on what seemed reasonable from papers
PARAMETER_GRID = {
    'voxel-size': [0.01, 0.05, 0.1, 0.2],
    'icp-max-iter': [10, 30, 50, 70, 100],
    'icp-tolerance': [1e-7, 1e-6, 1e-5, 1e-4],
    'icp-max-distance': [0.3, 0.5, 1.0, 2.0]
}

# quick test
QUICK_PARAMS = {
    'voxel-size': [0.05, 0.1],
    'icp-max-iter': [70, 100],
    'icp-tolerance': [1e-6],
    'icp-max-distance': [0.3, 0.5]
}

def run_single_experiment(params, dataset='intel', mode='hybrid', max_scans=None):
    
    param_str = '_'.join([f"{k}={v}" for k, v in params.items()])
    output_prefix = f"tuning/{mode}_{param_str}"
    
    cmd = [
        'python3', 'main.py',
        '--mode', mode,
        '--dataset', dataset,
        '--output-prefix', output_prefix
    ]
    
    if max_scans:
        cmd.extend(['--max-scans', str(max_scans)])
    
    for param, value in params.items():
        cmd.extend([f'--{param}', str(value)])
    
    print(f"Running: {' '.join(cmd[6:])}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=500) # 8 minutes timeout, had to increase this as was stopping early
        
        if result.returncode != 0:
            print(f"  ERROR: crashed with code {result.returncode}")
            if result.stderr:
                print(f"  stderr: {result.stderr[:200]}")
            return None
        
        results_file = f"{output_prefix}_{mode}_results.json"
        if not os.path.exists(results_file):
            print(f"  ERROR: cant find results file: {results_file}")
            return None
            
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # delete files to save space
        try:
            os.remove(results_file)
            # also remove the viz if it exists
            viz_file = f"{output_prefix}_{mode}_visualization.png"
            if os.path.exists(viz_file):
                os.remove(viz_file)
        except:
            pass # ignore errors deleting files
            
        return results
        
    except subprocess.TimeoutExpired:
        print(f"  ERROR: timed out")
        return None
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        return None

def calculate_drift(results, ground_truth):
    if not results or 'trajectory' not in results:
        return float('inf')  # failed = infinite drift
    
    traj = np.array(results['trajectory'])
    gt = np.array(ground_truth[:len(traj)])
    
    # euclidean distance at the end
    final_drift = np.sqrt(
        (traj[-1, 0] - gt[-1, 0])**2 + 
        (traj[-1, 1] - gt[-1, 1])**2
    )
    
    return final_drift

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Tune SLAM parameters')
    parser.add_argument('--quick', action='store_true', help='Quick test with fewer parameters')
    parser.add_argument('--max-scans', type=int, default=13631, help='Number of scans per test')
    parser.add_argument('--dataset', type=str, default='intel', help='Dataset to use')
    parser.add_argument('--mode', type=str, default='hybrid', help='SLAM mode to tune')
    args = parser.parse_args()
    
    # make sure output dir exists
    os.makedirs('tuning', exist_ok=True)
    
    # need ground truth to calculate drift
    print(f"Loading {args.dataset} ground truth...")
    _, _, ground_truth = load_dataset(args.dataset, args.max_scans)
    
    param_grid = QUICK_PARAMS if args.quick else PARAMETER_GRID
    
    # make all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"\nTesting {len(all_combinations)} parameter combinations")
    print(f"Dataset: {args.dataset}, Mode: {args.mode}, Scans: {args.max_scans}\n")
    
    results_list = []
    best_drift = float('inf')
    best_params = None
    
    for i, params in enumerate(all_combinations):
        print(f"\n[{i+1}/{len(all_combinations)}] Testing:")
        for k, v in params.items():
            print(f"  {k}: {v}")
        
        results = run_single_experiment(params, args.dataset, args.mode, args.max_scans)
        
        if results:
            drift = calculate_drift(results, ground_truth)
            avg_time = results['metrics']['avg_processing_time_ms']
            
            if args.mode != 'pure_odometry':
                convergence = results['metrics'].get('icp_convergence_rate', 0)
                print(f"  Drift: {drift:.3f}m, Time: {avg_time:.1f}ms, Convergence: {convergence:.1f}%")
            else:
                print(f"  Drift: {drift:.3f}m, Time: {avg_time:.1f}ms")
            
            results_list.append({
                'params': params,
                'drift': drift,
                'avg_time_ms': avg_time,
                'convergence_rate': results['metrics'].get('icp_convergence_rate', 100)
            })
            
            # new best?
            if drift < best_drift:
                best_drift = drift
                best_params = params
                print(f"  *** NEW BEST! ***")
        else:
            print(f"  FAILED :(")
            results_list.append({
                'params': params,
                'drift': float('inf'),
                'avg_time_ms': 0,
                'convergence_rate': 0
            })
    
    # save everything
    output_file = f'tuning/tuning_results_{args.dataset}_{args.mode}.json'
    with open(output_file, 'w') as f:
        json.dump({
            'dataset': args.dataset,
            'mode': args.mode,
            'max_scans': args.max_scans,
            'best_parameters': best_params,
            'best_drift': best_drift,
            'all_results': sorted(results_list, key=lambda x: x['drift'])
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print("DONE!")
    print(f"{'='*60}")
    print(f"\nBest params:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"\nBest drift: {best_drift:.3f} meters")
    print(f"\nSaved to: {output_file}")
    
    # how to use the best params
    print(f"\nRun with best params:")
    cmd_parts = ["python3", "main.py", "--mode", args.mode, "--dataset", args.dataset]
    for k, v in best_params.items():
        cmd_parts.extend([f"--{k}", str(v)])
    print(" ".join(cmd_parts))

if __name__ == '__main__':
    main()