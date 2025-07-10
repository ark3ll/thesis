#!/usr/bin/env python3
# runs all three slam modes and compares them

import subprocess
import json
import os
import numpy as np
from data_loader import load_intel_data
from visualization import create_thesis_visualization

def run_slam_mode(mode, params=None, max_scans=None):
    print(f"\nRunning {mode} SLAM...")
    
    cmd = [
        'python3', 'main.py',
        '--mode', mode,
        '--output-prefix', 'thesis_results/intel'
    ]
    
    if max_scans:
        cmd.extend(['--max-scans', str(max_scans)])
    
    # add icp params for modes that need them
    if params and mode != 'pure_odometry':
        for param, value in params.items():
            cmd.extend([f'--{param}', str(value)])
    
    # print(" ".join(cmd))  # debug - see what command we're running
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running {mode}:")
        print(result.stderr)
        return None
    
    # load the results json
    with open(f'thesis_results/intel_{mode}_results.json', 'r') as f:
        return json.load(f)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run thesis experiments')
    parser.add_argument('--max-scans', type=int, default=None,
                        help='Limit number of scans (default: all)')
    parser.add_argument('--params-file', type=str, default=None,
                        help='JSON file with tuned ICP parameters')
    args = parser.parse_args()
    
    os.makedirs('thesis_results', exist_ok=True)
    
    # load best params or use defaults
    icp_params = None
    if args.params_file:
        print(f"Loading ICP parameters from {args.params_file}")
        with open(args.params_file, 'r') as f:
            tuning_results = json.load(f)
            if 'best_parameters' in tuning_results:
                icp_params = tuning_results['best_parameters']
            else:
                icp_params = tuning_results
            print("Using tuned parameters:")
            for param, value in icp_params.items():
                print(f"  {param}: {value}")
    else:
        # set defaults to best params found from tuning
        print("Using default ICP parameters")
        icp_params = {
            'voxel-size': 0.05,
            'icp-max-iter': 70,
            'icp-tolerance': 1e-6,
            'icp-max-distance': 0.3
        }
    
    # need ground truth for calculating drift
    print("\nLoading ground truth data...")
    _, _, ground_truth = load_intel_data(
        'RawData/intel.clf',
        'RawData/intel.gfs.log',
        max_scans=args.max_scans
    )
    
    results = {}
    
    results['pure_odometry'] = run_slam_mode('pure_odometry', max_scans=args.max_scans)
    
    results['pure_icp'] = run_slam_mode('pure_icp', params=icp_params, max_scans=args.max_scans)
    
    results['hybrid'] = run_slam_mode('hybrid', params=icp_params, max_scans=args.max_scans)
     
    if all(results.values()):
        print("\nAll SLAM modes completed successfully!")
        
        print("\nCreating thesis visualization...")
        create_thesis_visualization(results, ground_truth)
        
        print("\n" + "="*60)
        print("THESIS RESULTS SUMMARY")
        print("="*60)
        
        for method, result in results.items():
            traj = np.array(result['trajectory'])
            gt_array = np.array(ground_truth)
            final_drift = np.sqrt(
                (traj[-1, 0] - gt_array[len(traj)-1, 0])**2 + 
                (traj[-1, 1] - gt_array[len(traj)-1, 1])**2
            )
            
            print(f"\n{method.replace('_', ' ').upper()}:")
            print(f"  Final drift: {final_drift:.3f} meters")
            print(f"  Processing time: {result['metrics']['avg_processing_time_ms']:.1f} ms/scan")
            print(f"  Total time: {result['metrics']['total_processing_time_s']:.1f} seconds")
            if method != 'pure_odometry':
                print(f"  ICP convergence: {result['metrics']['icp_convergence_rate']:.1f}%")
        
        combined_results = {
            'ground_truth_length': len(ground_truth),
            'methods': results,
            'icp_parameters': icp_params
        }
        
        with open('thesis_results/combined_results.json', 'w') as f:
            json.dump(combined_results, f, indent=2)
        
        print(f"\nCombined results saved to thesis_results/combined_results.json")
        
    else:
        print("\nError: Some SLAM modes failed to complete")

if __name__ == '__main__':
    main()