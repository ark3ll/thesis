#!/usr/bin/env python3
# main slam runner

import argparse
import json
import numpy as np

from data_loader import load_dataset
from slam import ScanToMapSLAM
from visualization import visualize_single_result

def main():
    parser = argparse.ArgumentParser(description='Simple Scan-to-Map SLAM')
    # lots of arguments here, probably should be in a config file
    parser.add_argument('--mode', type=str, default='hybrid', choices=['pure_odometry', 'pure_icp', 'hybrid'], help='SLAM mode to run')
    parser.add_argument('--max-scans', type=int, default=None, help='Maximum number of scans to process')
    parser.add_argument('--voxel-size', type=float, default=0.05, help='Voxel size for map downsampling (meters)')
    parser.add_argument('--icp-max-iter', type=int, default=70, help='Maximum ICP iterations')
    parser.add_argument('--icp-tolerance', type=float, default=1e-6, help='ICP convergence tolerance')
    parser.add_argument('--icp-max-distance', type=float, default=0.3, help='Maximum point matching distance')
    parser.add_argument('--output-prefix', type=str, default='slam', help='Output file prefix')
    parser.add_argument('--visualize', action='store_true', help='Create visualization after processing')
    parser.add_argument('--dataset', type=str, default='intel', help='Dataset to use: intel, or fr079')

    args = parser.parse_args()
    
    # load the dataset
    scans, noisy_odom, ground_truth = load_dataset(args.dataset, args.max_scans)
    
    # create slam system
    print(f"\nRunning {args.mode} SLAM...")
    slam = ScanToMapSLAM(
        mode=args.mode,
        voxel_size=args.voxel_size,
        icp_max_iter=args.icp_max_iter,
        icp_tolerance=args.icp_tolerance,
        icp_max_distance=args.icp_max_distance
    )
    
    # process each scan
    for i, (scan, odom) in enumerate(zip(scans, noisy_odom)):
        if i % 100 == 0:
            print(f"Processing scan {i}/{len(scans)}...")
        slam.process_scan(scan, odom[0], odom[1], odom[2])
    
    # get final results
    results = slam.get_results()
    
    # save to json
    output_file = f"{args.output_prefix}_{args.mode}_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # print final stats
    print(f"\nResults saved to {output_file}")
    final_drift = np.sqrt((results['trajectory'][-1][0] - ground_truth[-1][0])**2 + 
                         (results['trajectory'][-1][1] - ground_truth[-1][1])**2)
    print(f"Final drift: {final_drift:.2f} meters")
    print(f"Average processing time: {results['metrics']['avg_processing_time_ms']:.1f} ms")
    if args.mode != 'pure_odometry':
        print(f"ICP convergence rate: {results['metrics']['icp_convergence_rate']:.1f}%")
    
    # make a plot if asked
    if args.visualize:
        print("\nCreating visualization...")
        visualize_single_result(results, ground_truth, args.mode, output_file=f"{args.output_prefix}_{args.mode}_visualization.png")

if __name__ == '__main__':
    main()