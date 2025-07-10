#!/usr/bin/env python3
# loads and processes both datasets in different formats

import numpy as np

def load_intel_data(clf_file, gfs_file, max_scans=None):
    # load intel dataset, returns laser scans, noisy odometry, and ground truth
    scans = []
    noisy_odom = []
    ground_truth = []
    
    # clf file has laser + odometry mixed together
    print("Loading laser scans and noisy odometry...")
    with open(clf_file, 'r') as f:
        for line in f:
            if line.startswith('FLASER'):
                parts = line.split()
                num_readings = int(parts[1])
                
                # laser readings start at index 2
                laser_data = [float(parts[i]) for i in range(2, 2 + num_readings)]
                scans.append(laser_data)

                
                # odometry
                odom_idx = 2 + num_readings
                x = float(parts[odom_idx])
                y = float(parts[odom_idx + 1])
                theta = float(parts[odom_idx + 2])
                noisy_odom.append((x, y, theta))
                
                if max_scans and len(scans) >= max_scans:
                    break
    
    # gfs has the corrected poses
    print("Loading ground truth...")
    with open(gfs_file, 'r') as f:
        for line in f:
            if line.startswith('ODOM'):
                parts = line.split()
                x = float(parts[1])
                y = float(parts[2])
                theta = float(parts[3])
                ground_truth.append((x, y, theta))
                
                if max_scans and len(ground_truth) >= max_scans:
                    break
    
    # make sure everything matches up
    min_len = min(len(scans), len(noisy_odom), len(ground_truth))
    scans = scans[:min_len]
    noisy_odom = noisy_odom[:min_len]
    ground_truth = ground_truth[:min_len]
    
    print(f"Loaded {len(scans)} scans")
    return scans, noisy_odom, ground_truth

def load_freiburg_data(log_file, gt_file, max_scans=None):
    # carmen format, used by fr079 dataset
    scans = []
    noisy_odom = []
    ground_truth = []
    
    print(f"Loading fr079 log file: {log_file}")
    with open(log_file, 'r') as f:
        for line in f:
            if line.startswith('FLASER'):
                parts = line.split()
                num_readings = int(parts[1])
                
                # same format as intel, just copied the code
                laser_data = [float(parts[i]) for i in range(2, 2 + num_readings)]
                scans.append(laser_data)
                
                # odometry after scan
                odom_idx = 2 + num_readings
                x = float(parts[odom_idx])
                y = float(parts[odom_idx + 1])
                theta = float(parts[odom_idx + 2])
                noisy_odom.append((x, y, theta))
                
                if max_scans and len(scans) >= max_scans:
                    break
    
    # ground truth file is different format
    print(f"Loading ground truth file: {gt_file}")
    with open(gt_file, 'r') as f:
        for line in f:
            # skip comments in file
            if line.strip() and not line.startswith('%'):
                parts = line.split()
                if len(parts) >= 4:
                    # timestamp x y theta
                    x = float(parts[1])
                    y = float(parts[2])
                    theta = float(parts[3])
                    ground_truth.append((x, y, theta))
                    
                    if max_scans and len(ground_truth) >= max_scans:
                        break
    
    min_len = min(len(scans), len(noisy_odom), len(ground_truth))
    scans = scans[:min_len]
    noisy_odom = noisy_odom[:min_len]
    ground_truth = ground_truth[:min_len]
    
    # fr079 has different coordinate origin than ground truth
    if noisy_odom and ground_truth:
        odom_start = noisy_odom[0]
        gt_start = ground_truth[0]
        
        # shift odometry to match
        aligned_odom = []
        for x, y, theta in noisy_odom:
            aligned_odom.append((
                x - odom_start[0] + gt_start[0],
                y - odom_start[1] + gt_start[1],
                theta - odom_start[2] + gt_start[2]
            ))
        noisy_odom = aligned_odom
    
    print(f"Loaded {len(scans)} scans from fr079 dataset")
    return scans, noisy_odom, ground_truth

def load_dataset(dataset_name, max_scans=None):
    # wrapper to load different datasets
    if dataset_name == 'intel':
        return load_intel_data('RawData/intel.clf', 'RawData/intel.gfs.log', max_scans)
    
    elif dataset_name == 'fr079':
        return load_freiburg_data('RawData/fr079_uncorrected.log', 'RawData/fr079-corrected-odometry.gt.txt', max_scans)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")