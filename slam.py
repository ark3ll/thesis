#!/usr/bin/env python3
# main slam system that ties everything together

import numpy as np
import time

from icp import icp_scan_to_map, pose_to_transform, transform_to_pose, transform_points, downsample_points, MapManager

class ScanToMapSLAM:
    def __init__(self, mode='hybrid', voxel_size=0.1, max_map_points=50000, icp_max_iter=50, icp_tolerance=1e-6, icp_max_distance=1.0):
        self.mode = mode
        self.voxel_size = voxel_size
        self.icp_max_iter = icp_max_iter
        self.icp_tolerance = icp_tolerance
        self.icp_max_distance = icp_max_distance
        
        # map handles all the point clouds
        self.map_manager = MapManager(voxel_size, max_map_points)
        
        # rover state
        self.current_pose = np.eye(3)
        self.trajectory = []
        self.previous_odom_pose = None
        
        # keep track of performance
        self.processing_times = []
        self.icp_convergence = []
        self.icp_errors = []
    
    def process_scan(self, scan, odom_x, odom_y, odom_theta):
        start_time = time.time()
        
        # convert laser scan to points
        angles = np.linspace(-np.pi/2, np.pi/2, len(scan))
        # filter out bad readings
        valid_mask = (np.array(scan) > 0.1) & (np.array(scan) < 30.0)
        
        scan_points = np.column_stack([
            np.array(scan)[valid_mask] * np.cos(angles[valid_mask]),
            np.array(scan)[valid_mask] * np.sin(angles[valid_mask])
        ])
        
        # downsample to speed things up
        scan_points = downsample_points(scan_points, self.voxel_size * 0.5)
        
        # where odometry thinks we are
        odom_pose = pose_to_transform(odom_x, odom_y, odom_theta)
        
        # first scan, use it to init the map
        if self.map_manager.get_map_size() == 0:
            if self.mode == 'pure_icp':
                # start at origin if we're not using odometry
                self.current_pose = np.eye(3)
            else:
                # otherwise trust the odometry for now
                self.current_pose = odom_pose.copy()
            
            world_points = transform_points(scan_points, self.current_pose)
            self.map_manager.add_scan_to_map(world_points)
            
            # save trajectory
            x, y, theta = transform_to_pose(self.current_pose)
            self.trajectory.append([x, y, theta])
            self.previous_odom_pose = odom_pose
            
            self.processing_times.append(time.time() - start_time)
            return
        
        # figure out where we are based on mode
        if self.mode == 'pure_odometry':
            self.current_pose = odom_pose.copy()
            converged = True
            error = 0.0
            
        elif self.mode == 'pure_icp':
            # use last pose as initial guess
            initial_guess = self.current_pose
            
            # let icp figure out where we are
            self.current_pose, converged, error = icp_scan_to_map(
                scan_points, 
                self.map_manager.get_map_tree(), 
                self.map_manager.get_map_points(),
                initial_T=initial_guess,
                max_iterations=self.icp_max_iter,
                tolerance=self.icp_tolerance,
                max_distance=self.icp_max_distance
            )
            # if not converged:
            #     print(f"ICP didn't converge, reason: {error}")
            
        else:  # hybrid mode
            # use odometry change as initial guess
            if self.previous_odom_pose is not None:
                odom_delta = np.linalg.inv(self.previous_odom_pose) @ odom_pose
                initial_guess = self.current_pose @ odom_delta
            else:
                initial_guess = self.current_pose
            
            # refine with icp
            self.current_pose, converged, error = icp_scan_to_map(
                scan_points, 
                self.map_manager.get_map_tree(), 
                self.map_manager.get_map_points(),
                initial_T=initial_guess,
                max_iterations=self.icp_max_iter,
                tolerance=self.icp_tolerance,
                max_distance=self.icp_max_distance
            )
        
        # add new scan to map
        world_points = transform_points(scan_points, self.current_pose)
        self.map_manager.add_scan_to_map(world_points)
        
        # update trajectory
        x, y, theta = transform_to_pose(self.current_pose)
        self.trajectory.append([x, y, theta])
        self.previous_odom_pose = odom_pose
        
        # track metrics
        self.processing_times.append(time.time() - start_time)
        if self.mode != 'pure_odometry':
            self.icp_convergence.append(converged)
            self.icp_errors.append(error if not np.isnan(error) else 0.0)
    
    def get_results(self):
        # save results
        trajectory = np.array(self.trajectory)
        
        results = {
            'mode': self.mode,
            'trajectory': trajectory.tolist(),
            'map_points': self.map_manager.get_map_points().tolist(),
            'metrics': {
                'total_scans': len(self.trajectory),
                'avg_processing_time_ms': np.mean(self.processing_times) * 1000,
                'total_processing_time_s': np.sum(self.processing_times),
                'map_size': self.map_manager.get_map_size()
            }
        }
        
        if self.mode != 'pure_odometry':
            results['metrics']['icp_convergence_rate'] = np.mean(self.icp_convergence) * 100
            results['metrics']['avg_icp_error'] = np.mean(self.icp_errors)
        
        return results