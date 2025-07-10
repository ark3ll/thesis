#!/usr/bin/env python3
# core slam functions

import numpy as np
from scipy.spatial import KDTree

# transformation utilities 

def pose_to_transform(x, y, theta):
    # convert 2d pose to homogeneous transform matrix
    T = np.eye(3)
    cos, sin = np.cos(theta), np.sin(theta)
    T[0, 0], T[0, 1] = cos, -sin
    T[1, 0], T[1, 1] = sin, cos
    T[0, 2], T[1, 2] = x, y
    return T

def transform_to_pose(T):
    # extract pose from transform matrix
    x, y = T[0, 2], T[1, 2]
    theta = np.arctan2(T[1, 0], T[0, 0])
    return x, y, theta

def transform_points(points, T):
    # apply transformation to point cloud
    if points.size == 0:
        return points
    points_h = np.column_stack([points, np.ones(len(points))])
    transformed = (T @ points_h.T).T
    return transformed[:, :2]

def downsample_points(points, voxel_size):
    # voxel grid downsampling
    if len(points) == 0:
        return points
    
    voxel_indices = np.floor(points / voxel_size).astype(int)
    
    # keep one point per voxel
    unique_voxels, indices = np.unique(voxel_indices, axis=0, return_index=True)
    
    return points[indices]

# icp implementation

def icp_scan_to_map(scan, map_tree, map_points, initial_T=None, max_iterations=50, tolerance=1e-6, max_distance=1.0):
    # align scan to map using icp
    if initial_T is None:
        T = np.eye(3)
    else:
        T = initial_T.copy()
    
    prev_error = float('inf')
    converged = False
    error = float('inf')
    
    for iteration in range(max_iterations):
        # transform scan with current estimate
        transformed_scan = transform_points(scan, T)
        
        distances, indices = map_tree.query(transformed_scan, k=1)
        
        # only keep good matches
        mask = distances < max_distance
        if np.sum(mask) < 10:  # need enough points
            # print(f"ICP: Not enough correspondences: {np.sum(mask)}")
            break
            
        matched_scan = transformed_scan[mask]
        matched_map = map_points[indices[mask].flatten()]
        
        # svd gives us the optimal rotation
        scan_center = np.mean(matched_scan, axis=0)
        map_center = np.mean(matched_map, axis=0)
        
        scan_centered = matched_scan - scan_center
        map_centered = matched_map - map_center
        
        # svd
        H = scan_centered.T @ map_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # fix reflection if determinant is negative
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # get translation
        t = map_center - R @ scan_center
        
        T_increment = np.eye(3)
        T_increment[:2, :2] = R
        T_increment[:2, 2] = t
        
        T = T_increment @ T
        
        error = np.mean(distances[mask])
        if abs(prev_error - error) < tolerance:
            converged = True
            break
        prev_error = error
    
    return T, converged, error

# map manager

class MapManager:
    def __init__(self, voxel_size=0.1, max_points=50000):
        self.voxel_size = voxel_size
        self.max_points = max_points
        self.map_points = np.empty((0, 2))
        self.map_tree = None
    
    def add_scan_to_map(self, scan_points):
        # add new scan to global map
        if len(self.map_points) == 0:
            # for first scan just add everything
            self.map_points = downsample_points(scan_points, self.voxel_size)
            self.map_tree = KDTree(self.map_points)
            return
        
        # only add points that aren't already in the map
        if len(self.map_points) < self.max_points:
            new_points = []
            for point in downsample_points(scan_points, self.voxel_size):
                dist, _ = self.map_tree.query(point.reshape(1, -1), k=1)
                if dist[0] > self.voxel_size * 0.5:
                    new_points.append(point)
            
            if new_points:
                self.map_points = np.vstack([self.map_points, new_points])
                
                # downsample if map is getting too big
                if len(self.map_points) > self.max_points * 1.2:
                    # print(f"Map getting big ({len(self.map_points)} points), downsampling.")
                    self.map_points = downsample_points(self.map_points, self.voxel_size)
                
                # rebuild tree
                self.map_tree = KDTree(self.map_points)
    
    def get_map_points(self):
        return self.map_points
    
    def get_map_tree(self):
        return self.map_tree
    
    def get_map_size(self):
        return len(self.map_points)