#!/usr/bin/env python3
# visualization functions for slam results

import numpy as np
import matplotlib.pyplot as plt

def visualize_single_result(results, ground_truth, mode, output_file='slam_result.png'):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # convert to numpy arrays
    gt_array = np.array(ground_truth)
    traj = np.array(results['trajectory'])
    
    # trajectory plot
    ax = axes[0, 0]
    ax.plot(gt_array[:, 0], gt_array[:, 1], 'k--', linewidth=2, label='Ground Truth', alpha=0.7)
    ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, label=f'{mode.replace("_", " ").title()}')
    
    # start and end markers
    ax.scatter(traj[0, 0], traj[0, 1], c='green', s=100, marker='o', edgecolors='black', linewidth=2, label='Start', zorder=5)
    ax.scatter(traj[-1, 0], traj[-1, 1], c='red', s=100, marker='s', edgecolors='black', linewidth=2, label='End', zorder=5)
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title(f'Trajectory: {mode.replace("_", " ").title()}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # map visualization
    ax = axes[0, 1]
    map_points = np.array(results['map_points'])
    # print(f"Map has {len(map_points)} points")  # debug
    
    if len(map_points) > 0:
        # downsample for visualization if too many points
        if len(map_points) > 10000:
            indices = np.random.choice(len(map_points), 10000, replace=False)
            map_points_viz = map_points[indices]
            # print(f"Downsampled to {len(map_points_viz)} for viz")
        else:
            map_points_viz = map_points
        
        ax.scatter(map_points_viz[:, 0], map_points_viz[:, 1], s=0.5, c='gray', alpha=0.5)
    
    ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title(f'Final Map ({len(map_points)} points)')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # drift over time
    ax = axes[1, 0]
    drift = np.sqrt((traj[:, 0] - gt_array[:len(traj), 0])**2 + (traj[:, 1] - gt_array[:len(traj), 1])**2)
    ax.plot(drift, 'b-', linewidth=2)
    ax.axhline(y=drift[-1], color='r', linestyle='--', alpha=0.5, label=f'Final drift: {drift[-1]:.2f}m')
    ax.set_xlabel('Scan Number')
    ax.set_ylabel('Position Error (meters)')
    ax.set_title('Drift Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # metrics bar chart instead of boring text
    ax = axes[1, 1]
    
    # calculate metrics
    final_drift = drift[-1]
    avg_drift = np.mean(drift)
    max_drift = np.max(drift)
    avg_time = results['metrics']['avg_processing_time_ms']
    
    # create bar chart
    metrics = ['Final\nDrift', 'Average\nDrift', 'Max\nDrift']
    values = [final_drift, avg_drift, max_drift]
    
    bars = ax.bar(metrics, values, color=['#e74c3c', '#3498db', '#2ecc71'], alpha=0.8)
    
    # add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{val:.2f}m', ha='center', va='bottom', fontsize=12)
    
    ax.set_ylabel('Drift (meters)')
    ax.set_title('Performance Summary')
    ax.grid(True, axis='y', alpha=0.3)
    
    # add processing time as subtitle
    ax.text(0.5, -0.15, f'Avg processing time: {avg_time:.1f} ms/scan', transform=ax.transAxes, ha='center', fontsize=10)
    
    plt.suptitle(f'SLAM Results: {mode.replace("_", " ").title()}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {output_file}")

def visualize_comparison(results_dict, ground_truth, output_file='slam_comparison.png'):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # colors for each method
    colors = {
        'pure_odometry': 'red',
        'pure_icp': 'blue',
        'hybrid': 'green'
    }
    
    # convert ground truth
    gt_array = np.array(ground_truth)
    
    # trajectory comparison
    ax = axes[0, 0]
    ax.plot(gt_array[:, 0], gt_array[:, 1], 'k--', linewidth=2, label='Ground Truth')
    
    for method, results in results_dict.items():
        traj = np.array(results['trajectory'])
        ax.plot(traj[:, 0], traj[:, 1], colors[method], linewidth=2, label=method.replace('_', ' ').title())
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title('Trajectory Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # final map (hybrid method usually best)
    ax = axes[0, 1]
    if 'hybrid' in results_dict:
        map_points = np.array(results_dict['hybrid']['map_points'])
        if len(map_points) > 0:
            ax.scatter(map_points[:, 0], map_points[:, 1], s=0.5, c='gray', alpha=0.5)
        
        traj = np.array(results_dict['hybrid']['trajectory'])
        ax.plot(traj[:, 0], traj[:, 1], 'g-', linewidth=2, label='Hybrid Trajectory')
        
        # add start and end markers
        ax.scatter(traj[0, 0], traj[0, 1], c='green', s=100, marker='o', edgecolors='black', linewidth=2, label='Start', zorder=5)
        ax.scatter(traj[-1, 0], traj[-1, 1], c='red', s=100, marker='s', edgecolors='black', linewidth=2, label='End', zorder=5)
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title('Final Map (Hybrid Method)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # drift over time for all methods
    ax = axes[1, 0]
    
    for method, results in results_dict.items():
        traj = np.array(results['trajectory'])
        # compute drift
        drift = np.sqrt((traj[:, 0] - gt_array[:len(traj), 0])**2 + (traj[:, 1] - gt_array[:len(traj), 1])**2)
        ax.plot(drift, colors[method], linewidth=2, label=f"{method.replace('_', ' ').title()} (Final: {drift[-1]:.2f}m)")
    
    ax.set_xlabel('Scan Number')
    ax.set_ylabel('Position Error (meters)')
    ax.set_title('Drift Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # final drift comparison
    ax = axes[1, 1]
    
    methods = []
    final_drifts = []
    
    for method, results in results_dict.items():
        traj = np.array(results['trajectory'])
        final_drift = np.sqrt((traj[-1, 0] - gt_array[len(traj)-1, 0])**2 + (traj[-1, 1] - gt_array[len(traj)-1, 1])**2)
        methods.append(method.replace('_', ' ').title())
        final_drifts.append(final_drift)
    
    bars = ax.bar(methods, final_drifts, color=[colors[m] for m in results_dict.keys()], alpha=0.8)
    
    for bar, drift in zip(bars, final_drifts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{drift:.2f}m', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Final Drift (meters)')
    ax.set_title('Final Drift Comparison')
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.suptitle('SLAM Comparison: Pure Odometry vs Pure ICP vs Hybrid', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {output_file}")

def create_thesis_visualization(results_dict, ground_truth):    
    # create figure with specific layout
    fig = plt.figure(figsize=(16, 10))
    
    # main trajectory plot
    ax1 = plt.subplot(2, 3, (1, 2))
    
    # plot ground truth
    gt_array = np.array(ground_truth)
    ax1.plot(gt_array[:, 0], gt_array[:, 1], 'k--', linewidth=2, 
             label='Ground Truth', alpha=0.7)
    
    # plot each method
    colors = {
        'pure_odometry': '#e74c3c', 
        'pure_icp': '#3498db',        
        'hybrid': '#2ecc71'           
    }
    
    for method, results in results_dict.items():
        traj = np.array(results['trajectory'])
        
        # calculate final drift for label
        final_drift = np.sqrt(
            (traj[-1, 0] - gt_array[len(traj)-1, 0])**2 + 
            (traj[-1, 1] - gt_array[len(traj)-1, 1])**2
        )
        
        ax1.plot(traj[:, 0], traj[:, 1], 
                color=colors[method], 
                linewidth=2.5,
                label=f"{method.replace('_', ' ').title()} (drift: {final_drift:.2f}m)",
                alpha=0.8)
    
    # add start and end markers
    ax1.scatter(gt_array[0, 0], gt_array[0, 1], c='green', s=200, marker='o', edgecolors='black', linewidth=2, zorder=5)
    ax1.text(gt_array[0, 0] + 0.5, gt_array[0, 1], 'START', fontsize=10, fontweight='bold')
    ax1.scatter(gt_array[-1, 0], gt_array[-1, 1], c='red', s=200, marker='s', edgecolors='black', linewidth=2, zorder=5)
    ax1.text(gt_array[-1, 0] + 0.5, gt_array[-1, 1], 'END', fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('X Position (meters)', fontsize=12)
    ax1.set_ylabel('Y Position (meters)', fontsize=12)
    ax1.set_title('Trajectory Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # drift evolution plot
    ax2 = plt.subplot(2, 3, 3)
    
    for method, results in results_dict.items():
        traj = np.array(results['trajectory'])
        drift = np.sqrt(
            (traj[:, 0] - gt_array[:len(traj), 0])**2 + 
            (traj[:, 1] - gt_array[:len(traj), 1])**2
        )
        ax2.plot(drift, color=colors[method], linewidth=2, 
                label=method.replace('_', ' ').title())
    
    ax2.set_xlabel('Scan Number', fontsize=12)
    ax2.set_ylabel('Position Error (meters)', fontsize=12)
    ax2.set_title('Error Accumulation Over Time', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # map visualization (hybrid result)
    ax3 = plt.subplot(2, 3, 4)
    
    if 'hybrid' in results_dict:
        map_points = np.array(results_dict['hybrid']['map_points'])
        # print(f"Final map has {len(map_points)} points")
        
        if len(map_points) > 0:
            # downsample for visualization
            indices = np.random.choice(len(map_points), 
                                     min(10000, len(map_points)), 
                                     replace=False)
            ax3.scatter(map_points[indices, 0], map_points[indices, 1], s=0.5, c='gray', alpha=0.5)
        
        traj = np.array(results_dict['hybrid']['trajectory'])
        ax3.plot(traj[:, 0], traj[:, 1], color=colors['hybrid'], linewidth=2, label='Hybrid SLAM')
    
    ax3.set_xlabel('X Position (meters)', fontsize=12)
    ax3.set_ylabel('Y Position (meters)', fontsize=12)
    ax3.set_title('Scan-to-Map Result (Hybrid)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    # performance metrics
    ax4 = plt.subplot(2, 3, 5)
    
    methods = list(results_dict.keys())
    final_drifts = []
    
    for method in methods:
        traj = np.array(results_dict[method]['trajectory'])
        final_drift = np.sqrt(
            (traj[-1, 0] - gt_array[len(traj)-1, 0])**2 + 
            (traj[-1, 1] - gt_array[len(traj)-1, 1])**2
        )
        final_drifts.append(final_drift)
    
    x_pos = np.arange(len(methods))
    bars = ax4.bar(x_pos, final_drifts, 
                   color=[colors[m] for m in methods],
                   alpha=0.8)
    
    # add value labels on bars
    for i, (bar, drift) in enumerate(zip(bars, final_drifts)):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{drift:.2f}m', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([m.replace('_', '\n').title() for m in methods])
    ax4.set_ylabel('Final Position Error (meters)', fontsize=12)
    ax4.set_title('Final Drift Comparison', fontsize=14, fontweight='bold')
    ax4.grid(True, axis='y', alpha=0.3)
    
    # processing time comparison
    ax5 = plt.subplot(2, 3, 6)
    
    proc_times = []
    for method in methods:
        proc_times.append(results_dict[method]['metrics']['avg_processing_time_ms'])
    
    bars = ax5.bar(x_pos, proc_times, 
                   color=[colors[m] for m in methods],
                   alpha=0.8)
    
    # add value labels
    for bar, time in zip(bars, proc_times):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{time:.1f}ms', ha='center', va='bottom', fontsize=11)
    
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels([m.replace('_', '\n').title() for m in methods])
    ax5.set_ylabel('Processing Time (ms/scan)', fontsize=12)
    ax5.set_title('Computational Performance', fontsize=14, fontweight='bold')
    ax5.grid(True, axis='y', alpha=0.3)
    
    # overall title
    plt.suptitle('ICP Performance Comparison', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('thesis_final_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('thesis_final_comparison.pdf', bbox_inches='tight')  # pdf for thesis
    plt.close()
    
    print("\nVisualization saved as:")
    print("  - thesis_final_comparison.png")
    print("  - thesis_final_comparison.pdf")