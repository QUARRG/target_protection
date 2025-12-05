
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings
from collections import defaultdict

# ROS2 imports
import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py


# Suppress the Axes3D import warning
warnings.filterwarnings('ignore', message='Unable to import Axes3D')

# Configuration
base_dir = Path('/home/paulo/Documents/bags_filter')
drones = ['C04', 'C05', 'C14']

models = ['modelA_0_8', 'modelA_0_6', 'modelA_0_4', 'modelA_0_2']
# models = ['modelC_0_8', 'modelC_0_6', 'modelC_0_4', 'modelC_0_2']

# # Model A - Filter GPS
# data_paths = {
#     'modelA_0_2': {
#         'baseline': 'baselines/modelA/rosbag2_2025_11_20-17_44_22',
#         'filtered': 'experiments_gps/modelA/rosbag2_2025_11_25-15_44_48'
#     },
#     'modelA_0_4': {
#         'baseline': 'baselines/modelA/rosbag2_2025_11_20-17_46_36',
#         'filtered': 'experiments_gps/modelA/rosbag2_2025_11_25-15_38_20'
#     },
#     'modelA_0_6': {
#         'baseline': 'baselines/modelA/rosbag2_2025_11_20-17_49_09',
#         'filtered': 'experiments_gps/modelA/rosbag2_2025_11_25-15_34_11'
#     },
#     'modelA_0_8': {
#         'baseline': 'baselines/modelA/rosbag2_2025_11_20-17_51_13',
#         'filtered': 'experiments_gps/modelA/rosbag2_2025_11_25-15_32_20'
#     }
# }


# Model A - Filter Relative
data_paths = {
    'modelA_0_2': {
        'baseline': 'baselines/modelA/rosbag2_2025_11_20-17_44_22',
        'filtered': 'relative/modelA_45/rosbag2_2025_12_01-10_22_32'
    },
    'modelA_0_4': {
        'baseline': 'baselines/modelA/rosbag2_2025_11_20-17_46_36',
        'filtered': 'relative/modelA_45/rosbag2_2025_12_01-10_20_47'
    },
    'modelA_0_6': {
        'baseline': 'baselines/modelA/rosbag2_2025_11_20-17_49_09',
        'filtered': 'relative/modelA_45/rosbag2_2025_12_01-10_18_47'
    },
    'modelA_0_8': {
        'baseline': 'baselines/modelA/rosbag2_2025_11_20-17_51_13',
        'filtered': 'relative/modelA_45/rosbag2_2025_12_01-10_16_36'
    }
}



# Model C - Filter Relative
# data_paths = {
#     'modelC_0_2': {
#         'baseline': 'baselines/modelC/rosbag2_2025_11_20-18_05_28',
#         'filtered': 'experiments_relative/modelC/rosbag2_2025_11_25-16_15_19'
#     },
#     'modelC_0_4': {
#         'baseline': 'baselines/modelC/rosbag2_2025_11_20-18_08_37',
#         'filtered': 'experiments_relative/modelC/rosbag2_2025_11_25-16_17_42'
#     },
#     'modelC_0_6': {
#         'baseline': 'baselines/modelC/rosbag2_2025_11_20-18_10_12',
#         'filtered': 'experiments_relative/modelC/rosbag2_2025_11_25-16_19_44'
#     },
#     'modelC_0_8': {
#         'baseline': 'baselines/modelC/rosbag2_2025_11_20-18_12_18',
#         'filtered': 'experiments_relative/modelC/rosbag2_2025_11_25-16_21_44'
#     }
# }


def read_bag(bag_path):
    """Read ROS2 bag and extract relevant data."""
    storage_options = rosbag2_py.StorageOptions(uri=str(bag_path), storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    
    topic_types = reader.get_all_topics_and_types()
    type_map = {topic.name: topic.type for topic in topic_types}
    
    data = defaultdict(list)
    
    while reader.has_next():
        (topic, raw_data, t) = reader.read_next()
        
        if topic not in type_map:
            continue
            
        msg_type = get_message(type_map[topic])
        msg = deserialize_message(raw_data, msg_type)
        
        # Store encircle flag
        if topic == '/encircle':
            data['/encircle'].append({'time': t, 'data': msg.data})
        
        # Store pose data for drones
        for drone in drones:
            # Baseline and filtered have /pose suffix
            for source in ['baseline', 'filtered']:
                pose_topic = f'/{drone}/{source}/pose'
                if topic == pose_topic:
                    data[pose_topic].append({
                        'time': t,
                        'stamp': msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec,
                        'x': msg.pose.position.x,
                        'y': msg.pose.position.y,
                        'z': msg.pose.position.z
                    })
                
                # Phase data
                phase_topic = f'/{drone}/{source}/phase'
                if topic == phase_topic:
                    data[phase_topic].append({
                        'time': t,
                        'data': msg.data
                    })
                
                # Phase difference data (leader and follower)
                for neighbor in ['leader', 'follower']:
                    phase_diff_topic = f'/{drone}/{source}/phase_diff/{neighbor}'
                    if topic == phase_diff_topic:
                        data[phase_diff_topic].append({
                            'time': t,
                            'data': msg.data
                        })
            
            # vicon_position and gps_position don't have /pose suffix
            for source in ['vicon_position', 'gps_position']:
                pose_topic = f'/{drone}/{source}'
                if topic == pose_topic:
                    data[pose_topic].append({
                        'time': t,
                        'stamp': msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec,
                        'x': msg.pose.position.x,
                        'y': msg.pose.position.y,
                        'z': msg.pose.position.z
                    })
    
    return data


def get_encircle_start_time(data, drone, source='baseline'):
    """
    Find the timestamp when /encircle becomes 1.
    Returns the closest drone pose timestamp.
    """
    encircle_data = data.get('/encircle', [])
    
    # Determine the correct topic name based on source
    if source in ['vicon_position', 'gps_position']:
        pose_topic = f'/{drone}/{source}'
    else:
        pose_topic = f'/{drone}/{source}/pose'
    
    pose_data = data.get(pose_topic, [])
    
    if not encircle_data or not pose_data:
        print(f"    Warning: Missing data - encircle: {len(encircle_data)}, {pose_topic}: {len(pose_data)}")
        return None
    
    # Find first time encircle becomes True/1
    encircle_start = None
    for entry in encircle_data:
        if entry['data']:
            encircle_start = entry['time']
            break
    
    if encircle_start is None:
        return None
    
    # Find closest pose timestamp
    min_diff = float('inf')
    closest_stamp = None
    for pose in pose_data:
        diff = abs(pose['time'] - encircle_start)
        if diff < min_diff:
            min_diff = diff
            closest_stamp = pose['stamp']
    
    return closest_stamp


def load_data(model):
    """Load baseline and filtered data for a given model."""
    baseline_path = base_dir / data_paths[model]['baseline']
    filtered_path = base_dir / data_paths[model]['filtered']
    
    baseline = read_bag(baseline_path)
    filtered = read_bag(filtered_path)
    
    return baseline, filtered


def get_position_data(data, drone, source='baseline', start_time=None, duration=80.0):
    """Extract x, y, z position data for a specific drone, optionally cropped from start_time."""
    # Determine the correct topic name based on source
    if source in ['vicon_position', 'gps_position']:
        pose_topic = f'/{drone}/{source}'
    else:
        pose_topic = f'/{drone}/{source}/pose'
    
    pose_data = data.get(pose_topic, [])
    
    if not pose_data:
        return {'time': np.array([]), 'x': np.array([]), 'y': np.array([]), 'z': np.array([])}
    
    # Filter data from start_time if provided
    if start_time is not None:
        # Crop to duration (in seconds) after start_time
        end_time = start_time + duration * 1e9  # Convert duration to nanoseconds
        pose_data = [p for p in pose_data if start_time <= p['stamp'] <= end_time]
    
    times = np.array([p['stamp'] for p in pose_data])
    x = np.array([p['x'] for p in pose_data])
    y = np.array([p['y'] for p in pose_data])
    z = np.array([p['z'] for p in pose_data])
    
    return {'time': times, 'x': x, 'y': y, 'z': z}


def get_phase_data(data, drone, source='baseline', start_time=None, duration=80.0):
    """Extract phase data for a specific drone."""
    phase_topic = f'/{drone}/{source}/phase'
    phase_data = data.get(phase_topic, [])
    
    if not phase_data:
        return {'time': np.array([]), 'phase': np.array([])}
    
    # Filter data from start_time if provided
    if start_time is not None:
        # Crop to duration (in seconds) after start_time
        end_time = start_time + duration * 1e9  # Convert duration to nanoseconds
        phase_data = [p for p in phase_data if start_time <= p['time'] <= end_time]
    
    times = np.array([p['time'] for p in phase_data])
    phase = np.array([p['data'] for p in phase_data])
    
    return {'time': times, 'phase': phase}


def compute_phase_diff(phase1_data, phase2_data):
    """
    Compute phase difference between two drones by interpolating to common timestamps.
    Returns the phase difference in degrees, wrapped to [-180, 180].
    Includes outlier filtering using median filtering.
    """
    if len(phase1_data['time']) == 0 or len(phase2_data['time']) == 0:
        return {'time': np.array([]), 'phase_diff': np.array([])}
    
    # Use the timestamps from phase1 as reference
    times = phase1_data['time']
    
    # Interpolate phase2 to match phase1 timestamps
    phase2_interp = np.interp(times, phase2_data['time'], phase2_data['phase'])
    
    # Compute phase difference (in radians)
    phase_diff_rad = phase1_data['phase'] - phase2_interp
    
    # Wrap phase difference to [-pi, pi] range
    phase_diff_rad = np.arctan2(np.sin(phase_diff_rad), np.cos(phase_diff_rad))
    
    # Convert to degrees (will be in [-180, 180] range)
    phase_diff_deg = np.degrees(np.abs(phase_diff_rad))
    
    # Apply median filter to remove outliers/spikes
    if len(phase_diff_deg) > 5:
        from scipy.signal import medfilt
        # Use kernel size of 5 (must be odd)
        phase_diff_filtered = medfilt(phase_diff_deg, kernel_size=5)
    else:
        phase_diff_filtered = phase_diff_deg
    
    # Additional outlier removal: remove points that differ too much from local median
    if len(phase_diff_filtered) > 10:
        window_size = 10
        phase_diff_cleaned = phase_diff_filtered.copy()
        
        for i in range(len(phase_diff_filtered)):
            # Define local window
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(phase_diff_filtered), i + window_size // 2 + 1)
            local_window = phase_diff_filtered[start_idx:end_idx]
            
            # Compute local statistics
            local_median = np.median(local_window)
            local_std = np.std(local_window)
            
            # Replace outliers (> 3 std from median) with median
            if abs(phase_diff_filtered[i] - local_median) > 3 * local_std:
                phase_diff_cleaned[i] = local_median
        
        phase_diff_final = phase_diff_cleaned
    else:
        phase_diff_final = phase_diff_filtered
    
    return {'time': times, 'phase_diff': phase_diff_final}


def compute_3d_rmse(baseline_data, filtered_data):
    """
    Compute RMSE of 3D position between filtered and baseline.
    Interpolates to common timestamps before computing RMSE.
    """
    if len(baseline_data['time']) == 0 or len(filtered_data['time']) == 0:
        return np.nan
    
    # Use baseline timestamps as reference
    times = baseline_data['time']
    
    # Interpolate filtered data to match baseline timestamps
    x_filtered_interp = np.interp(times, filtered_data['time'], filtered_data['x'])
    y_filtered_interp = np.interp(times, filtered_data['time'], filtered_data['y'])
    z_filtered_interp = np.interp(times, filtered_data['time'], filtered_data['z'])
    
    # Compute 3D Euclidean distance at each timestamp
    distances = np.sqrt(
        (baseline_data['x'] - x_filtered_interp)**2 +
        (baseline_data['y'] - y_filtered_interp)**2 +
        (baseline_data['z'] - z_filtered_interp)**2
    )
    
    # Compute RMSE
    rmse = np.sqrt(np.mean(distances**2))
    
    return rmse


def preprocess_all_data():
    """
    Preprocess all data once to avoid redundant computation.
    Returns a nested dictionary with all processed data.
    """
    processed_data = {}
    
    for model in models:
        print(f"Preprocessing {model}...")
        processed_data[model] = {}
        
        try:
            baseline, filtered = load_data(model)
            
            for drone in drones:
                print(f"  Preprocessing {drone}...")
                processed_data[model][drone] = {}
                
                try:
                    # Find when encircle starts for both baseline and filtered
                    baseline_start_time = get_encircle_start_time(baseline, drone, source='vicon_position')
                    filtered_start_time = get_encircle_start_time(filtered, drone, source='vicon_position')
                    
                    # Get position data cropped from encircle start
                    baseline_data = get_position_data(baseline, drone, 'vicon_position', start_time=baseline_start_time)
                    filtered_data = get_position_data(filtered, drone, 'vicon_position', start_time=filtered_start_time)
                    
                    # Get phase data
                    baseline_phase = get_phase_data(baseline, drone, 'baseline', start_time=baseline_start_time)
                    filtered_phase = get_phase_data(filtered, drone, 'filtered', start_time=filtered_start_time)
                    
                    # Compute RMSE
                    rmse = compute_3d_rmse(baseline_data, filtered_data)
                    
                    # Store all processed data
                    processed_data[model][drone] = {
                        'baseline_start_time': baseline_start_time,
                        'filtered_start_time': filtered_start_time,
                        'baseline_data': baseline_data,
                        'filtered_data': filtered_data,
                        'baseline_phase': baseline_phase,
                        'filtered_phase': filtered_phase,
                        'rmse': rmse,
                        'baseline_raw': baseline,
                        'filtered_raw': filtered
                    }
                    
                    print(f"    RMSE: {rmse:.4f} m")
                    
                except Exception as e:
                    print(f"    Error preprocessing {drone}: {e}")
                    processed_data[model][drone] = None
                    
        except Exception as e:
            print(f"  Error loading {model}: {e}")
    
    return processed_data


def create_montage(processed_data):
    """Create a 4x3 montage of figures showing baseline vs filtered data."""
    # Create figure with subplots: 4 rows (models) x 3 columns (drones)
    # Each subplot will have 3 axes stacked vertically for x, y, z
    fig = plt.figure(figsize=(20, 24))
    
    # Outer grid for models (rows) and drones (columns)
    outer_grid = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.25,
                                   left=0.08, right=0.95, top=0.95, bottom=0.05)
    
    for i, model in enumerate(models):
        print(f"Plotting {model}...")
        
        for j, drone in enumerate(drones):
            print(f"  Plotting {drone}...")
            
            # Create inner grid for x, y, z plots
            inner_grid = outer_grid[i, j].subgridspec(3, 1, hspace=0.15)
            
            # Get preprocessed data
            try:
                drone_data = processed_data[model][drone]
                if drone_data is None:
                    raise ValueError("No data available")
                
                baseline_start_time = drone_data['baseline_start_time']
                filtered_start_time = drone_data['filtered_start_time']
                baseline_data = drone_data['baseline_data']
                filtered_data = drone_data['filtered_data']
                rmse = drone_data['rmse']
                    
                # Normalize time to start from 0 (aligned at encircle start)
                # Both datasets start from their respective encircle events and are normalized to t=0
                if len(baseline_data['time']) > 0:
                    # Subtract the encircle start time to align at t=0
                    t_base = (baseline_data['time'] - baseline_start_time) / 1e9  # Convert to seconds
                else:
                    t_base = np.array([])
                
                if len(filtered_data['time']) > 0:
                    # Subtract the encircle start time to align at t=0
                    t_filt = (filtered_data['time'] - filtered_start_time) / 1e9  # Convert to seconds
                else:
                    t_filt = np.array([])
                
                print(f"    Baseline data points: {len(t_base)}, time range: [{t_base[0] if len(t_base) > 0 else 0:.2f}, {t_base[-1] if len(t_base) > 0 else 0:.2f}]")
                print(f"    Filtered data points: {len(t_filt)}, time range: [{t_filt[0] if len(t_filt) > 0 else 0:.2f}, {t_filt[-1] if len(t_filt) > 0 else 0:.2f}]")
                
                # Plot x, y, z
                axes_labels = ['x (m)', 'y (m)', 'z (m)']
                data_keys = ['x', 'y', 'z']
                
                for k, (label, key) in enumerate(zip(axes_labels, data_keys)):
                    ax = fig.add_subplot(inner_grid[k])
                    
                    # Plot baseline and filtered data
                    if len(t_base) > 0:
                        ax.plot(t_base, baseline_data[key], 'b-', label='Baseline', alpha=0.7, linewidth=1.5)
                    if len(t_filt) > 0:
                        ax.plot(t_filt, filtered_data[key], 'r-', label='Filtered', alpha=0.7, linewidth=1.5)
                    
                    ax.set_ylabel(label, fontsize=10)
                    ax.grid(True, alpha=0.3)
                    ax.tick_params(labelsize=9)
                    
                    # Add legend only to the first (top) plot
                    if k == 0:
                        ax.legend(loc='upper right', fontsize=9)
                        # Add title for the first plot in each column
                        if i == 0:
                            ax.set_title(drone, fontsize=12, fontweight='bold')
                    
                    # Add x-label only to the bottom plot
                    if k == 2:
                        ax.set_xlabel('Time (s)', fontsize=10)
                    else:
                        ax.set_xticklabels([])
                    
                    # Add model label on the left
                    if j == 0 and k == 1:  # Middle row of leftmost column
                        # Use reversed index to get correct omega label (0.2 at top, 0.8 at bottom)
                        label_model = models[len(models) - 1 - i]
                        omega_value = label_model.split('_')[-1].replace('_', '.')
                        ax.text(-0.25, 0.5, f'ω = 0.{omega_value}', 
                                transform=ax.transAxes,
                                fontsize=14, fontweight='bold',
                                va='center', ha='center', rotation=90)
                
            except Exception as e:
                print(f"    Error plotting {drone}: {e}")
                # Create empty subplot
                ax = fig.add_subplot(inner_grid[1])
                ax.text(0.5, 0.5, f'No data\n{drone}', 
                       transform=ax.transAxes,
                       ha='center', va='center', fontsize=10, color='red')
                ax.axis('off')
    
    # Add main title
    fig.suptitle('Baseline vs Filtered Comparison - Position Data (x, y, z)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Apply tight layout
    # plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save figure
    output_path = base_dir / 'montage_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nMontage saved to: {output_path}")
    
    # plt.show()


def create_3d_montage(processed_data):
    """Create a 4x3 montage of 3D trajectory plots."""
    # Create figure with subplots: 4 rows (models) x 3 columns (drones)
    fig = plt.figure(figsize=(20, 24))
    
    # Outer grid for models (rows) and drones (columns)
    outer_grid = fig.add_gridspec(4, 3, hspace=0.25, wspace=0.2,
                                   left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    for i, model in enumerate(models):
        print(f"Plotting 3D {model}...")
        
        for j, drone in enumerate(drones):
            print(f"  Plotting 3D {drone}...")
            
            # Create 3D subplot
            ax = fig.add_subplot(outer_grid[i, j], projection='3d')
            
            try:
                # Get preprocessed data
                drone_data = processed_data[model][drone]
                if drone_data is None:
                    raise ValueError("No data available")
                
                baseline_start_time = drone_data['baseline_start_time']
                filtered_start_time = drone_data['filtered_start_time']
                baseline_data = drone_data['baseline_data']
                filtered_data = drone_data['filtered_data']
                    
                # Normalize time to start from 0
                if len(baseline_data['time']) > 0:
                    t_base = (baseline_data['time'] - baseline_start_time) / 1e9
                    x_base = baseline_data['x']
                    y_base = baseline_data['y']
                    z_base = baseline_data['z']
                else:
                    x_base = y_base = z_base = t_base = np.array([])
                
                if len(filtered_data['time']) > 0:
                    t_filt = (filtered_data['time'] - filtered_start_time) / 1e9
                    x_filt = filtered_data['x']
                    y_filt = filtered_data['y']
                    z_filt = filtered_data['z']
                else:
                    x_filt = y_filt = z_filt = t_filt = np.array([])
                
                # Create common time reference by interpolating both to a unified time grid
                # This ensures both trajectories grow at the same rate
                if len(t_base) > 0 and len(t_filt) > 0:
                    # Use the longer time series as reference
                    if len(t_base) >= len(t_filt):
                        t_common = t_base
                        x_base_interp = x_base
                        y_base_interp = y_base
                        z_base_interp = z_base
                        x_filt_interp = np.interp(t_common, t_filt, x_filt)
                        y_filt_interp = np.interp(t_common, t_filt, y_filt)
                        z_filt_interp = np.interp(t_common, t_filt, z_filt)
                    else:
                        t_common = t_filt
                        x_filt_interp = x_filt
                        y_filt_interp = y_filt
                        z_filt_interp = z_filt
                        x_base_interp = np.interp(t_common, t_base, x_base)
                        y_base_interp = np.interp(t_common, t_base, y_base)
                        z_base_interp = np.interp(t_common, t_base, z_base)
                    
                    # Add progressive z-offset to create spiral effect (0.05m per second)
                    z_base_spiral = z_base_interp + t_common * 0.1
                    z_filt_spiral = z_filt_interp + t_common * 0.1
                elif len(t_base) > 0:
                    t_common = t_base
                    x_base_interp = x_base
                    y_base_interp = y_base
                    z_base_spiral = z_base + t_common * 0.1
                    x_filt_interp = y_filt_interp = z_filt_spiral = np.array([])
                elif len(t_filt) > 0:
                    t_common = t_filt
                    x_filt_interp = x_filt
                    y_filt_interp = y_filt
                    z_filt_spiral = z_filt + t_common * 0.1
                    x_base_interp = y_base_interp = z_base_spiral = np.array([])
                else:
                    x_base_interp = y_base_interp = z_base_spiral = np.array([])
                    x_filt_interp = y_filt_interp = z_filt_spiral = np.array([])
                
                # Plot 3D trajectories with spiral effect
                if len(x_base_interp) > 0:
                    ax.plot(x_base_interp, y_base_interp, z_base_spiral, 'b-', label='Baseline', alpha=0.7, linewidth=2)
                    # Mark start point
                    ax.scatter(x_base_interp[0], y_base_interp[0], z_base_spiral[0], c='blue', marker='o', s=100, alpha=0.8)
                
                if len(x_filt_interp) > 0:
                    ax.plot(x_filt_interp, y_filt_interp, z_filt_spiral, 'r-', label='Filtered', alpha=0.7, linewidth=2)
                    # Mark start point
                    ax.scatter(x_filt_interp[0], y_filt_interp[0], z_filt_spiral[0], c='red', marker='o', s=100, alpha=0.8)
                
                # Set labels
                ax.set_xlabel('X (m)', fontsize=10)
                ax.set_ylabel('Y (m)', fontsize=10)
                ax.set_zlabel('Z (m)', fontsize=10)
                
                # Set title for top row
                if i == 0:
                    ax.set_title(drone, fontsize=12, fontweight='bold')
                
                # Add legend
                ax.legend(loc='upper right', fontsize=9)
                
                # Set viewing angle and orthographic projection
                ax.view_init(elev=20, azim=45)
                ax.set_proj_type('ortho')
                
                # Set equal aspect ratio for better visualization
                if len(x_base_interp) > 0 or len(x_filt_interp) > 0:
                    all_x = np.concatenate([x_base_interp, x_filt_interp]) if len(x_base_interp) > 0 and len(x_filt_interp) > 0 else (x_base_interp if len(x_base_interp) > 0 else x_filt_interp)
                    all_y = np.concatenate([y_base_interp, y_filt_interp]) if len(y_base_interp) > 0 and len(y_filt_interp) > 0 else (y_base_interp if len(y_base_interp) > 0 else y_filt_interp)
                    all_z = np.concatenate([z_base_spiral, z_filt_spiral]) if len(x_base_interp) > 0 and len(x_filt_interp) > 0 else (z_base_spiral if len(x_base_interp) > 0 else z_filt_spiral)
                    
                    max_range = np.array([all_x.max()-all_x.min(), 
                                            all_y.max()-all_y.min(), 
                                            all_z.max()-all_z.min()]).max() / 2.0
                    
                    mid_x = (all_x.max()+all_x.min()) * 0.5
                    mid_y = (all_y.max()+all_y.min()) * 0.5
                    mid_z = (all_z.max()+all_z.min()) * 0.5
                    
                    ax.set_xlim(mid_x - max_range, mid_x + max_range)
                    ax.set_ylim(mid_y - max_range, mid_y + max_range)
                    ax.set_zlim(mid_z - max_range, mid_z + max_range)
                    
                # Add model label on the left side
                if j == 0:
                    # Use reversed index to get correct omega label (0.2 at top, 0.8 at bottom)
                    label_model = models[len(models) - 1 - i]
                    omega_value = label_model.split('_')[-1].replace('_', '.')
                    ax.text2D(-0.15, 0.5, f'ω = 0.{omega_value}', 
                             transform=ax.transAxes,
                             fontsize=14, fontweight='bold',
                             va='center', ha='center', rotation=90)
                
                print(f"    3D plot created with {len(x_base_interp)} baseline and {len(x_filt_interp)} filtered points")
            
            except Exception as e:
                print(f"    Error plotting 3D {drone}: {e}")
                import traceback
                traceback.print_exc()
                ax.text2D(0.5, 0.5, f'No data\n{drone}', 
                         transform=ax.transAxes,
                         ha='center', va='center', fontsize=10, color='red')
    
    # Add main title
    fig.suptitle('3D Trajectory Comparison - Baseline vs Filtered', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Apply tight layout
    # plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save figure
    output_path = base_dir / 'montage_3d_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n3D Montage saved to: {output_path}")
    
    plt.close()


def create_phase_montage(processed_data):
    """Create a 4x3 montage of phase difference plots."""
    # Create figure with subplots: 4 rows (models) x 3 columns (drones)
    # Each subplot will have 2 axes stacked vertically for leader and follower phase diff
    fig = plt.figure(figsize=(20, 24))
    
    # Outer grid for models (rows) and drones (columns)
    outer_grid = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.25,
                                   left=0.08, right=0.95, top=0.95, bottom=0.05)
    
    for i, model in enumerate(models):
        print(f"Plotting phase {model}...")
        
        for j, drone in enumerate(drones):
            print(f"  Plotting phase {drone}...")
            
            # Create inner grid for leader and follower phase diff plots
            inner_grid = outer_grid[i, j].subgridspec(2, 1, hspace=0.15)
            
            try:
                # Get preprocessed data
                drone_data = processed_data[model][drone]
                if drone_data is None:
                    raise ValueError("No data available")
                
                baseline_start_time = drone_data['baseline_start_time']
                filtered_start_time = drone_data['filtered_start_time']
                baseline_raw = drone_data['baseline_raw']
                filtered_raw = drone_data['filtered_raw']
                    
                # Define neighbor relationships
                # C04: follower=C05, leader=C14
                # C05: follower=C14, leader=C04
                # C14: follower=C04, leader=C05
                neighbor_map = {
                    'C04': {'leader': 'C14', 'follower': 'C05'},
                    'C05': {'leader': 'C04', 'follower': 'C14'},
                    'C14': {'leader': 'C05', 'follower': 'C04'}
                }
                
                # Plot for leader and follower
                neighbors = ['leader', 'follower']
                
                for k, neighbor in enumerate(neighbors):
                    ax = fig.add_subplot(inner_grid[k])
                    
                    # Get the actual drone name for this relationship
                    neighbor_drone = neighbor_map[drone][neighbor]
                    label = f'Phase Diff to {neighbor.capitalize()} ({neighbor_drone})'
                    
                    # Get individual phase data for ego drone and neighbor drone
                    # Baseline
                    ego_phase_base = get_phase_data(baseline_raw, drone, 'baseline', start_time=baseline_start_time)
                    neighbor_phase_base = get_phase_data(baseline_raw, neighbor_drone, 'baseline', start_time=baseline_start_time)
                    
                    # Filtered
                    ego_phase_filt = get_phase_data(filtered_raw, drone, 'filtered', start_time=filtered_start_time)
                    neighbor_phase_filt = get_phase_data(filtered_raw, neighbor_drone, 'filtered', start_time=filtered_start_time)
                    
                    # Compute absolute phase difference directly
                    # Baseline
                    if len(ego_phase_base['time']) > 0 and len(neighbor_phase_base['time']) > 0:
                        # Interpolate neighbor phase to match ego timestamps
                        neighbor_phase_interp = np.interp(ego_phase_base['time'], neighbor_phase_base['time'], neighbor_phase_base['phase'])
                        # Compute absolute phase difference in radians, then convert to degrees
                        phase_diff_rad = ego_phase_base['phase'] - neighbor_phase_interp
                        phase_diff_rad = np.arctan2(np.sin(phase_diff_rad), np.cos(phase_diff_rad))
                        baseline_phase_abs = np.degrees(np.abs(phase_diff_rad))
                        t_base = (ego_phase_base['time'] - baseline_start_time) / 1e9
                    else:
                        baseline_phase_abs = np.array([])
                        t_base = np.array([])
                    
                    # Filtered
                    if len(ego_phase_filt['time']) > 0 and len(neighbor_phase_filt['time']) > 0:
                        # Interpolate neighbor phase to match ego timestamps
                        neighbor_phase_interp = np.interp(ego_phase_filt['time'], neighbor_phase_filt['time'], neighbor_phase_filt['phase'])
                        # Compute absolute phase difference in radians, then convert to degrees
                        phase_diff_rad = ego_phase_filt['phase'] - neighbor_phase_interp
                        phase_diff_rad = np.arctan2(np.sin(phase_diff_rad), np.cos(phase_diff_rad))
                        filtered_phase_abs = np.degrees(np.abs(phase_diff_rad))
                        t_filt = (ego_phase_filt['time'] - filtered_start_time) / 1e9
                    else:
                        filtered_phase_abs = np.array([])
                        t_filt = np.array([])
                    
                    # Plot absolute phase difference data
                    if len(t_base) > 0:
                        ax.plot(t_base, baseline_phase_abs, 'b-', 
                                label='Baseline', alpha=0.7, linewidth=1.5)
                    if len(t_filt) > 0:
                        ax.plot(t_filt, filtered_phase_abs, 'r-', 
                                label='Filtered', alpha=0.7, linewidth=1.5)
                    
                    # Add theoretical value lines
                    # Leader is ahead: ego - leader = -120°
                    # Follower is behind: ego - follower = +120°
                    # if neighbor == 'leader':
                    #     ax.axhline(y=120, color='g', linestyle='--', linewidth=2, 
                    #               label='Theoretical (-120°)', alpha=0.7)
                    # else:  # follower
                    ax.axhline(y=120, color='g', linestyle='--', linewidth=2, 
                                label='Theoretical', alpha=0.7)
                    
                    ax.set_ylabel('Phase Difference (deg)', fontsize=10)
                    ax.grid(True, alpha=0.3)
                    ax.tick_params(labelsize=9)
                    
                    # Set y-axis limits to show the full range [-180, 180]
                    ax.set_ylim(100, 140)
                    
                    # Add legend only to the first (top) plot
                    if k == 0:
                        ax.legend(loc='upper right', fontsize=9)
                        # Add title for the first plot in each column
                        if i == 0:
                            ax.set_title(drone, fontsize=12, fontweight='bold')
                    
                    # Add subplot label showing which drone is the neighbor
                    ax.text(0.02, 0.95, label, transform=ax.transAxes,
                            fontsize=9, va='top', ha='left',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    
                    # Add x-label only to the bottom plot
                    if k == 1:
                        ax.set_xlabel('Time (s)', fontsize=10)
                    else:
                        ax.set_xticklabels([])
                        
                    # Add model label on the left
                    if j == 0 and k == 0:  # Top row of leftmost column
                        # Use reversed index to get correct omega label (0.2 at top, 0.8 at bottom)
                        label_model = models[len(models) - 1 - i]
                        omega_value = label_model.split('_')[-1].replace('_', '.')
                        ax.text(-0.25, 0.5, f'ω = 0.{omega_value}', 
                               transform=ax.transAxes,
                               fontsize=14, fontweight='bold',
                               va='center', ha='center', rotation=90)
            
            except Exception as e:
                print(f"    Error plotting phase {drone}: {e}")
                import traceback
                traceback.print_exc()
                # Create empty subplot
                ax = fig.add_subplot(inner_grid[0])
                ax.text(0.5, 0.5, f'No data\n{drone}', 
                       transform=ax.transAxes,
                       ha='center', va='center', fontsize=10, color='red')
                ax.axis('off')
    
    # Add main title
    fig.suptitle('Phase Difference Comparison - Baseline vs Filtered (Theoretical: 120°)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Apply tight layout
    # plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save figure
    output_path = base_dir / 'montage_phase_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPhase Montage saved to: {output_path}")
    
    plt.close()


if __name__ == '__main__':
    # Preprocess all data once
    print("=" * 80)
    print("PREPROCESSING ALL DATA")
    print("=" * 80)
    processed_data = preprocess_all_data()
    
    print("\n" + "=" * 80)
    print("CREATING PLOTS")
    print("=" * 80)
    
    # Create all plots using preprocessed data
    create_montage(processed_data)
    create_3d_montage(processed_data)
    create_phase_montage(processed_data)

