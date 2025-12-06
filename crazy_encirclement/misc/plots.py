import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Configuration
base_dir = Path('/home/paulo/Documents/DATA')
groups = ['baseline', 'gps', 'relative']
models = ['modelA', 'modelC']
speeds = ['0_2', '0_4', '0_6', '0_8']
drones = ['C04', 'C05', 'C14']

# Duration to crop after encircle flag (in seconds)
CROP_DURATION = 60.0

# Smoothing configuration
ENABLE_SMOOTHING = False  # Enable Savitzky-Golay smoothing
SMOOTHING_WINDOW = 11  # Window size for Savitzky-Golay filter (must be odd)
SMOOTHING_METHOD = 'savgol'  # Use Savitzky-Golay filter to preserve dynamics
ENABLE_OUTLIER_REMOVAL = True  # Enable outlier detection and interpolation
OUTLIER_THRESHOLD = 1.5  # Threshold for outlier detection (std from median, lower = more aggressive)
OUTLIER_WINDOW = 11  # Window size for outlier detection

# Drone relationships: ego -> (follower, leader)
DRONE_RELATIONSHIPS = {
    'C05': ('C14', 'C04'),
    'C04': ('C05', 'C14'),
    'C14': ('C04', 'C05')
}


def find_csv_files(group, model, speed):
    """
    Find CSV files for a given group, model, and speed.
    For baseline: expects 1 CSV file
    For gps/relative: expects 5 CSV files (with seed numbers)
    """
    search_path = base_dir / group / model / speed
    
    if not search_path.exists():
        print(f"Warning: Path does not exist: {search_path}")
        return []
    
    csv_files = list(search_path.glob('*.csv'))
    
    if len(csv_files) == 0:
        print(f"Warning: No CSV files found in {search_path}")
    
    return sorted(csv_files)


def load_and_crop_csv(csv_path, crop_duration=CROP_DURATION):
    """
    Load a CSV file and crop data from when _encircle becomes True.
    Returns the cropped dataframe.
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Check if _encircle_data column exists
        if '_encircle_data' not in df.columns:
            print(f"Warning: '_encircle_data' column not found in {csv_path}")
            return None
        
        # Find the first index where _encircle_data becomes True
        encircle_idx = df[df['_encircle_data'] == True].index
        
        if len(encircle_idx) == 0:
            print(f"Warning: _encircle_data never becomes True in {csv_path}")
            return None
        
        start_idx = encircle_idx[0]
        
        # Assume there's a timestamp column (common names)
        timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'stamp' in col.lower()]
        
        if len(timestamp_cols) == 0:
            print(f"Warning: No timestamp column found in {csv_path}")
            # If no timestamp, crop by row count (assuming constant rate)
            # This is a fallback - adjust as needed
            return df.iloc[start_idx:start_idx + int(crop_duration * 100)]  # Assuming ~100Hz
        
        # Use the first timestamp column found
        time_col = timestamp_cols[0]
        start_time = df.loc[start_idx, time_col]
        end_time = start_time + crop_duration
        
        # Crop the dataframe
        cropped_df = df[(df[time_col] >= start_time) & (df[time_col] <= end_time)].copy()
        
        # Reset time to start from 0
        cropped_df[time_col] = cropped_df[time_col] - start_time
        
        return cropped_df
    
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None


def get_phase_column(df, drone, source='filtered'):
    """
    Find the phase column for a specific drone and source.
    Expected format: something like 'C04_filtered_phase' or similar
    """
    phase_cols = [col for col in df.columns if drone in col and 'phase' in col.lower() and source in col.lower()]
    
    if len(phase_cols) == 0:
        # Try without source keyword
        phase_cols = [col for col in df.columns if drone in col and 'phase' in col.lower()]
    
    if len(phase_cols) > 0:
        return phase_cols[0]   
    
    return None


def compute_phase_diff_unit_vector(phase_ego, phase_other):
    """
    Compute phase difference using unit vector dot product.
    Returns absolute phase difference in degrees.
    
    Args:
        phase_ego: phase of ego drone (radians)
        phase_other: phase of other drone (radians)
    
    Returns:
        Absolute phase difference in degrees
    """
    unit_ego = np.array([np.cos(phase_ego), np.sin(phase_ego)])
    unit_other = np.array([np.cos(phase_other), np.sin(phase_other)])
    
    # Compute dot product and clip to [-1, 1] to avoid numerical issues with arccos
    dot_product = np.dot(unit_ego, unit_other)
    
    # Phase difference in radians, then convert to degrees
    phi_diff_rad = np.arccos(dot_product)
    phi_diff_deg = np.degrees(phi_diff_rad)
    
    return phi_diff_deg


def smooth_signal(signal, window_size=11, method='savgol'):
    """
    Smooth a signal using Savitzky-Golay filter to preserve dynamics.
    
    Args:
        signal: 1D numpy array to smooth
        window_size: Size of the smoothing window (must be odd)
        method: 'savgol', 'median', or 'moving_average'
    
    Returns:
        Smoothed signal
    """
    if len(signal) < 5:
        return signal
    
    if method == 'savgol':
        from scipy.signal import savgol_filter
        # Ensure window size is odd and valid
        if window_size % 2 == 0:
            window_size += 1
        window_size = min(window_size, len(signal) if len(signal) % 2 == 1 else len(signal) - 1)
        if window_size < 5:
            window_size = 5
        # Use polynomial order 3 to preserve shape
        polyorder = min(3, window_size - 2)
        smoothed = savgol_filter(signal, window_length=window_size, polyorder=polyorder)
    elif method == 'median':
        from scipy.signal import medfilt
        # Ensure window size is odd
        if window_size % 2 == 0:
            window_size += 1
        smoothed = medfilt(signal, kernel_size=window_size)
    elif method == 'moving_average':
        # Simple moving average
        smoothed = np.convolve(signal, np.ones(window_size)/window_size, mode='same')
    else:
        smoothed = signal
    
    return smoothed


def remove_outliers(signal, threshold=2.5, window_size=15):
    """
    Find outliers, remove them, and replace with interpolated values.
    
    Args:
        signal: 1D numpy array
        threshold: Number of std for outlier detection (lower = more aggressive)
        window_size: Size of the local window for outlier detection
    
    Returns:
        Signal with outliers replaced by interpolation
    """
    if len(signal) < 3:
        return signal
    
    cleaned = signal.copy()
    outlier_indices = []
    
    # Step 1: Identify all outlier indices
    for i in range(len(signal)):
        # Define local window
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(signal), i + window_size // 2 + 1)
        local_window = signal[start_idx:end_idx]
        
        # Compute local statistics
        local_median = np.median(local_window)
        local_std = np.std(local_window)
        
        # Mark as outlier if deviation is too large
        if local_std > 0 and abs(signal[i] - local_median) > threshold * local_std:
            outlier_indices.append(i)
    
    # Step 2: Replace outliers with interpolated values
    if len(outlier_indices) > 0:
        # Create array of good (non-outlier) indices
        good_indices = np.array([i for i in range(len(signal)) if i not in outlier_indices])
        
        if len(good_indices) >= 2:
            # Get good values
            good_values = signal[good_indices]
            
            # Interpolate outlier values from good values
            outlier_indices_array = np.array(outlier_indices)
            interpolated_values = np.interp(outlier_indices_array, good_indices, good_values)
            
            # Replace outliers with interpolated values
            cleaned[outlier_indices_array] = interpolated_values
    
    return cleaned


def plot_phase_differences():
    """
    Create a 4x3 montage showing phase differences for GPS group.
    Rows: speeds (0.2, 0.4, 0.6, 0.8)
    Columns: drones (C04, C05, C14)
    
    For each drone:
    - Red lines: phase difference to follower (5 CSV files)
    - Blue lines: phase difference to leader (5 CSV files)
    - Green dashed line: theoretical 120° separation
    """
    fig, axes = plt.subplots(4, 3, figsize=(18, 16), sharex=True, sharey=True)
    fig.suptitle('Phase Differences - GPS Group (ModelA)', fontsize=16, fontweight='bold')
    
    # Process GPS group with modelA
    group = 'gps'
    model = 'modelA'
    source = 'filtered'  # or 'baseline' depending on your data
    
    for i, speed in enumerate(speeds):
        for j, drone in enumerate(drones):
            ax = axes[i, j]
            
            try:
                # Get follower and leader for this ego drone
                follower, leader = DRONE_RELATIONSHIPS[drone]
                
                # Find CSV files (should be 5 for GPS group)
                csv_files = find_csv_files(group, model, speed)
                
                if len(csv_files) == 0:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue
                
                print(f"Processing speed={speed}, drone={drone}: found {len(csv_files)} CSV files")
                
                # Plot theoretical line at 120°
                ax.axhline(y=120, color='k', linestyle='--', linewidth=2, 
                          label='Theoretical (120°)', alpha=0.7, zorder=1)
                
                # Process each CSV file
                for csv_idx, csv_path in enumerate(csv_files):
                    df = load_and_crop_csv(csv_path)
                    
                    if df is None or len(df) == 0:
                        print(f"  Skipping {csv_path.name}: no valid data")
                        continue
                    
                    # Get phase columns for ego, follower, and leader
                    ego_phase_col = get_phase_column(df, drone, source)
                    follower_phase_col = get_phase_column(df, follower, source)
                    leader_phase_col = get_phase_column(df, leader, source)
                    
                    if ego_phase_col is None or follower_phase_col is None or leader_phase_col is None:
                        print(f"  Skipping {csv_path.name}: missing phase columns")
                        continue
                    
                    # Get time column
                    timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'stamp' in col.lower()]
                    time_col = timestamp_cols[0] if len(timestamp_cols) > 0 else None
                                       
                    # Get valid data for each drone separately (data may not be synced)
                    ego_valid = df[[time_col, ego_phase_col]].dropna()
                    follower_valid = df[[time_col, follower_phase_col]].dropna()
                    leader_valid = df[[time_col, leader_phase_col]].dropna()
                    
                    if len(ego_valid) == 0 or len(follower_valid) == 0 or len(leader_valid) == 0:
                        print(f"  Skipping {csv_path.name}: insufficient valid data")
                        continue

                    total_length = min(len(ego_valid), len(follower_valid), len(leader_valid))
                    
                    # Extract time and phase for each drone
                    time_ego = ego_valid[time_col].values
                    time_ego = time_ego[:total_length]
                    phase_ego = ego_valid[ego_phase_col].values
                    phase_ego = phase_ego[:total_length]
                    
                    time_follower = follower_valid[time_col].values
                    time_follower = time_follower[:total_length]
                    phase_follower = follower_valid[follower_phase_col].values
                    phase_follower = phase_follower[:total_length]
                    
                    time_leader = leader_valid[time_col].values
                    time_leader = time_leader[:total_length]
                    phase_leader = leader_valid[leader_phase_col].values
                    phase_leader = phase_leader[:total_length]

                    # Wrap phases to 0-2pi
                    # phase_ego = np.mod(phase_ego, 2 * np.pi)
                    # phase_follower = np.mod(phase_follower, 2 * np.pi)
                    # phase_leader = np.mod(phase_leader, 2 * np.pi)
                    
                    # Unwrap phases to handle discontinuities (assumes phases in degrees)
                    # phase_ego = np.unwrap(phase_ego, period=360)
                    # phase_follower = np.unwrap(phase_follower, period=360)
                    # phase_leader = np.unwrap(phase_leader, period=360)
                    
                    # # Remove outliers and smooth raw phase signals if enabled
                    # if ENABLE_OUTLIER_REMOVAL:
                    #     phase_ego = remove_outliers(phase_ego, threshold=OUTLIER_THRESHOLD, window_size=OUTLIER_WINDOW)
                    #     phase_follower = remove_outliers(phase_follower, threshold=OUTLIER_THRESHOLD, window_size=OUTLIER_WINDOW)
                    #     phase_leader = remove_outliers(phase_leader, threshold=OUTLIER_THRESHOLD, window_size=OUTLIER_WINDOW)
                    
                    # if ENABLE_SMOOTHING:
                    #     phase_ego = smooth_signal(phase_ego, window_size=SMOOTHING_WINDOW, method=SMOOTHING_METHOD)
                    #     phase_follower = smooth_signal(phase_follower, window_size=SMOOTHING_WINDOW, method=SMOOTHING_METHOD)
                    #     phase_leader = smooth_signal(phase_leader, window_size=SMOOTHING_WINDOW, method=SMOOTHING_METHOD)
                    
                    # # Use ego drone's timeline as reference
                    time_reference = time_ego
                    
                    # # Interpolate follower and leader phases to match ego timeline
                    # phase_follower_interp = np.interp(time_reference, time_follower, phase_follower)
                    # phase_leader_interp = np.interp(time_reference, time_leader, phase_leader)

                    # fig, ax = plt.subplots()
                    # ax.plot(time_reference, np.degrees(phase_ego), 'k-', label='Ego Phase', alpha=0.7)
                    # ax.plot(time_reference, np.degrees(phase_follower_interp), 'r-', label='Follower Phase', alpha=0.7)
                    # ax.plot(time_reference, np.degrees(phase_leader_interp), 'b-', label='Leader Phase', alpha=0.7)
                    # plt.legend()
                    # plt.savefig('/home/paulo/ros_ws/src/crazy_encirclement/crazy_encirclement/misc/tmp/debug_phases.png', dpi=150)
                    # # plt.close(fig)
                    

                    # print(f"    Using {total_length} samples from {csv_path.name}")
                    # print(f"lenghts: ego={len(phase_ego)}, follower={len(phase_follower)}, leader={len(phase_leader)}")
                    
                    # Compute phase differences using unit vector method (from smoothed phases)
                    phi_diff_follower = np.array([
                        compute_phase_diff_unit_vector(phase_ego[k], phase_follower[k])
                        for k in range(len(phase_ego))
                    ])
                    
                    phi_diff_leader = np.array([
                        compute_phase_diff_unit_vector(phase_ego[k], phase_leader[k])
                        for k in range(len(phase_ego))
                    ])
                    
                    # Plot phase differences
                    # Red for follower, blue for leader
                    alpha = 0.5 if len(csv_files) > 1 else 0.7
                    label_follower = f'to Follower ({follower})' if csv_idx == 0 else None
                    label_leader = f'to Leader ({leader})' if csv_idx == 0 else None
                    
                    ax.plot(time_reference, phi_diff_follower, 'r-', linewidth=1.0, 
                           alpha=alpha, label=label_follower, zorder=2)
                    ax.plot(time_reference, phi_diff_leader, 'b-', linewidth=1.0, 
                           alpha=alpha, label=label_leader, zorder=2)
                
                # Configure plot
                ax.grid(True, alpha=0.3)
                ax.set_ylim(75, 165)
                
                # Labels
                if i == 0:
                    ax.set_title(f'{drone}', fontsize=12, fontweight='bold')
                if j == 0:
                    omega_value = speed.split('_')[1]
                    ax.set_ylabel(f'ω = 0.{omega_value}\nPhase Diff (deg)', fontsize=10)
                else:
                    ax.set_ylabel('Phase Diff (deg)', fontsize=10)
                
                if i == len(speeds) - 1:
                    ax.set_xlabel('Time (s)', fontsize=10)
                
                ax.tick_params(labelsize=9)
                
                # Add legend (only for first subplot)
                if i == 0 and j == 0:
                    ax.legend(loc='upper right', fontsize=8)
                
            except Exception as e:
                print(f"Error plotting speed={speed}, drone={drone}: {e}")
                import traceback
                traceback.print_exc()
                ax.text(0.5, 0.5, f'Error:\n{str(e)[:50]}', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=8, color='red')
    
    plt.tight_layout()
    
    # Save figure
    output_path = base_dir / 'phase_differences_gps_montage.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    plt.close()


def plot_phase_errors():
    """
    Create a 4x3 montage showing phase errors (difference from theoretical 120°) for GPS group.
    Rows: speeds (0.2, 0.4, 0.6, 0.8)
    Columns: drones (C04, C05, C14)
    
    For each drone:
    - Red line/envelope: mean error to follower ± 3σ (5 CSV files)
    - Blue line/envelope: mean error to leader ± 3σ (5 CSV files)
    - Green dashed line: zero error (theoretical 120° separation)
    """
    fig, axes = plt.subplots(4, 3, figsize=(18, 16), sharex=True, sharey=True)
    fig.suptitle('Phase Errors (from Theoretical 120°) - GPS Group (ModelA)', fontsize=16, fontweight='bold')
    
    # Process GPS group with modelA
    group = 'gps'
    model = 'modelA'
    source = 'filtered'
    theoretical_phase = 120.0  # degrees
    
    for i, speed in enumerate(speeds):
        for j, drone in enumerate(drones):
            ax = axes[i, j]
            
            try:
                # Get follower and leader for this ego drone
                follower, leader = DRONE_RELATIONSHIPS[drone]
                
                # Find CSV files (should be 5 for GPS group)
                csv_files = find_csv_files(group, model, speed)
                
                if len(csv_files) == 0:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue
                
                print(f"Processing errors speed={speed}, drone={drone}: found {len(csv_files)} CSV files")
                
                # Plot zero error line (theoretical)
                ax.axhline(y=0, color='k', linestyle='--', linewidth=3, 
                          label='Zero Error (120°)', alpha=0.7, zorder=1)
                
                # Collect all phase errors from CSV files
                errors_follower_all = []
                errors_leader_all = []
                time_references = []
                
                # Process each CSV file
                for csv_idx, csv_path in enumerate(csv_files):
                    df = load_and_crop_csv(csv_path)
                    
                    if df is None or len(df) == 0:
                        print(f"  Skipping {csv_path.name}: no valid data")
                        continue
                    
                    # Get phase columns for ego, follower, and leader
                    ego_phase_col = get_phase_column(df, drone, source)
                    follower_phase_col = get_phase_column(df, follower, source)
                    leader_phase_col = get_phase_column(df, leader, source)
                    
                    if ego_phase_col is None or follower_phase_col is None or leader_phase_col is None:
                        print(f"  Skipping {csv_path.name}: missing phase columns")
                        continue
                    
                    # Get time column
                    timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'stamp' in col.lower()]
                    time_col = timestamp_cols[0] if len(timestamp_cols) > 0 else None
                    
                    # Get valid data for each drone separately (data may not be synced)
                    ego_valid = df[[time_col, ego_phase_col]].dropna()
                    follower_valid = df[[time_col, follower_phase_col]].dropna()
                    leader_valid = df[[time_col, leader_phase_col]].dropna()
                    
                    if len(ego_valid) == 0 or len(follower_valid) == 0 or len(leader_valid) == 0:
                        print(f"  Skipping {csv_path.name}: insufficient valid data")
                        continue

                    total_length = min(len(ego_valid), len(follower_valid), len(leader_valid))
                    
                    # Extract time and phase for each drone
                    time_ego = ego_valid[time_col].values[:total_length]
                    phase_ego = ego_valid[ego_phase_col].values[:total_length]
                    
                    time_follower = follower_valid[time_col].values[:total_length]
                    phase_follower = follower_valid[follower_phase_col].values[:total_length]
                    
                    time_leader = leader_valid[time_col].values[:total_length]
                    phase_leader = leader_valid[leader_phase_col].values[:total_length]
                    
                    # Unwrap phases to handle discontinuities (assumes phases in degrees)
                    # phase_ego = np.unwrap(phase_ego, period=360)
                    # phase_follower = np.unwrap(phase_follower, period=360)
                    # phase_leader = np.unwrap(phase_leader, period=360)
                    
                    # # Remove outliers and smooth raw phase signals if enabled
                    # if ENABLE_OUTLIER_REMOVAL:
                    #     phase_ego = remove_outliers(phase_ego, threshold=OUTLIER_THRESHOLD, window_size=OUTLIER_WINDOW)
                    #     phase_follower = remove_outliers(phase_follower, threshold=OUTLIER_THRESHOLD, window_size=OUTLIER_WINDOW)
                    #     phase_leader = remove_outliers(phase_leader, threshold=OUTLIER_THRESHOLD, window_size=OUTLIER_WINDOW)
                    
                    # if ENABLE_SMOOTHING:
                    # phase_ego = smooth_signal(phase_ego, window_size=SMOOTHING_WINDOW, method=SMOOTHING_METHOD)
                    # phase_follower = smooth_signal(phase_follower, window_size=SMOOTHING_WINDOW, method=SMOOTHING_METHOD)
                    # phase_leader = smooth_signal(phase_leader, window_size=SMOOTHING_WINDOW, method=SMOOTHING_METHOD)
                    
                    # phase_follower_interp = np.interp(time_ego, time_follower, phase_follower)
                    # phase_leader_interp = np.interp(time_ego, time_leader, phase_leader)
                    
                    # Compute phase differences using unit vector method (from smoothed phases)
                    phi_diff_follower = np.array([
                        compute_phase_diff_unit_vector(phase_ego[k], phase_follower[k])
                        for k in range(len(phase_ego))
                    ])
                    
                    phi_diff_leader = np.array([
                        compute_phase_diff_unit_vector(phase_ego[k], phase_leader[k])
                        for k in range(len(phase_ego))
                    ])
                    
                    # Compute errors (difference from theoretical 120°)
                    error_follower = phi_diff_follower - theoretical_phase
                    error_leader = phi_diff_leader - theoretical_phase
                    
                    # Store errors and time
                    errors_follower_all.append(error_follower)
                    errors_leader_all.append(error_leader)
                    time_references.append(time_ego)
                
                # If we have data from multiple runs, compute statistics
                if len(errors_follower_all) > 0:
                    # Find the maximum time series length (longest run)
                    max_length = max(len(t) for t in time_references)
                    
                    # Create a common time grid based on the longest run
                    # Use the first time reference as template, but extend to max_length if needed
                    longest_idx = np.argmax([len(t) for t in time_references])
                    time_common = time_references[longest_idx]
                    
                    # Interpolate all runs to the common time grid
                    errors_follower_interp = []
                    errors_leader_interp = []
                    
                    for idx, (time_ref, err_f, err_l) in enumerate(zip(time_references, errors_follower_all, errors_leader_all)):
                        # Interpolate to common time grid, using NaN for extrapolation
                        err_f_interp = np.interp(time_common, time_ref, err_f, left=np.nan, right=np.nan)
                        err_l_interp = np.interp(time_common, time_ref, err_l, left=np.nan, right=np.nan)
                        
                        errors_follower_interp.append(err_f_interp)
                        errors_leader_interp.append(err_l_interp)
                    
                    # Stack all interpolated errors
                    errors_follower_stacked = np.array(errors_follower_interp)
                    errors_leader_stacked = np.array(errors_leader_interp)
                    
                    # Compute mean and std, ignoring NaN values
                    mean_error_follower = np.nanmean(errors_follower_stacked, axis=0)
                    std_error_follower = np.nanstd(errors_follower_stacked, axis=0)
                    
                    mean_error_leader = np.nanmean(errors_leader_stacked, axis=0)
                    std_error_leader = np.nanstd(errors_leader_stacked, axis=0)
                    
                    # Plot mean with 3-sigma envelope
                    # Follower (red)
                    ax.plot(time_common, mean_error_follower, 'r-', linewidth=2.0, 
                           label=f'Error to Follower ({follower})', zorder=3)
                    ax.fill_between(time_common, 
                                    mean_error_follower - std_error_follower,
                                    mean_error_follower + std_error_follower,
                                    color='red', alpha=0.2, zorder=2)
                    
                    # Leader (blue)
                    ax.plot(time_common, mean_error_leader, 'b-', linewidth=2.0, 
                           label=f'Error to Leader ({leader})', zorder=3)
                    ax.fill_between(time_common, 
                                    mean_error_leader - std_error_leader,
                                    mean_error_leader + std_error_leader,
                                    color='blue', alpha=0.2, zorder=2)
                
                # Configure plot
                ax.grid(True, alpha=0.3)
                ax.set_ylim(-45, 45)
                
                # Labels
                if i == 0:
                    ax.set_title(f'{drone}', fontsize=12, fontweight='bold')
                if j == 0:
                    omega_value = speed.replace('_', '.')
                    ax.set_ylabel(f'ω = 0.{omega_value}\nError (deg)', fontsize=10)
                else:
                    ax.set_ylabel('Error (deg)', fontsize=10)
                
                if i == len(speeds) - 1:
                    ax.set_xlabel('Time (s)', fontsize=10)
                
                # Add legend (only for first subplot)
                if i == 0 and j == 0:
                    ax.legend(loc='upper right', fontsize=8)
                
            except Exception as e:
                print(f"Error plotting errors speed={speed}, drone={drone}: {e}")
                import traceback
                traceback.print_exc()
                ax.text(0.5, 0.5, f'Error:\n{str(e)[:50]}', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=8, color='red')
    
    plt.tight_layout()
    
    # Save figure
    output_path = base_dir / 'phase_errors_gps_montage.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nError plot saved to: {output_path}")
    
    plt.close()


if __name__ == '__main__':
    print("=" * 80)
    print("PLOTTING PHASE DIFFERENCES - GPS GROUP")
    print("=" * 80)
    plot_phase_differences()
    
    print("\n" + "=" * 80)
    print("PLOTTING PHASE ERRORS - GPS GROUP")
    print("=" * 80)
    plot_phase_errors()

