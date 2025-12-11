import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

plt.rcParams.update({'text.usetex': False, 'font.size': 20, 'figure.dpi': 300})

# Configuration
base_dir = Path('/home/paulo/Documents/DATA_NEW_2/')
groups = ['baseline', 'gps', 'relative']
models = ['modelA', 'modelC']
speeds = ['0_2', '0_4', '0_6', '0_8']
drones = ['C14', 'C05', 'C04']
colormap_name = 'gist_rainbow'  # Can be changed to: 'plasma', 'inferno', 'magma', 'cividis', 'tab10', etc.
labels = {
    'C04': 'Quadcopter 3',
    'C05': 'Quadcopter 2',
    'C14': 'Quadcopter 1'
}

# Duration to crop after encircle flag (in seconds)
CROP_DURATION = 60.0

# Drone relationships: ego -> (follower, leader)
DRONE_RELATIONSHIPS = {
    'C05': ('C14', 'C04'),
    'C04': ('C05', 'C14'),
    'C14': ('C04', 'C05')
}


def find_csv_files(group, model, speed):
    """
    Find CSV files for a given group, model, and speed.
    New structure: base_dir / group / model / speed / seed_XX / *.csv
    Seed folders are: seed_40, seed_45, seed_50, seed_55, seed_60
    """
    search_path = base_dir / group / model / speed
    
    if not search_path.exists():
        print(f"Warning: Path does not exist: {search_path}")
        return []
    
    # Find all seed folders and their CSV files
    csv_files = []
    seed_folders = [f"seed_{s}" for s in [40, 45, 50, 55, 60]]
    
    for seed_folder in seed_folders:
        seed_path = search_path / seed_folder
        if seed_path.exists():
            csv_files.extend(list(seed_path.glob('*.csv')))
    
    if len(csv_files) == 0:
        print(f"Warning: No CSV files found in {search_path}/seed_*")
    
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
    model = 'modelC'
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
                    
                    # Get phase_diff columns (already computed in the data)
                    phase_diff_follower_cols = [col for col in df.columns if drone in col and 'phase_diff' in col.lower() and 'follower' in col.lower() and source in col.lower()]
                    phase_diff_leader_cols = [col for col in df.columns if drone in col and 'phase_diff' in col.lower() and 'leader' in col.lower() and source in col.lower()]
                    
                    # If not found with source, try without
                    if len(phase_diff_follower_cols) == 0:
                        phase_diff_follower_cols = [col for col in df.columns if drone in col and 'phase_diff' in col.lower() and 'follower' in col.lower()]
                    if len(phase_diff_leader_cols) == 0:
                        phase_diff_leader_cols = [col for col in df.columns if drone in col and 'phase_diff' in col.lower() and 'leader' in col.lower()]
                    
                    if len(phase_diff_follower_cols) == 0 or len(phase_diff_leader_cols) == 0:
                        print(f"  Skipping {csv_path.name}: missing phase_diff columns")
                        continue
                    
                    phase_diff_follower_col = phase_diff_follower_cols[0]
                    phase_diff_leader_col = phase_diff_leader_cols[0]
                    
                    # Get time column
                    timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'stamp' in col.lower()]
                    time_col = timestamp_cols[0] if len(timestamp_cols) > 0 else None
                    
                    # Get valid data
                    # valid_data = df[[time_col, phase_diff_follower_col, phase_diff_leader_col]].dropna()
                    
                    # if len(valid_data) == 0:
                    #     print(f"  Skipping {csv_path.name}: insufficient valid data")
                    #     continue
                    
                    # Extract time and phase differences (already in degrees)
                    # time_data = df[time_col].values
                    phi_diff_follower = np.rad2deg(df[[time_col, phase_diff_follower_col]].dropna().values)
                    phi_diff_leader = np.rad2deg(df[[time_col, phase_diff_leader_col]].dropna().values)

                    # print(f'{len(phi_diff_follower)} samples found in {csv_path.name}')
                    
                    # Plot phase differences
                    # Red for follower, blue for leader
                    alpha = 0.5 if len(csv_files) > 1 else 0.7
                    label_follower = f'to Follower ({follower})' if csv_idx == 0 else None
                    label_leader = f'to Leader ({leader})' if csv_idx == 0 else None
                    
                    ax.plot(phi_diff_follower[:,0], phi_diff_follower[:,1], 'r-', linewidth=1.0, 
                           alpha=alpha, label=label_follower, zorder=2)
                    ax.plot(phi_diff_leader[:,0], phi_diff_leader[:,1], 'b-', linewidth=1.0, 
                           alpha=alpha, label=label_leader, zorder=2)
                
                # Configure plot
                ax.grid(True, alpha=0.3, linestyle=':')
                # ax.set_ylim(75, 165)
                
                # Labels
                if i == 0:
                    ax.set_title(f'{drone}', fontsize=12, fontweight='bold')
                if j == 0:
                    omega_value = speed.split('_')[1]
                    ax.set_ylabel(f'$\omega = 0.{omega_value}$\n\nPhase Diff (deg)', fontsize=10)
                # else:
                #     ax.set_ylabel('Phase Diff (deg)', fontsize=10)
                
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
    # fig.suptitle('Phase Errors (from Theoretical 120°) - GPS Group (ModelA)', fontsize=16, fontweight='bold')
    
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
                ax.axhline(y=0, color='k', linestyle='-', linewidth=1, 
                          label='Zero Error (120°)', alpha=0.9, zorder=1)
                
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
                    
                    # Get phase_diff columns (already computed in the data)
                    phase_diff_follower_cols = [col for col in df.columns if drone in col and 'phase_diff' in col.lower() and 'follower' in col.lower() and source in col.lower()]
                    phase_diff_leader_cols = [col for col in df.columns if drone in col and 'phase_diff' in col.lower() and 'leader' in col.lower() and source in col.lower()]
                    
                    # If not found with source, try without
                    if len(phase_diff_follower_cols) == 0:
                        phase_diff_follower_cols = [col for col in df.columns if drone in col and 'phase_diff' in col.lower() and 'follower' in col.lower()]
                    if len(phase_diff_leader_cols) == 0:
                        phase_diff_leader_cols = [col for col in df.columns if drone in col and 'phase_diff' in col.lower() and 'leader' in col.lower()]
                    
                    if len(phase_diff_follower_cols) == 0 or len(phase_diff_leader_cols) == 0:
                        print(f"  Skipping {csv_path.name}: missing phase_diff columns")
                        continue
                    
                    phase_diff_follower_col = phase_diff_follower_cols[0]
                    phase_diff_leader_col = phase_diff_leader_cols[0]
                    
                    # Get time column
                    timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'stamp' in col.lower()]
                    time_col = timestamp_cols[0] if len(timestamp_cols) > 0 else None
                    
                    # Get valid data
                    valid_data = df[[time_col, phase_diff_follower_col, phase_diff_leader_col]].dropna()
                    
                    if len(valid_data) == 0:
                        print(f"  Skipping {csv_path.name}: insufficient valid data")
                        continue
                    
                    # Extract time and phase differences (already in degrees)
                    time_data = valid_data[time_col].values
                    phi_diff_follower = valid_data[phase_diff_follower_col].values
                    phi_diff_leader = valid_data[phase_diff_leader_col].values
                    
                    # Compute errors (difference from theoretical 120°)
                    error_follower = phi_diff_follower - theoretical_phase
                    error_leader = phi_diff_leader - theoretical_phase
                    
                    # Store errors and time
                    errors_follower_all.append(error_follower)
                    errors_leader_all.append(error_leader)
                    time_references.append(time_data)
                
                # If we have data from multiple runs, compute statistics
                if len(errors_follower_all) > 0:                    
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
                                    color='red', alpha=0.15, zorder=2)
                    
                    # Leader (blue)
                    ax.plot(time_common, mean_error_leader, 'b-', linewidth=2.0, 
                           label=f'Error to Leader ({leader})', zorder=3)
                    ax.fill_between(time_common, 
                                    mean_error_leader - std_error_leader,
                                    mean_error_leader + std_error_leader,
                                    color='blue', alpha=0.15, zorder=2)
                
                # Process baseline data (single run)
                baseline_csv_files = find_csv_files('baseline', model, speed)
                if len(baseline_csv_files) > 0:
                    print(f"  Processing baseline: found {len(baseline_csv_files)} CSV file(s)")
                    
                    # Use the first (and likely only) baseline file
                    baseline_csv = baseline_csv_files[0]
                    df_baseline = load_and_crop_csv(baseline_csv)
                    
                    if df_baseline is not None and len(df_baseline) > 0:
                        # Get phase_diff columns from baseline
                        phase_diff_follower_cols_base = [col for col in df_baseline.columns if drone in col and 'phase_diff' in col.lower() and 'follower' in col.lower()]
                        phase_diff_leader_cols_base = [col for col in df_baseline.columns if drone in col and 'phase_diff' in col.lower() and 'leader' in col.lower()]
                        
                        if len(phase_diff_follower_cols_base) > 0 and len(phase_diff_leader_cols_base) > 0:
                            phase_diff_follower_col_base = phase_diff_follower_cols_base[0]
                            phase_diff_leader_col_base = phase_diff_leader_cols_base[0]
                            
                            # Get time column
                            timestamp_cols = [col for col in df_baseline.columns if 'time' in col.lower() or 'stamp' in col.lower()]
                            time_col_base = timestamp_cols[0] if len(timestamp_cols) > 0 else None
                            
                            # Get valid data
                            valid_data_base = df_baseline[[time_col_base, phase_diff_follower_col_base, phase_diff_leader_col_base]].dropna()
                            
                            if len(valid_data_base) > 0:
                                # Extract time and phase differences (already in degrees)
                                time_base = valid_data_base[time_col_base].values
                                phi_diff_follower_base = valid_data_base[phase_diff_follower_col_base].values
                                phi_diff_leader_base = valid_data_base[phase_diff_leader_col_base].values
                                
                                # Compute errors
                                error_follower_base = phi_diff_follower_base - theoretical_phase
                                error_leader_base = phi_diff_leader_base - theoretical_phase
                                
                                # Plot baseline as dashed lines
                                ax.plot(time_base, error_follower_base, 'r--', linewidth=2.5, 
                                       label='Baseline to Follower', alpha=0.9, zorder=4)
                                ax.plot(time_base, error_leader_base, 'b--', linewidth=2.5, 
                                       label='Baseline to Leader', alpha=0.9, zorder=4)
                        else:
                            print(f"    Baseline: missing phase_diff columns for {drone}")
                    else:
                        print(f"    Baseline: no valid data in {baseline_csv.name}")
                
                # Configure plot
                ax.grid(True, linestyle=':')
                ax.set_ylim(-45, 45)
                
                # Labels
                if i == 0:
                    ax.set_title(f'{labels[drone]}')
                if j == 0:
                    omega_value = speed.split('_')[1]
                    ax.set_ylabel(rf'$\omega = 0.{omega_value}$ (rad/s)' + '\n\n' + rf'$\mathbf{{\varepsilon_{{\phi}}}}$ (deg)')
                # else:
                #     ax.set_ylabel('Error (deg)', fontsize=10)
                
                if i == len(speeds) - 1:
                    ax.set_xlabel('Time (s)')
                
                # Add legend to the first subplot
                # if i == 0 and j == 2:
                #     ax.legend(loc='upper right', fontsize=7, framealpha=0.9)
                
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


def plot_radius_errors():
    """
    Create a 4x3 montage showing radius values for GPS group.
    Rows: speeds (0.2, 0.4, 0.6, 0.8)
    Columns: drones (C04, C05, C14)
    
    For each drone:
    - Blue lines: individual radius values from 5 CSV files
    - Black dashed line: nominal radius (1.0 m)
    """
    fig, axes = plt.subplots(4, 3, figsize=(18, 16), sharex=True, sharey=True)
    
    # Process GPS group with modelA
    group = 'gps'
    model = 'modelC'
    source = 'filtered'
    nominal_radius = 1.0  # meters
    
    for i, speed in enumerate(speeds):
        for j, drone in enumerate(drones):
            ax = axes[i, j]
            
            try:
                # Find CSV files (should be 5 for GPS group)
                csv_files = find_csv_files(group, model, speed)
                
                if len(csv_files) == 0:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue
                
                print(f"Processing radius errors speed={speed}, drone={drone}: found {len(csv_files)} CSV files")
                
                # Plot nominal radius line
                ax.axhline(y=nominal_radius, color='k', linestyle='--', linewidth=2, 
                          label=f'Nominal ({nominal_radius} m)', alpha=0.7, zorder=1)
                
                # Process each CSV file
                for csv_idx, csv_path in enumerate(csv_files):
                    df = load_and_crop_csv(csv_path)
                    
                    if df is None or len(df) == 0:
                        print(f"  Skipping {csv_path.name}: no valid data")
                        continue
                    
                    # Get radius column for this drone
                    radius_cols = [col for col in df.columns if drone in col and 'radius' in col.lower() and source in col.lower()]
                    
                    # If not found with source, try without
                    if len(radius_cols) == 0:
                        radius_cols = [col for col in df.columns if drone in col and 'radius' in col.lower()]
                    
                    if len(radius_cols) == 0:
                        print(f"  Skipping {csv_path.name}: missing radius column")
                        continue
                    
                    radius_col = radius_cols[0]
                    
                    # Get time column
                    timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'stamp' in col.lower()]
                    time_col = timestamp_cols[0] if len(timestamp_cols) > 0 else None
                    
                    # Get valid data
                    valid_data = df[[time_col, radius_col]].dropna()
                    
                    if len(valid_data) == 0:
                        print(f"  Skipping {csv_path.name}: insufficient valid data")
                        continue
                    
                    # Extract time and radius
                    time_data = valid_data[time_col].values
                    radius_data = valid_data[radius_col].values
                    
                    # Plot individual radius
                    alpha = 0.5 if len(csv_files) > 1 else 0.7
                    label = 'Radius' if csv_idx == 0 else None
                    
                    ax.plot(time_data, radius_data, 'b-', linewidth=1.0, 
                           alpha=alpha, label=label, zorder=2)
                
                # Configure plot
                ax.grid(True, alpha=0.3, linestyle=':')
                ax.set_ylim(0.7, 1.3)
                
                # Labels
                if i == 0:
                    ax.set_title(f'{labels[drone]}')
                if j == 0:
                    omega_value = speed.split('_')[1]
                    ax.set_ylabel(rf'$\omega = 0.{omega_value}$ (rad/s)' + '\n\n' + 'Radius (m)')
                
                if i == len(speeds) - 1:
                    ax.set_xlabel('Time (s)')
                
                ax.tick_params(labelsize=9)
                
                # Add legend (only for first subplot)
                if i == 0 and j == 0:
                    ax.legend(loc='upper right', fontsize=8)
                
            except Exception as e:
                print(f"Error plotting radius errors speed={speed}, drone={drone}: {e}")
                import traceback
                traceback.print_exc()
                ax.text(0.5, 0.5, f'Error:\n{str(e)[:50]}', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=8, color='red')
    
    plt.tight_layout()
    
    # Save figure
    output_path = base_dir / 'radius_gps_montage.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nRadius plot saved to: {output_path}")
    
    plt.close()


def plot_omega_errors():
    """
    Create a 1x4 montage showing omega errors (difference from nominal omega) for GPS group.
    Columns: speeds (0.2, 0.4, 0.6, 0.8)
    
    For each speed:
    - Three colored lines: one for each drone (C14, C05, C04)
    - Solid lines with envelopes: mean omega error ± σ (5 CSV files)
    - Dashed lines: baseline omega error (1 CSV file)
    - Black dashed line: zero error (nominal omega)
    """
    fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharex=True, sharey=True)
    
    # Get colormap for drones
    cmap = plt.get_cmap(colormap_name)
    colors = [cmap(i) for i in [0.125,0.65,0.9]]
    
    # Process GPS group with modelA
    group = 'gps'
    model = 'modelA'
    
    # Nominal omega values corresponding to each speed
    nominal_omegas = {
        '0_2': 0.2,
        '0_4': 0.4,
        '0_6': 0.6,
        '0_8': 0.8
    }
    
    for i, speed in enumerate(speeds):
        ax = axes[i]
        nominal_omega = nominal_omegas[speed]
        
        try:
            print(f"Processing omega errors for speed={speed} (nominal ω={nominal_omega})")
            
            # Plot zero error line (theoretical)
            ax.axhline(y=0, color='k', linestyle='-', linewidth=1, 
                      label='Zero Error', alpha=0.9, zorder=1)
            
            # Process each drone
            for drone_idx, drone in enumerate(drones):
                drone_color = colors[drone_idx]
                
                # Find CSV files (should be 5 for GPS group)
                csv_files = find_csv_files(group, model, speed)
                
                if len(csv_files) == 0:
                    print(f"  No data for {drone}")
                    continue
                
                print(f"  Processing {drone}: found {len(csv_files)} CSV files")
                
                # Collect all omega errors from CSV files
                errors_omega_all = []
                time_references = []
                
                # Process each CSV file
                for csv_idx, csv_path in enumerate(csv_files):
                    df = load_and_crop_csv(csv_path)
                    
                    if df is None or len(df) == 0:
                        print(f"    Skipping {csv_path.name}: no valid data")
                        continue
                    
                    # Get omega column for this drone
                    omega_cols = [col for col in df.columns if drone in col and 'omega' in col.lower() and 'filtered' in col.lower()]
                    
                    if len(omega_cols) == 0:
                        # Try without 'filtered' keyword
                        omega_cols = [col for col in df.columns if drone in col and 'omega' in col.lower()]
                    
                    if len(omega_cols) == 0:
                        print(f"    Skipping {csv_path.name}: no omega column for {drone}")
                        continue
                    
                    omega_col = omega_cols[0]
                    
                    # Get time column
                    timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'stamp' in col.lower()]
                    time_col = timestamp_cols[0] if len(timestamp_cols) > 0 else None
                    
                    # Get valid data
                    omega_valid = df[[time_col, omega_col]].dropna()
                    
                    if len(omega_valid) == 0:
                        print(f"    Skipping {csv_path.name}: insufficient valid data")
                        continue
                    
                    # Extract time and omega
                    time_data = omega_valid[time_col].values
                    omega_data = omega_valid[omega_col].values
                    
                    # Compute errors (difference from nominal omega)
                    error_omega = omega_data - nominal_omega
                    
                    # Store errors and time
                    errors_omega_all.append(error_omega)
                    time_references.append(time_data)
                
                # If we have data from multiple runs, compute statistics
                if len(errors_omega_all) > 0:
                    # Create a common time grid based on the longest run
                    longest_idx = np.argmax([len(t) for t in time_references])
                    time_common = time_references[longest_idx]
                    
                    # Interpolate all runs to the common time grid
                    errors_omega_interp = []
                    
                    for idx, (time_ref, err_omega) in enumerate(zip(time_references, errors_omega_all)):
                        # Interpolate to common time grid, using NaN for extrapolation
                        err_omega_interp = np.interp(time_common, time_ref, err_omega, left=np.nan, right=np.nan)
                        errors_omega_interp.append(err_omega_interp)
                    
                    # Stack all interpolated errors
                    errors_omega_stacked = np.array(errors_omega_interp)
                    
                    # Compute mean and std, ignoring NaN values
                    mean_error_omega = np.nanmean(errors_omega_stacked, axis=0)
                    std_error_omega = np.nanstd(errors_omega_stacked, axis=0)
                    
                    # Plot mean with 1-sigma envelope
                    ax.plot(time_common, mean_error_omega, '-', color=drone_color, linewidth=2.0, 
                           label=f'{labels[drone]}', zorder=3)
                    ax.fill_between(time_common, 
                                    mean_error_omega - std_error_omega,
                                    mean_error_omega + std_error_omega,
                                    color=drone_color, alpha=0.15, zorder=2)
                
                # Process baseline data (single run)
                # baseline_csv_files = find_csv_files('baseline', model, speed)
                # if len(baseline_csv_files) > 0:
                #     # Use the first (and likely only) baseline file
                #     baseline_csv = baseline_csv_files[0]
                #     df_baseline = load_and_crop_csv(baseline_csv)
                    
                #     if df_baseline is not None and len(df_baseline) > 0:
                #         # Get omega column for this drone from baseline
                #         omega_cols_base = [col for col in df_baseline.columns if drone in col and 'omega' in col.lower()]
                        
                #         if len(omega_cols_base) > 0:
                #             omega_col_base = omega_cols_base[0]
                            
                #             # Get time column
                #             timestamp_cols = [col for col in df_baseline.columns if 'time' in col.lower() or 'stamp' in col.lower()]
                #             time_col_base = timestamp_cols[0] if len(timestamp_cols) > 0 else None
                            
                #             # Get valid data
                #             omega_valid_base = df_baseline[[time_col_base, omega_col_base]].dropna()
                            
                #             if len(omega_valid_base) > 0:
                #                 # Extract time and omega
                #                 time_data_base = omega_valid_base[time_col_base].values
                #                 omega_data_base = omega_valid_base[omega_col_base].values
                                
                #                 # Compute errors
                #                 error_omega_base = omega_data_base - nominal_omega
                                
                #                 # Plot baseline as dashed line
                #                 ax.plot(time_data_base, error_omega_base, '--', color=drone_color, 
                #                        linewidth=1.5, label=f'{labels[drone]} (Baseline)', 
                #                        alpha=0.9, zorder=4)
            
            # Configure plot
            ax.grid(True, linestyle=':')
            ax.set_ylim(-0.125, 0.125)
            ax.set_xlim(0, CROP_DURATION)
            
            # Labels
            omega_value = speed.split('_')[1]
            ax.set_title(rf'$\omega_{{d}} = 0.{omega_value}$ rad/s')
            ax.set_xlabel('Time (s)')
            
            if i == 0:
                ax.set_ylabel(rf'$\mathbf{{\varepsilon_{{\omega}}}}$ (rad/s)')
            
            # Add legend to the last subplot
            # if i == len(speeds) - 1:
            #     ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
            
        except Exception as e:
            print(f"Error plotting omega errors for speed={speed}: {e}")
            import traceback
            traceback.print_exc()
            ax.text(0.5, 0.5, f'Error:\n{str(e)[:50]}', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=8, color='red')
    
    plt.tight_layout()
    
    # Save figure
    output_path = base_dir / 'omega_errors_gps_montage.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nOmega error plot saved to: {output_path}")
    
    plt.close()


def plot_phase_errors_single_drone(drone='C05'):
    """
    Create a 4x4 montage showing phase errors for a single drone across different experiments.
    Rows: speeds (0.2, 0.4, 0.6, 0.8)
    Columns: experiments (GPS+ModelA, GPS+ModelC, Relative+ModelA, Relative+ModelC)
    
    For each cell:
    - Red line/envelope: mean error to follower ± σ (5 CSV files for gps/relative, 1 for baseline)
    - Blue line/envelope: mean error to leader ± σ (5 CSV files for gps/relative, 1 for baseline)
    - Black line: zero error (theoretical 120° separation)
    """
    fig, axes = plt.subplots(4, 4, figsize=(24, 16), sharex=True, sharey=True)
    
    # Get follower and leader for this ego drone
    follower, leader = DRONE_RELATIONSHIPS[drone]
    
    theoretical_phase = 120.0  # degrees
    source = 'filtered'
    
    # Experiment configurations: (group, model)
    experiments = [
        ('gps', 'modelA'),
        ('relative', 'modelA'),
        ('gps', 'modelC'),
        ('relative', 'modelC')
    ]
    
    experiment_labels = [
        'Filter 1 + Model A',
        'Filter 2 + Model A',
        'Filter 1 + Model C',
        'Filter 2 + Model C'
    ]
    
    for i, speed in enumerate(speeds):
        for j, (group, model) in enumerate(experiments):
            ax = axes[i, j]
            
            try:
                print(f"Processing speed={speed}, experiment={group}+{model}")
                
                # Plot zero error line (theoretical)
                ax.axhline(y=0, color='k', linestyle='-', linewidth=1, 
                        #   label='Zero Error (120°)',
                            alpha=0.9, zorder=1)
                
                # Find CSV files
                csv_files = find_csv_files(group, model, speed)
                
                if len(csv_files) == 0:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    continue
                
                print(f"  Found {len(csv_files)} CSV files")
                
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
                    
                    # Get phase_diff columns (already computed in the data)
                    phase_diff_follower_cols = [col for col in df.columns if drone in col and 'phase_diff' in col.lower() and 'follower' in col.lower() and source in col.lower()]
                    phase_diff_leader_cols = [col for col in df.columns if drone in col and 'phase_diff' in col.lower() and 'leader' in col.lower() and source in col.lower()]
                    
                    # If not found with source, try without
                    if len(phase_diff_follower_cols) == 0:
                        phase_diff_follower_cols = [col for col in df.columns if drone in col and 'phase_diff' in col.lower() and 'follower' in col.lower()]
                    if len(phase_diff_leader_cols) == 0:
                        phase_diff_leader_cols = [col for col in df.columns if drone in col and 'phase_diff' in col.lower() and 'leader' in col.lower()]
                    
                    if len(phase_diff_follower_cols) == 0 or len(phase_diff_leader_cols) == 0:
                        print(f"  Skipping {csv_path.name}: missing phase_diff columns")
                        continue
                    
                    phase_diff_follower_col = phase_diff_follower_cols[0]
                    phase_diff_leader_col = phase_diff_leader_cols[0]
                    
                    # Get time column
                    timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'stamp' in col.lower()]
                    time_col = timestamp_cols[0] if len(timestamp_cols) > 0 else None
                    
                    # Get valid data
                    valid_data = df[[time_col, phase_diff_follower_col, phase_diff_leader_col]].dropna()
                    
                    if len(valid_data) == 0:
                        print(f"  Skipping {csv_path.name}: insufficient valid data")
                        continue
                    
                    # Extract time and phase differences (already in degrees)
                    time_data = valid_data[time_col].values
                    phi_diff_follower = valid_data[phase_diff_follower_col].values
                    phi_diff_leader = valid_data[phase_diff_leader_col].values
                    
                    # Compute errors (difference from theoretical 120°)
                    error_follower = phi_diff_follower - theoretical_phase
                    error_leader = phi_diff_leader - theoretical_phase
                    
                    # Store errors and time
                    errors_follower_all.append(error_follower)
                    errors_leader_all.append(error_leader)
                    time_references.append(time_data)
                
                # If we have data from multiple runs, compute statistics
                if len(errors_follower_all) > 0:                    
                    # Create a common time grid based on the longest run
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
                    
                    # Plot mean with 1-sigma envelope
                    # Follower (red)
                    ax.plot(time_common, mean_error_follower, 'r-', linewidth=2.0, 
                           label=f'Error to Follower', zorder=3)
                    ax.fill_between(time_common, 
                                    mean_error_follower - std_error_follower,
                                    mean_error_follower + std_error_follower,
                                    color='red', alpha=0.15, zorder=2)
                    
                    # Leader (blue)
                    ax.plot(time_common, mean_error_leader, 'b-', linewidth=2.0, 
                           label=f'Error to Leader', zorder=3)
                    ax.fill_between(time_common, 
                                    mean_error_leader - std_error_leader,
                                    mean_error_leader + std_error_leader,
                                    color='blue', alpha=0.15, zorder=2)
                
                # Process baseline data (single run) for comparison
                baseline_csv_files = find_csv_files('baseline', model, speed)
                if len(baseline_csv_files) > 0:
                    print(f"  Processing baseline: found {len(baseline_csv_files)} CSV file(s)")
                    
                    # Use the first (and likely only) baseline file
                    baseline_csv = baseline_csv_files[0]
                    df_baseline = load_and_crop_csv(baseline_csv)
                    
                    if df_baseline is not None and len(df_baseline) > 0:
                        # Get phase_diff columns from baseline
                        phase_diff_follower_cols_base = [col for col in df_baseline.columns if drone in col and 'phase_diff' in col.lower() and 'follower' in col.lower()]
                        phase_diff_leader_cols_base = [col for col in df_baseline.columns if drone in col and 'phase_diff' in col.lower() and 'leader' in col.lower()]
                        
                        if len(phase_diff_follower_cols_base) > 0 and len(phase_diff_leader_cols_base) > 0:
                            phase_diff_follower_col_base = phase_diff_follower_cols_base[0]
                            phase_diff_leader_col_base = phase_diff_leader_cols_base[0]
                            
                            # Get time column
                            timestamp_cols = [col for col in df_baseline.columns if 'time' in col.lower() or 'stamp' in col.lower()]
                            time_col_base = timestamp_cols[0] if len(timestamp_cols) > 0 else None
                            
                            # Get valid data
                            valid_data_base = df_baseline[[time_col_base, phase_diff_follower_col_base, phase_diff_leader_col_base]].dropna()
                            
                            if len(valid_data_base) > 0:
                                # Extract time and phase differences (already in degrees)
                                time_base = valid_data_base[time_col_base].values
                                phi_diff_follower_base = valid_data_base[phase_diff_follower_col_base].values
                                phi_diff_leader_base = valid_data_base[phase_diff_leader_col_base].values
                                
                                # Compute errors
                                error_follower_base = phi_diff_follower_base - theoretical_phase
                                error_leader_base = phi_diff_leader_base - theoretical_phase
                                
                                # Plot baseline as dashed lines
                                ax.plot(time_base, error_follower_base, 'm--', linewidth=2.5, 
                                       label='Baseline to Follower', alpha=1.0, zorder=4)
                                ax.plot(time_base, error_leader_base, 'c--', linewidth=2.5, 
                                       label='Baseline to Leader', alpha=1.0, zorder=4)
                
                # Configure plot
                ax.grid(True, linestyle=':')
                ax.set_ylim(-45, 45)

                ax.set_xlim(0, CROP_DURATION)
                
                # Labels
                if i == 0:
                    ax.set_title(f'{experiment_labels[j]}', fontweight='bold')
                if j == 0:
                    omega_value = speed.split('_')[1]
                    ax.set_ylabel(rf'$\omega = 0.{omega_value}$ (rad/s)' + '\n\n' + rf'$\mathbf{{\varepsilon_{{\phi}}}}$ (deg)')
                
                if i == len(speeds) - 1:
                    ax.set_xlabel('Time (s)')
                
                # Add subplot label (a) to (p) - for 4x4 grid
                subplot_index = i * 4 + j  # 0 to 15
                subplot_label = chr(97 + subplot_index)  # 'a' to 'p'
                ax.text(0.98, 0.03, f'({subplot_label})', transform=ax.transAxes,
                       fontsize=18, fontweight='bold', va='bottom', ha='right',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
                
                # Add legend to the top-right subplot
                if i == 0 and j == 3:
                    ax.legend(loc='upper right', fontsize=14, framealpha=0.9)
                
            except Exception as e:
                print(f"Error plotting speed={speed}, experiment={group}+{model}: {e}")
                import traceback
                traceback.print_exc()
                ax.text(0.5, 0.5, f'Error:\n{str(e)[:50]}', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=8, color='red')
    
    # fig.suptitle(f'Phase Errors for {labels[drone]} - Comparison Across Experiments', 
    #              fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save figure
    output_path = base_dir / f'phase_errors_{drone}_experiments_montage.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPhase errors plot for {drone} saved to: {output_path}")
    
    plt.close()


def plot_radius_errors_single_drone(drone='C05'):
    """
    Create a 4x4 montage showing radius errors for a single drone across different experiments.
    Rows: speeds (0.2, 0.4, 0.6, 0.8)
    Columns: experiments (GPS+ModelA, GPS+ModelC, Relative+ModelA, Relative+ModelC)
    
    For each cell:
    - Mean radius error ± σ (difference from nominal 1.0 m)
    - Black line: zero error (nominal radius = 1.0 m)
    - Baseline radius error shown as dashed line
    """
    fig, axes = plt.subplots(4, 4, figsize=(24, 16), sharex=True, sharey=True)
    
    nominal_radius = 1.0  # meters
    source = 'filtered'
    
    # Experiment configurations: (group, model)
    experiments = [
        ('gps', 'modelA'),
        ('relative', 'modelA'),
        ('gps', 'modelC'),
        ('relative', 'modelC')
    ]
    
    experiment_labels = [
        'Filter 1 + Model A',
        'Filter 2 + Model A',
        'Filter 1 + Model C',
        'Filter 2 + Model C'
    ]
    
    for i, speed in enumerate(speeds):
        for j, (group, model) in enumerate(experiments):
            ax = axes[i, j]
            
            try:
                print(f"Processing radius errors: speed={speed}, experiment={group}+{model}")
                
                # Plot zero error line (theoretical)
                ax.axhline(y=0, color='k', linestyle='-', linewidth=1, alpha=0.9, zorder=1)
                
                # Find CSV files
                csv_files = find_csv_files(group, model, speed)
                
                if len(csv_files) == 0:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    continue
                
                print(f"  Found {len(csv_files)} CSV files")
                
                # Collect all radius errors from CSV files
                radius_errors_all = []
                time_references = []
                
                # Process each CSV file
                for csv_idx, csv_path in enumerate(csv_files):
                    df = load_and_crop_csv(csv_path)
                    
                    if df is None or len(df) == 0:
                        print(f"  Skipping {csv_path.name}: no valid data")
                        continue
                    
                    # Get radius column for this drone
                    radius_cols = [col for col in df.columns if drone in col and 'radius' in col.lower() and source in col.lower()]
                    
                    # If not found with source, try without
                    if len(radius_cols) == 0:
                        radius_cols = [col for col in df.columns if drone in col and 'radius' in col.lower()]
                    
                    if len(radius_cols) == 0:
                        print(f"  Skipping {csv_path.name}: missing radius column")
                        continue
                    
                    radius_col = radius_cols[0]
                    
                    # Get time column
                    timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'stamp' in col.lower()]
                    time_col = timestamp_cols[0] if len(timestamp_cols) > 0 else None
                    
                    # Get valid data
                    valid_data = df[[time_col, radius_col]].dropna()
                    
                    if len(valid_data) == 0:
                        print(f"  Skipping {csv_path.name}: insufficient valid data")
                        continue
                    
                    # Extract time and radius
                    time_data = valid_data[time_col].values
                    radius_data = valid_data[radius_col].values
                    
                    # Compute error (difference from nominal)
                    radius_error = radius_data - nominal_radius
                    
                    # Store errors and time
                    radius_errors_all.append(radius_error)
                    time_references.append(time_data)
                
                # If we have data from multiple runs, compute statistics
                if len(radius_errors_all) > 0:
                    # Create a common time grid based on the longest run
                    longest_idx = np.argmax([len(t) for t in time_references])
                    time_common = time_references[longest_idx]
                    
                    # Interpolate all runs to the common time grid
                    radius_errors_interp = []
                    
                    for idx, (time_ref, err) in enumerate(zip(time_references, radius_errors_all)):
                        # Interpolate to common time grid, using NaN for extrapolation
                        err_interp = np.interp(time_common, time_ref, err, left=np.nan, right=np.nan)
                        radius_errors_interp.append(err_interp)
                    
                    # Stack all interpolated errors
                    radius_errors_stacked = np.array(radius_errors_interp)
                    
                    # Compute mean and std, ignoring NaN values
                    mean_radius_error = np.nanmean(radius_errors_stacked, axis=0)
                    std_radius_error = np.nanstd(radius_errors_stacked, axis=0)
                    
                    # Plot mean with 1-sigma envelope
                    ax.plot(time_common, mean_radius_error, 'b-', linewidth=2.0, 
                           label=f'Radius Error', zorder=3)
                    ax.fill_between(time_common, 
                                    mean_radius_error - std_radius_error,
                                    mean_radius_error + std_radius_error,
                                    color='blue', alpha=0.15, zorder=2)
                
                # Process baseline data (single run) for comparison
                baseline_csv_files = find_csv_files('baseline', model, speed)
                if len(baseline_csv_files) > 0:
                    print(f"  Processing baseline: found {len(baseline_csv_files)} CSV file(s)")
                    
                    # Use the first (and likely only) baseline file
                    baseline_csv = baseline_csv_files[0]
                    df_baseline = load_and_crop_csv(baseline_csv)
                    
                    if df_baseline is not None and len(df_baseline) > 0:
                        # Get position columns for this drone from baseline
                        x_cols_base = [col for col in df_baseline.columns if drone in col and '_x' in col.lower()]
                        y_cols_base = [col for col in df_baseline.columns if drone in col and '_y' in col.lower()]
                        
                        if len(x_cols_base) > 0 and len(y_cols_base) > 0:
                            x_col_base = x_cols_base[0]
                            y_col_base = y_cols_base[0]
                            
                            # Get time column
                            timestamp_cols = [col for col in df_baseline.columns if 'time' in col.lower() or 'stamp' in col.lower()]
                            time_col_base = timestamp_cols[0] if len(timestamp_cols) > 0 else None
                            
                            # Get valid data
                            valid_data_base = df_baseline[[time_col_base, x_col_base, y_col_base]].dropna()
                            
                            if len(valid_data_base) > 0:
                                # Extract time and positions
                                time_base = valid_data_base[time_col_base].values
                                x_base = valid_data_base[x_col_base].values
                                y_base = valid_data_base[y_col_base].values
                                
                                # Compute radius
                                radius_base = np.sqrt(x_base**2 + y_base**2)
                                
                                # Compute error
                                radius_error_base = radius_base - nominal_radius
                                
                                # Plot baseline as dashed line
                                ax.plot(time_base, radius_error_base, 'r--', linewidth=2.5, 
                                       label='Baseline', alpha=1.0, zorder=4)
                
                # Configure plot
                ax.grid(True, linestyle=':')
                ax.set_ylim(-0.3, 0.3)
                ax.set_xlim(0, CROP_DURATION)
                
                # Labels
                if i == 0:
                    ax.set_title(f'{experiment_labels[j]}', fontweight='bold')
                if j == 0:
                    omega_value = speed.split('_')[1]
                    ax.set_ylabel(rf'$\omega = 0.{omega_value}$ (rad/s)' + '\n\n' + rf'$\mathbf{{\varepsilon_{{r}}}}$ (m)')
                
                if i == len(speeds) - 1:
                    ax.set_xlabel('Time (s)')
                
                # Add subplot label (a) to (p) - for 4x4 grid
                subplot_index = i * 4 + j  # 0 to 15
                subplot_label = chr(97 + subplot_index)  # 'a' to 'p'
                ax.text(0.98, 0.03, f'({subplot_label})', transform=ax.transAxes,
                       fontsize=18, fontweight='bold', va='bottom', ha='right',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
                
                # Add legend to the top-right subplot
                if i == 0 and j == 3:
                    ax.legend(loc='upper right', fontsize=14, framealpha=0.9)
                
            except Exception as e:
                print(f"Error plotting radius errors: speed={speed}, experiment={group}+{model}: {e}")
                import traceback
                traceback.print_exc()
                ax.text(0.5, 0.5, f'Error:\n{str(e)[:50]}', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=8, color='red')
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save figure
    output_path = base_dir / f'radius_errors_{drone}_experiments_montage.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nRadius errors plot for {drone} saved to: {output_path}")
    
    plt.close()


def plot_3d_trajectories_single_drone(drone='C05'):
    """
    Create two 2x4 montages showing 3D trajectories for a single drone across different experiments.
    Figure 1 (Model A): GPS+ModelA, Relative+ModelA
    Figure 2 (Model C): GPS+ModelC, Relative+ModelC
    Rows: filters (GPS, Relative)
    Columns: speeds (0.2, 0.4, 0.6, 0.8)
    
    For each cell:
    - Black dotted line: baseline trajectory (1 CSV file)
    - Colored lines: filtered trajectories with alpha (5 CSV files)
    - Z-axis includes spiral effect for temporal visualization
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    # Process each model separately
    models_to_plot = ['modelA', 'modelC']
    
    for model in models_to_plot:
        fig = plt.figure(figsize=(24, 12))
        gs = fig.add_gridspec(2, 4, hspace=0.08, wspace=0.12, left=0.05, right=0.98, top=0.95, bottom=0.05)
        
        # Experiment configurations for this model: (group, model)
        experiments = [
            ('gps', model),
            ('relative', model)
        ]
        
        experiment_labels = [
            'GPS',
            'Relative'
        ]
        
        # Get colormap for the 5 filtered runs
        cmap = plt.get_cmap('inferno')
        colors = [cmap(i / 4) for i in range(5)]  # 5 colors for 5 runs
        
        source = 'filtered'
        spiral_rate = 0.5  # Spiral effect rate for z-axis
        
        for i, (group, current_model) in enumerate(experiments):
            for j, speed in enumerate(speeds):
                ax = fig.add_subplot(gs[i, j], projection='3d')
                
                try:
                    print(f"Processing 3D trajectory speed={speed}, experiment={group}+{current_model}, drone={drone}")
                    
                    # Process baseline data first
                    baseline_csv_files = find_csv_files('baseline', current_model, speed)
                    
                    baseline_x = baseline_y = baseline_z = baseline_t = None
                    
                    if len(baseline_csv_files) > 0:
                        baseline_csv = baseline_csv_files[0]
                        df_baseline = load_and_crop_csv(baseline_csv)
                        
                        if df_baseline is not None and len(df_baseline) > 0:
                            # Get position columns for this drone
                            x_cols = [col for col in df_baseline.columns if drone in col and 'vicon_position_pos_x' in col.lower()]
                            y_cols = [col for col in df_baseline.columns if drone in col and 'vicon_position_pos_y' in col.lower()]
                            z_cols = [col for col in df_baseline.columns if drone in col and 'vicon_position_pos_z' in col.lower()]
                            
                            # Get time column
                            timestamp_cols = [col for col in df_baseline.columns if 'time' in col.lower() or 'stamp' in col.lower()]
                            time_col = timestamp_cols[0] if len(timestamp_cols) > 0 else None
                            
                            if len(x_cols) > 0 and len(y_cols) > 0 and len(z_cols) > 0 and time_col:
                                x_col = x_cols[0]
                                y_col = y_cols[0]
                                z_col = z_cols[0]
                                
                                # Get valid data
                                valid_data = df_baseline[[time_col, x_col, y_col, z_col]].dropna()
                                
                                if len(valid_data) > 0:
                                    baseline_t = valid_data[time_col].values
                                    baseline_x = valid_data[x_col].values
                                    baseline_y = valid_data[y_col].values
                                    baseline_z = valid_data[z_col].values
                                    
                                    # Apply spiral effect: z_spiral = z + t * spiral_rate
                                    baseline_z_spiral = baseline_z + baseline_t * spiral_rate
                                    
                                    # Plot baseline in black with solid line
                                    ax.plot(baseline_x, baseline_y, baseline_z_spiral, 'k-', 
                                           label='Baseline', alpha=1.0, linewidth=2.0, zorder=0)
                                    # Mark start point
                                    ax.scatter(baseline_x[0], baseline_y[0], baseline_z_spiral[0], 
                                             c='black', marker='o', s=150, alpha=1.0, zorder=1)
                    
                    # Process filtered/experimental data (5 runs)
                    csv_files = find_csv_files(group, current_model, speed)
                    
                    if len(csv_files) == 0:
                        ax.text2D(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                        continue
                    
                    for csv_idx, csv_path in enumerate(csv_files):
                        df = load_and_crop_csv(csv_path)
                        
                        if df is None or len(df) == 0:
                            continue
                        
                        # Get position columns for this drone
                        x_cols = [col for col in df.columns if drone in col and '_x' in col.lower() and source in col.lower()]
                        y_cols = [col for col in df.columns if drone in col and '_y' in col.lower() and source in col.lower()]
                        z_cols = [col for col in df.columns if drone in col and '_z' in col.lower() and source in col.lower()]
                        
                        # If not found with 'filtered', try without
                        if len(x_cols) == 0:
                            x_cols = [col for col in df.columns if drone in col and '_x' in col.lower()]
                        if len(y_cols) == 0:
                            y_cols = [col for col in df.columns if drone in col and '_y' in col.lower()]
                        if len(z_cols) == 0:
                            z_cols = [col for col in df.columns if drone in col and '_z' in col.lower()]
                        
                        # Get time column
                        timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'stamp' in col.lower()]
                        time_col = timestamp_cols[0] if len(timestamp_cols) > 0 else None
                        
                        if len(x_cols) > 0 and len(y_cols) > 0 and len(z_cols) > 0 and time_col:
                            x_col = x_cols[0]
                            y_col = y_cols[0]
                            z_col = z_cols[0]
                            
                            # Get valid data
                            valid_data = df[[time_col, x_col, y_col, z_col]].dropna()
                            
                            if len(valid_data) > 0:
                                t_data = valid_data[time_col].values
                                x_data = valid_data[x_col].values
                                y_data = valid_data[y_col].values
                                z_data = valid_data[z_col].values
                                
                                # Apply spiral effect
                                z_spiral = z_data + t_data * spiral_rate
                                
                                # Plot filtered trajectory with color and alpha
                                run_color = colors[csv_idx % 5]
                                label = f'Run {csv_idx + 1}' if csv_idx < 5 else None
                                ax.plot(x_data, y_data, z_spiral, '-', color=run_color,
                                       label=label, alpha=0.6, linewidth=1.5, zorder=5)
                                # Mark start point
                                ax.scatter(x_data[0], y_data[0], z_spiral[0], 
                                         c=[run_color], marker='o', s=50, alpha=0.8, zorder=6)
                                
                                # ax.view_init(elev=20, azim=-45, roll=0)
                                # ax.set_box_aspect([10, 10, 10])  # Aspect ratio
                    
                    # Set labels
                    ax.set_xlabel('X (m)', fontsize=10)
                    ax.set_ylabel('Y (m)', fontsize=10)
                    ax.set_zlabel('Z (m)', fontsize=10)
                    
                    # Set title for top row (show speed)
                    if i == 0:
                        omega_value = speed.split('_')[1]
                        ax.set_title(rf'$\omega = 0.{omega_value}$ rad/s', fontsize=14, fontweight='bold')
                    
                    # Add legend only to first subplot
                    if i == 0 and j == 0:
                        ax.legend(loc='upper right', fontsize=8)
                    
                    # Set viewing angle and orthographic projection
                    ax.view_init(elev=20, azim=45)
                    ax.set_proj_type('ortho')
                    
                    # Set equal aspect ratio
                    # Collect all data points for setting limits
                    all_x = []
                    all_y = []
                    all_z = []
                    
                    if baseline_x is not None:
                        all_x.extend(baseline_x)
                        all_y.extend(baseline_y)
                        all_z.extend(baseline_z_spiral)
                    
                    # Add filtered data limits
                    for csv_idx, csv_path in enumerate(csv_files[:5]):  # Only first 5
                        df = load_and_crop_csv(csv_path)
                        if df is not None:
                            x_cols = [col for col in df.columns if drone in col and '_x' in col.lower()]
                            y_cols = [col for col in df.columns if drone in col and '_y' in col.lower()]
                            z_cols = [col for col in df.columns if drone in col and '_z' in col.lower()]
                            timestamp_cols = [col for col in df.columns if 'time' in col.lower()]
                            
                            if len(x_cols) > 0 and len(y_cols) > 0 and len(z_cols) > 0 and len(timestamp_cols) > 0:
                                valid_data = df[[timestamp_cols[0], x_cols[0], y_cols[0], z_cols[0]]].dropna()
                                if len(valid_data) > 0:
                                    t_data = valid_data[timestamp_cols[0]].values
                                    all_x.extend(valid_data[x_cols[0]].values)
                                    all_y.extend(valid_data[y_cols[0]].values)
                                    all_z.extend(valid_data[z_cols[0]].values + t_data * spiral_rate)
                    
                    if len(all_x) > 0:
                        all_x = np.array(all_x)
                        all_y = np.array(all_y)
                        all_z = np.array(all_z)
                        
                        max_range_x = all_x.max()-all_x.min()
                        max_range_y = all_y.max()-all_y.min()
                        max_range_z = all_z.max()-all_z.min()
                        
                        mid_x = (all_x.max()+all_x.min()) * 0.5
                        mid_y = (all_y.max()+all_y.min()) * 0.5
                        mid_z = (all_z.max()+all_z.min()) * 0.5
                        
                        ax.set_xlim(mid_x - max_range_x, mid_x + max_range_x)
                        ax.set_ylim(mid_y - max_range_y, mid_y + max_range_y)
                        ax.set_zlim(mid_z - max_range_z, mid_z + max_range_z)
                    
                    # Add filter label on the left
                    if j == 0:
                        ax.text2D(-0.18, 0.5, experiment_labels[i], 
                                 transform=ax.transAxes,
                                 fontsize=16, fontweight='bold',
                                 va='center', ha='center', rotation=90)
                    
                    # Add subplot label
                    subplot_index = i * 4 + j  # 0 to 7 for 2x4 grid
                    subplot_label = chr(97 + subplot_index)
                    ax.text2D(0.98, 0.02, f'({subplot_label})', transform=ax.transAxes,
                             fontsize=14, fontweight='bold', va='bottom', ha='right',
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
                    
                except Exception as e:
                    print(f"Error plotting 3D trajectory speed={speed}, experiment={group}+{current_model}: {e}")
                    import traceback
                    traceback.print_exc()
                    ax.text2D(0.5, 0.5, f'Error:\n{str(e)[:50]}', ha='center', va='center', 
                             transform=ax.transAxes, fontsize=8, color='red')
        
        model_name = 'Model A' if model == 'modelA' else 'Model C'
        
        # Save figure
        output_path = base_dir / f'3d_trajectories_{drone}_{model}_montage.png'
        plt.savefig(output_path, dpi=150)
        print(f"\n3D trajectories plot for {drone} ({model_name}) saved to: {output_path}")
        
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
    
    print("\n" + "=" * 80)
    print("PLOTTING OMEGA ERRORS - GPS GROUP")
    print("=" * 80)
    plot_omega_errors()

    print("\n" + "=" * 80)
    print("PLOTTING OMEGA ERRORS - GPS GROUP")
    print("=" * 80)
    plot_radius_errors()
    
    # # Plot phase errors for each drone across all experiments
    # for drone in drones:
    #     print("\n" + "=" * 80)
    #     print(f"PLOTTING PHASE ERRORS FOR {drone} ACROSS EXPERIMENTS")
    #     print("=" * 80)
    #     plot_phase_errors_single_drone(drone=drone)
    
    # # Plot radius errors for each drone across all experiments
    # for drone in drones:
    #     print("\n" + "=" * 80)
    #     print(f"PLOTTING RADIUS ERRORS FOR {drone} ACROSS EXPERIMENTS")
    #     print("=" * 80)
    #     plot_radius_errors_single_drone(drone=drone)
    
    # Plot 3D trajectories for each drone across all experiments
    for drone in drones:
        print("\n" + "=" * 80)
        print(f"PLOTTING 3D TRAJECTORIES FOR {drone} ACROSS EXPERIMENTS")
        print("=" * 80)
        plot_3d_trajectories_single_drone(drone=drone)

