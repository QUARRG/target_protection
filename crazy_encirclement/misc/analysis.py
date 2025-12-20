
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from tabulate import tabulate

warnings.filterwarnings('ignore')

plt.rcParams.update({'text.usetex': True, 'font.size': 20, 'figure.dpi': 150})

# helpers
from crazy_encirclement.filters import wrap_to_pi

# Configuration
base_dir = Path('/home/paulo/Documents/k_10/')
plots_dir = base_dir / 'plots'
plots_dir.mkdir(exist_ok=True)
# groups = ['baseline', 'gps', 'relative']
groups_list = [
    # 'baseline',
    # 'gps',
    # 'relative',
    # 'total_outage_wind_mild',
    'total_outage_wind_strong',
    # 'combined_wind_mild',
    'combined_wind_strong'
    ]
group_labels = [
    # 'Filter 1 (Outage and mild wind)',
    'Filter 1 (Outage and strong wind)',
    # 'Filter 1 + 2 (Outage and mild wind)',
    'Filter 1 + 2 (Outage and strong wind)',
    ]

models = [
    'modelA',
    # 'modelC'
    ]
model_labels = [
    'Model A',
    # 'Model B'
    ]

# Group labels for columns
speeds = ['0_2']
drones = ['C14', 'C05', 'C04']
colormap_name = 'gist_rainbow'  # Can be changed to: 'plasma', 'inferno', 'magma', 'cividis', 'tab10', etc.
labels = {
    'C04': 'Quadcopter 3',
    'C05': 'Quadcopter 2',
    'C14': 'Quadcopter 1'
}

# Global color mapping: drones[0] -> colors[0], drones[1] -> colors[1], drones[2] -> colors[2]
cmap = plt.get_cmap(colormap_name)
colors = [cmap(i) for i in [0.125, 0.65, 0.9]]  # [Quadcopter 1, Quadcopter 2, Quadcopter 3]

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
    If _encircle flag is not available, use the first filtered/omega entry as the start point.
    Returns the cropped dataframe.
    """
    try:
        df = pd.read_csv(csv_path)
        
        start_idx = None
        
        # Try to use _encircle_data column first
        if '_encircle_data' in df.columns:
            encircle_idx = df[df['_encircle_data'] == True].index
            if len(encircle_idx) > 0:
                start_idx = encircle_idx[0]
                # print(f"Using _encircle_data flag at index {start_idx} for {csv_path.name}")
        
        # Fallback: use first filtered/omega entry (any drone)
        if start_idx is None:
            print(f"Warning: _encircle_data not found or never True in {csv_path.name}, using filtered/omega fallback")
            
            # Find all omega columns with 'filtered' in the name
            omega_cols = [col for col in df.columns if 'omega' in col.lower() and 'filtered' in col.lower()]
            
            if len(omega_cols) == 0:
                print(f"Error: No omega columns found in {csv_path.name}")
                return None
            
            # Find the first non-NaN entry across all omega columns
            for omega_col in omega_cols:
                non_nan_idx = df[omega_col].dropna().index
                if len(non_nan_idx) > 0:
                    if start_idx is None or non_nan_idx[0] < start_idx:
                        start_idx = non_nan_idx[0]
            
            if start_idx is None:
                print(f"Error: No valid omega data found in {csv_path.name}")
                return None
            
            # print(f"Using first filtered/omega entry at index {start_idx} for {csv_path.name}")
        
        # Get timestamp column
        timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'stamp' in col.lower()]
        
        if len(timestamp_cols) == 0:
            print(f"Warning: No timestamp column found in {csv_path.name}")
            # If no timestamp, crop by row count (assuming constant rate)
            return df.iloc[start_idx:start_idx + int(crop_duration * 100)]  # Assuming ~100Hz
        
        # Use the first timestamp column found
        time_col = timestamp_cols[0]
        start_time = df.loc[start_idx, time_col]
        end_time = start_time + crop_duration + 1.0  # Add 1 second buffer to ensure we have data at crop_duration
        
        # Crop the dataframe
        cropped_df = df[(df[time_col] >= start_time) & (df[time_col] <= end_time)].copy()
        
        # Reset time to start from 0
        cropped_df[time_col] = cropped_df[time_col] - start_time
        
        return cropped_df
    
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None


def compute_settling_time_and_bounds(time_data, error_data, lower_bound=-5, upper_bound=5):
    """
    Compute time to reach and maintain bounds.
    
    Returns:
        time_to_enter: time when error first enters [lower_bound, upper_bound]
        time_to_stabilize: time when error enters bounds and stays there until end
        percent_time_in_bounds: percentage of time spent within bounds
    """
    in_bounds = (error_data >= lower_bound) & (error_data <= upper_bound)
    
    # Find first time entering bounds
    time_to_enter = None
    for i, in_bound in enumerate(in_bounds):
        if in_bound:
            time_to_enter = time_data[i]
            break
    
    # Find time when it enters bounds and stays there
    time_to_stabilize = None
    if time_to_enter is not None:
        # Find first index where it enters bounds
        first_idx = np.where(in_bounds)[0][0]
        
        # Check if it stays within bounds from that point onward
        remaining_in_bounds = in_bounds[first_idx:]
        
        # Find consecutive True values at the end
        consecutive_at_end = 0
        for val in reversed(remaining_in_bounds):
            if val:
                consecutive_at_end += 1
            else:
                break
        
        # If at least 80% of remaining time is in bounds from first entry, consider it stabilized
        if len(remaining_in_bounds) > 0 and consecutive_at_end / len(remaining_in_bounds) >= 0.8:
            time_to_stabilize = time_data[first_idx]
        else:
            time_to_stabilize = None
    
    # Compute percentage of time in bounds
    percent_time_in_bounds = np.sum(in_bounds) / len(in_bounds) * 100 if len(in_bounds) > 0 else 0
    
    return time_to_enter, time_to_stabilize, percent_time_in_bounds


def compute_integrated_absolute_error(time_data, error_data):
    """
    Compute Integrated Time-weighted Absolute Error (ITAE).
    ITAE = ∫t·|error(t)|dt
    
    Time-weighting emphasizes steady-state errors (later in flight) over startup errors.
    Uses trapezoidal rule for numerical integration.
    
    Args:
        time_data: array of time points
        error_data: array of error values
    
    Returns:
        itae: integrated time-weighted absolute error value
    """
    # Use trapezoidal rule for numerical integration with time-weighting
    abs_error = np.abs(error_data)
    time_weighted_error = time_data * abs_error
    itae = np.trapezoid(time_weighted_error, time_data)
    return itae


def plot_phases_differences_errors_experiments():
    """
    Create a 2x3 montage showing phase difference errors with leader.
    Rows: models (modelA, modelC)
    Columns: groups (baseline, gps/Filter1, relative/Filter2)
    Shows ±5 deg confidence bounds and computes settling time.
    """
    n_models = len(models)
    n_groups = len(groups_list)
    x_size = 6 * n_groups
    y_size = 5 * n_models
    fig, axes = plt.subplots(n_models, n_groups, figsize=(x_size, y_size), sharex=True, sharey=True)
    
    # Nominal phase difference (120 degrees = 2π/3 radians)
    nominal_phase_diff_deg = 120.0  # degrees
    confidence_bound = 5.0  # ±5 degrees    
    speed = speeds[0]  # Using only one speed for now
    
    for i_model, model in enumerate(models):
        for j_group, (group, group_label) in enumerate(zip(groups_list, group_labels)):
            ax = axes[i_model, j_group] if n_models > 1 else axes[j_group]
            
            try:
                print(f"Processing model={model}, group={group}")
                
                # Plot zero error line
                ax.axhline(y=0, color='k', linestyle='-', linewidth=1, alpha=0.9, zorder=1)
                
                # Plot confidence bounds (±5 degrees)
                ax.axhline(y=confidence_bound, color='k', linestyle=':', linewidth=1.5, alpha=0.9, zorder=1)
                ax.axhline(y=-confidence_bound, color='k', linestyle=':', linewidth=1.5, alpha=0.9, zorder=1)
                # ax.fill_between(ax.get_xlim(), -confidence_bound, confidence_bound, 
                #                color='gray', alpha=0.05, zorder=0)
                
                # Find CSV files for this group/model/speed
                csv_files = find_csv_files(group, model, speed)
                csv_files = [f for f in csv_files if 'processed' in f.name]
                
                if len(csv_files) == 0:
                    print(f"  Warning: No processed CSV files found for {group}/{model}/{speed}")
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=10, color='gray')
                else:
                    # Process each drone across all seed runs
                    for drone_idx, drone in enumerate(drones):
                        drone_color = colors[drone_idx]
                        
                        phase_diff_errors_all = []
                        time_references = []
                        
                        for csv_file in csv_files:
                            df = load_and_crop_csv(csv_file)
                            
                            if df is None or len(df) == 0:
                                continue
                            
                            # Get time column
                            timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'stamp' in col.lower()]
                            time_col = timestamp_cols[0] if len(timestamp_cols) > 0 else None
                            if time_col is None:
                                continue
                            
                            # Require measured phases
                            follower, leader = DRONE_RELATIONSHIPS[drone]
                            ego_meas_col = f"_{drone}_measured_phase"
                            leader_meas_col = f"_{leader}_measured_phase"
                            follower_meas_col = f"_{follower}_measured_phase"
                            
                            if not (ego_meas_col in df.columns and leader_meas_col in df.columns):
                                continue
                            
                            # Extract and align phase data
                            phase_leader_data = df[[time_col, leader_meas_col]].dropna().values
                            phase_ego_data = df[[time_col, ego_meas_col]].dropna().values
                            phase_follower_data = df[[time_col, follower_meas_col]].dropna().values
                            
                            lengths = [len(phase_leader_data), len(phase_ego_data), len(phase_follower_data)]
                            min_length = min(lengths) if lengths else 0
                            if min_length == 0:
                                continue
                            
                            phase_leader_data = phase_leader_data[:min_length, :]
                            phase_ego_data = phase_ego_data[:min_length, :]
                            
                            time_ref = phase_ego_data[:, 0]
                            phase_leader = phase_leader_data[:, 1]
                            phase_ego = phase_ego_data[:, 1]
                            
                            # Compute phase difference error
                            phase_diff_leader = wrap_to_pi(phase_leader - phase_ego)
                            phase_diff_deg = np.rad2deg(phase_diff_leader)
                            error_phase_diff = phase_diff_deg - nominal_phase_diff_deg
                            
                            phase_diff_errors_all.append(error_phase_diff)
                            time_references.append(time_ref)
                        
                        # Plot aggregated statistics
                        if len(phase_diff_errors_all) > 0:
                            longest_idx = np.argmax([len(t) for t in time_references])
                            time_common = time_references[longest_idx]
                            
                            errors_interp = [np.interp(time_common, t, e, left=np.nan, right=np.nan) 
                                            for t, e in zip(time_references, phase_diff_errors_all)]
                            stacked = np.array(errors_interp)
                            mean_err = np.nanmean(stacked, axis=0)
                            std_err = np.nanstd(stacked, axis=0)
                            
                            ax.plot(time_common, mean_err, '-', color=drone_color, linewidth=2.0, 
                                   label=labels[drone], zorder=3)
                            ax.fill_between(time_common, mean_err - std_err, mean_err + std_err,
                                           color=drone_color, alpha=0.15, zorder=2)
                
                # Configure plot
                ax.grid(True, linestyle=':')
                ax.set_ylim(-20, 20)
                ax.set_xlim(0, CROP_DURATION)
                
                # Column titles
                if i_model == 0:
                    ax.set_title(group_label, fontweight='bold')
                
                # Row labels (model names)
                if j_group == 0:
                    model_label = model_labels[i_model]
                    ax.set_ylabel(f'{model_label}\n\nError (deg)', fontweight='bold')
                
                # Bottom labels
                if i_model == n_models - 1:
                    ax.set_xlabel('Time (s)')
                
                # Legend on top-left cell
                if i_model == 0 and j_group == 0:
                    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
                
            except Exception as e:
                print(f"Error plotting model={model}, group={group}: {e}")
                import traceback
                traceback.print_exc()
                ax.text(0.5, 0.5, f'Error:\\n{str(e)[:50]}', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=8, color='red')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    output_path = plots_dir / 'phase_diff_leader_errors.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Phase difference leader errors plot saved to: {output_path}")
    plt.close()


def compute_phase_diff_settling_summary():
    """
    Compute and print settling time summary table for phase difference errors.
    """
    confidence_bound = 5.0  # ±5 degrees
    nominal_phase_diff_deg = 120.0    
    speed = speeds[0]
    
    # Store data for all models/groups/drones
    settling_summary = {model: {group: {drone: {'enter': None, 'stabilize': None, 'percent': None} 
                                       for drone in drones} 
                               for group in groups_list} 
                       for model in models}
    
    for model in models:
        for group in groups_list:            
            csv_files = find_csv_files(group, model, speed)
            csv_files = [f for f in csv_files if 'processed' in f.name]
            
            if len(csv_files) == 0:
                print("  No processed CSV files found")
                continue
            
            for drone in drones:
                phase_diff_errors_all = []
                time_references = []
                
                for csv_file in csv_files:
                    df = load_and_crop_csv(csv_file)
                    
                    if df is None or len(df) == 0:
                        continue
                    
                    # Get time column
                    timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'stamp' in col.lower()]
                    time_col = timestamp_cols[0] if len(timestamp_cols) > 0 else None
                    if time_col is None:
                        continue
                    
                    # Get measured phases
                    follower, leader = DRONE_RELATIONSHIPS[drone]
                    ego_meas_col = f"_{drone}_measured_phase"
                    leader_meas_col = f"_{leader}_measured_phase"
                    
                    if not (ego_meas_col in df.columns and leader_meas_col in df.columns):
                        continue
                    
                    # Extract and align phase data
                    phase_leader_data = df[[time_col, leader_meas_col]].dropna().values
                    phase_ego_data = df[[time_col, ego_meas_col]].dropna().values
                    
                    lengths = [len(phase_leader_data), len(phase_ego_data)]
                    min_length = min(lengths) if lengths else 0
                    if min_length == 0:
                        continue
                    
                    phase_leader_data = phase_leader_data[:min_length, :]
                    phase_ego_data = phase_ego_data[:min_length, :]
                    
                    time_ref = phase_ego_data[:, 0]
                    phase_leader = phase_leader_data[:, 1]
                    phase_ego = phase_ego_data[:, 1]
                    
                    # Compute phase difference error
                    phase_diff_leader = wrap_to_pi(phase_leader - phase_ego)
                    phase_diff_deg = np.rad2deg(phase_diff_leader)
                    error_phase_diff = phase_diff_deg - nominal_phase_diff_deg
                    
                    phase_diff_errors_all.append(error_phase_diff)
                    time_references.append(time_ref)
                
                # Compute aggregate statistics
                if len(phase_diff_errors_all) > 0:
                    longest_idx = np.argmax([len(t) for t in time_references])
                    time_common = time_references[longest_idx]
                    
                    errors_interp = [np.interp(time_common, t, e, left=np.nan, right=np.nan) 
                                    for t, e in zip(time_references, phase_diff_errors_all)]
                    stacked = np.array(errors_interp)
                    mean_err = np.nanmean(stacked, axis=0)
                    
                    # Compute settling time
                    time_to_enter, time_to_stabilize, percent_in_bounds = compute_settling_time_and_bounds(
                        time_common, mean_err, lower_bound=-confidence_bound, upper_bound=confidence_bound
                    )
                    
                    settling_summary[model][group][drone] = {
                        'enter': time_to_enter,
                        'stabilize': time_to_stabilize,
                        'percent': percent_in_bounds
                    }
    
    # Print comprehensive summary tables
    print("\n" + "=" * 140)
    print("SUMMARY TABLE - Phase Difference Settling Times")
    print("=" * 140)
    
    for group in groups_list:
        print(f"\n{group.upper()}:")
        
        # Combined comprehensive table for this group
        table_data_combined = []
        for model in models:
            for drone in drones:
                enter = settling_summary[model][group][drone]['enter']
                stabilize = settling_summary[model][group][drone]['stabilize']
                percent = settling_summary[model][group][drone]['percent']
                
                enter_str = f"{enter:.2f}" if enter is not None else "N/A"
                stabilize_str = f"{stabilize:.2f}" if stabilize is not None else "N/A"
                percent_str = f"{percent:.1f}%" if percent is not None else "N/A"
                
                table_data_combined.append([
                    labels[drone],
                    enter_str, stabilize_str, percent_str
                ])
        
        headers = [
            'Drone',
            f'{model}\nEnter (s)', f'{model}\nStabilize (s)', f'{model}\nIn Bounds (%)'
        ]
        print(tabulate(table_data_combined, headers=headers, tablefmt='grid', stralign='center'))


def plot_omega_errors_experiments():
    """
    Create a 2x3 montage showing omega errors.
    Rows: models (modelA, modelC)
    Columns: groups (baseline, gps/Filter1, relative/Filter2)
    """
    n_models = len(models)
    n_groups = len(groups_list)
    x_size = 6 * n_groups
    y_size = 5 * n_models
    fig, axes = plt.subplots(n_models, n_groups, figsize=(x_size, y_size), sharex=True, sharey=True)
    
    # Nominal omega values
    nominal_omegas = {
        '0_2': 0.2,
        '0_4': 0.4,
        '0_6': 0.6,
        '0_8': 0.8
    }
    
    speed = speeds[0]  # Using only one speed for now
    nominal_omega = nominal_omegas[speed]
    
    for i_model, model in enumerate(models):
        for j_group, (group, group_label) in enumerate(zip(groups_list, group_labels)):
            ax = axes[i_model, j_group] if n_models > 1 else axes[j_group]
            
            try:
                print(f"Processing model={model}, group={group}")
                
                # Plot zero error line
                ax.axhline(y=0, color='k', linestyle='-', linewidth=1, alpha=0.9, zorder=1)
                
                # Find CSV files for this group/model/speed
                csv_files = find_csv_files(group, model, speed)
                
                if len(csv_files) == 0:
                    print(f"  Warning: No CSV files found for {group}/{model}/{speed}")
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=10, color='gray')
                else:
                    # Process each drone across all seed runs
                    for drone_idx, drone in enumerate(drones):
                        drone_color = colors[drone_idx]
                        
                        errors_omega_all = []
                        time_references = []
                        
                        for csv_path in csv_files:
                            df = load_and_crop_csv(csv_path)
                            
                            if df is None or len(df) == 0:
                                continue
                            
                            # Get omega column for this drone
                            omega_cols = [col for col in df.columns if drone in col and 'omega' in col.lower()]
                            
                            if len(omega_cols) == 0:
                                continue
                            
                            omega_col = omega_cols[0]
                            
                            # Get time column
                            timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'stamp' in col.lower()]
                            time_col = timestamp_cols[0] if len(timestamp_cols) > 0 else None
                            if time_col is None:
                                continue
                            
                            # Get valid data
                            omega_valid = df[[time_col, omega_col]].dropna()
                            
                            if len(omega_valid) == 0:
                                continue
                            
                            time_data = omega_valid[time_col].values
                            omega_data = omega_valid[omega_col].values
                            error_omega = omega_data - nominal_omega
                            
                            errors_omega_all.append(error_omega)
                            time_references.append(time_data)
                        
                        # Plot aggregated statistics and compute ITAE
                        if len(errors_omega_all) > 0:
                            longest_idx = np.argmax([len(t) for t in time_references])
                            time_common = time_references[longest_idx]
                            
                            errors_interp = [np.interp(time_common, t, e, left=np.nan, right=np.nan) 
                                            for t, e in zip(time_references, errors_omega_all)]
                            stacked = np.array(errors_interp)
                            mean_err = np.nanmean(stacked, axis=0)
                            std_err = np.nanstd(stacked, axis=0)
                            
                            ax.plot(time_common, mean_err, '-', color=drone_color, linewidth=2.0, 
                                   label=f'{labels[drone]}', zorder=3)
                            ax.fill_between(time_common, mean_err - std_err, mean_err + std_err,
                                           color=drone_color, alpha=0.15, zorder=2)
                            
                
                # Configure plot
                ax.grid(True, linestyle=':')
                ax.set_ylim(-0.15, 0.15)
                ax.set_xlim(0, CROP_DURATION)
                
                # Column titles
                if i_model == 0:
                    ax.set_title(group_label, fontweight='bold')
                
                # Row labels (model names)
                if j_group == 0:
                    model_label = model_labels[i_model]
                    ax.set_ylabel(f'{model_label}\n\nError (rad/s)', fontweight='bold')
                
                # Bottom labels
                if i_model == n_models - 1:
                    ax.set_xlabel('Time (s)')
                
                # Legend on top-left cell
                if i_model == 0 and j_group == 0:
                    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
                
            except Exception as e:
                print(f"Error plotting model={model}, group={group}: {e}")
                import traceback
                traceback.print_exc()
                ax.text(0.5, 0.5, f'Error:\\n{str(e)[:50]}', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=8, color='red')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    output_path = plots_dir / 'omega_errors.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Omega errors plot saved to: {output_path}")
    plt.close()


def compute_itae_summary():
    """
    Compute and print ITAE (Integrated Time-weighted Absolute Error) summary table for all models, groups, and drones.
    Time-weighting emphasizes steady-state errors over initial convergence errors.
    """
    nominal_omegas = {
        '0_2': 0.2,
        '0_4': 0.4,
        '0_6': 0.6,
        '0_8': 0.8
    }
    speed = speeds[0]
    nominal_omega = nominal_omegas[speed]
    
    # Create a summary dictionary
    itae_summary = {model: {group: {drone: [] for drone in drones} for group in groups_list} 
                    for model in models}
    
    for model in models:
        for group in groups_list:           
            csv_files = find_csv_files(group, model, speed)
            
            if len(csv_files) == 0:
                print(f"    No CSV files found")
                continue
            
            for drone in drones:
                errors_all = []
                time_refs = []
                
                for csv_file in csv_files:
                    df = load_and_crop_csv(csv_file)
                    
                    if df is None or len(df) == 0:
                        continue
                    
                    # Get omega column
                    omega_cols = [col for col in df.columns if drone in col and 'omega' in col.lower()]
                    
                    if len(omega_cols) == 0:
                        continue
                    
                    omega_col = omega_cols[0]
                    
                    # Get time column
                    timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'stamp' in col.lower()]
                    time_col = timestamp_cols[0] if len(timestamp_cols) > 0 else None
                    if time_col is None:
                        continue
                    
                    # Get valid data
                    omega_valid = df[[time_col, omega_col]].dropna()
                    
                    if len(omega_valid) == 0:
                        continue
                    
                    time_data = omega_valid[time_col].values
                    omega_data = omega_valid[omega_col].values
                    error_omega = omega_data - nominal_omega
                    
                    errors_all.append(error_omega)
                    time_refs.append(time_data)
                
                # Compute mean ITAE across all seeds
                if len(errors_all) > 0:
                    longest_idx = np.argmax([len(t) for t in time_refs])
                    time_common = time_refs[longest_idx]
                    
                    errors_interp = [np.interp(time_common, t, e, left=np.nan, right=np.nan) 
                                    for t, e in zip(time_refs, errors_all)]
                    stacked = np.array(errors_interp)
                    mean_err = np.nanmean(stacked, axis=0)
                    
                    itae = compute_integrated_absolute_error(time_common, mean_err)
                    itae_summary[model][group][drone] = itae
    
    # Print summary table using tabulate
    print("\n" + "=" * 140)
    print("SUMMARY TABLE - ITAE Values (Time-Weighted Absolute Error)")
    print("=" * 140)
    
    for group in groups_list:
        print(f"\n{group.upper()}:")
        
        table_data = []
        for model in models:
            for drone in drones:
                itae = itae_summary[model][group].get(drone, 0)
                
                table_data.append([
                    labels[drone],
                    f"{itae:.4f}"
                ])
        
        headers = [
            'Drone',
        ]
        for model in models:
            model_label = model_labels[models.index(model)]
            headers.extend([
                f'{model_label}'
            ])
        print(tabulate(table_data, headers=headers, 
                      tablefmt='grid', stralign='center'))


def compute_phase_diff_variance_snapshots():
    """
    Compute standard deviation of phase difference errors at time snapshots (20s, 40s, 60s).
    Shows how inter-seed variance evolves over time.
    """
    time_snapshots = [20.0, 40.0, 60.0]
    nominal_phase_diff_deg = 120.0
    speed = speeds[0]
    
    # Store variance data: {model: {group: {drone: {t: {'mean': ..., 'std': ...}}}}}
    variance_summary = {model: {group: {drone: {t: {'mean': None, 'std': None} for t in time_snapshots} 
                                       for drone in drones} 
                               for group in groups_list} 
                       for model in models}
    
    for model in models:
        for group in groups_list:
            for drone in drones:
                csv_files = find_csv_files(group, model, speed)
                csv_files = [f for f in csv_files if 'processed' in f.name]
                
                if len(csv_files) == 0:
                    continue
                
                # Get relationship
                follower, leader = DRONE_RELATIONSHIPS[drone]
                
                phase_diff_errors_all = []
                time_references = []
                
                for csv_file in csv_files:
                    df = load_and_crop_csv(csv_file)
                    
                    if df is None or len(df) == 0:
                        continue
                    
                    # Get time column
                    timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'stamp' in col.lower()]
                    time_col = timestamp_cols[0] if len(timestamp_cols) > 0 else None
                    if time_col is None:
                        continue
                    
                    # Get measured phases
                    ego_meas_col = f"_{drone}_measured_phase"
                    leader_meas_col = f"_{leader}_measured_phase"
                    
                    if not (ego_meas_col in df.columns and leader_meas_col in df.columns):
                        continue
                    
                    # Extract and align phase data
                    phase_leader_data = df[[time_col, leader_meas_col]].dropna().values
                    phase_ego_data = df[[time_col, ego_meas_col]].dropna().values
                    
                    lengths = [len(phase_leader_data), len(phase_ego_data)]
                    min_length = min(lengths) if lengths else 0
                    if min_length == 0:
                        continue
                    
                    phase_leader_data = phase_leader_data[:min_length, :]
                    phase_ego_data = phase_ego_data[:min_length, :]
                    
                    time_ref = phase_ego_data[:, 0]
                    phase_leader = phase_leader_data[:, 1]
                    phase_ego = phase_ego_data[:, 1]
                    
                    # Compute phase difference error
                    phase_diff_leader = wrap_to_pi(phase_leader - phase_ego)
                    phase_diff_deg = np.rad2deg(phase_diff_leader)
                    error_phase_diff = phase_diff_deg - nominal_phase_diff_deg
                    
                    phase_diff_errors_all.append(error_phase_diff)
                    time_references.append(time_ref)
                
                # Compute variance at snapshots
                if len(phase_diff_errors_all) > 0:
                    longest_idx = np.argmax([len(t) for t in time_references])
                    time_common = time_references[longest_idx]
                    
                    errors_interp = [np.interp(time_common, t, e, left=np.nan, right=np.nan) 
                                    for t, e in zip(time_references, phase_diff_errors_all)]
                    stacked = np.array(errors_interp)
                    
                    # Get mean and variance at each snapshot time
                    for t_snap in time_snapshots:
                        # Find closest time index
                        idx_snap = np.argmin(np.abs(time_common - t_snap))
                        mean_at_snap = np.nanmean(stacked[:, idx_snap])
                        std_at_snap = np.nanstd(stacked[:, idx_snap])
                        variance_summary[model][group][drone][t_snap] = {
                            'mean': mean_at_snap,
                            'std': std_at_snap
                        }
    
    # Print summary tables
    print("\n" + "=" * 140)
    print("SUMMARY TABLE - Phase Difference Error Variance (Standard Deviation)")
    print("=" * 140)
    
    for group in groups_list:
        print(f"\n{group.upper()}:")
        
        table_data = []
        for model in models:
            for drone in drones:
                std_20 = variance_summary[model][group][drone][20.0]
                std_40 = variance_summary[model][group][drone][40.0]
                std_60 = variance_summary[model][group][drone][60.0]
                
                # Format as "mean ± std" for each snapshot
                def fmt_error_std(data_dict):
                    if data_dict is None or data_dict['mean'] is None:
                        return "N/A"
                    return f"{data_dict['mean']:+.2f}°±{data_dict['std']:.2f}°"
                
                table_data.append([
                    labels[drone],
                    fmt_error_std(std_20),
                    fmt_error_std(std_40),
                    fmt_error_std(std_60),
                ])
        
        headers = [
            'Drone',
        ]
        for model in models:
            model_label = model_labels[models.index(model)]
            headers.extend([
                f'{model_label}\n20s', f'{model_label}\n40s', f'{model_label}\n60s'
            ])
        print(tabulate(table_data, headers=headers, tablefmt='grid', stralign='center'))


if __name__ == "__main__":
    print("\n" + "=" * 140)
    print("PLOTTING PHASE DIFFERENCE ERRORS - EXPERIMENTS")
    print("=" * 140)
    plot_phases_differences_errors_experiments()

    print("\n" + "=" * 140)
    print("PLOTTING OMEGA ERRORS - EXPERIMENTS")
    print("=" * 140)
    plot_omega_errors_experiments()
    
    # Summary of metrics
    compute_itae_summary()
    compute_phase_diff_settling_summary()
    compute_phase_diff_variance_snapshots()