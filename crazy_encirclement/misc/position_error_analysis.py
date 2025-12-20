"""
Position Error Analysis Module

Computes position errors (x, y, z) by comparing Vicon measurements with theoretical
encirclement trajectories. The theoretical trajectory is computed using the embedding
model with initial phases estimated from the measured data.

Key parameters:
- dt = 0.01 (time step in seconds)
- k_phi = 10.0 (controller gain)
- Initial phases: averaged across all experiments for each drone
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

plt.rcParams.update({'text.usetex': True, 'font.size': 20, 'figure.dpi': 150})

# Import helpers
from crazy_encirclement.filters import (
    build_Re,
    wrap_to_pi,
    wrap_to_2pi,
    exp_SO3,
    orthonormalize,
    get_phase,
    build_Rc,
    REGISTRED_OMEGA_FUNCTIONS as REGISTERED_OMEGA_FUNCTIONS,
    omega_func_modelA,
    omega_func_modelC,
)

# Configuration
base_dir = Path('/home/paulo/Documents/k_10/')
plots_dir = base_dir / 'plots'
plots_dir.mkdir(exist_ok=True)

groups = ['baseline', 'gps', 'relative']
models = ['modelA', 'modelC']
model_labels = ['Model A', 'Model B']
speeds = ['0_2']
drones = ['C14', 'C05', 'C04']
colormap_name = 'gist_rainbow'

labels = {
    'C04': 'Quadcopter 3',
    'C05': 'Quadcopter 2',
    'C14': 'Quadcopter 1'
}

# Global color mapping: drones[0] -> colors[0], drones[1] -> colors[1], drones[2] -> colors[2]
cmap = plt.get_cmap(colormap_name)
colors = [cmap(i) for i in [0.125, 0.65, 0.9]]  # [Quadcopter 1, Quadcopter 2, Quadcopter 3]

# Analysis parameters
DT = 0.01  # Time step (seconds)
K_PHI = 10.0  # Controller gain
RADIUS_NOMINAL = 1.0  # Nominal radius (meters)
CROP_DURATION = 60.0  # Crop duration after encircle flag (seconds)

# Drone relationships: ego -> (follower, leader)
DRONE_RELATIONSHIPS = {
    'C05': ('C14', 'C04'),
    'C04': ('C05', 'C14'),
    'C14': ('C04', 'C05')
}


def find_csv_files(group, model, speed):
    """Find CSV files for a given group, model, and speed."""
    search_path = base_dir / group / model / speed
    
    if not search_path.exists():
        print(f"Warning: Path does not exist: {search_path}")
        return []
    
    csv_files = []
    seed_folders = [f"seed_{s}" for s in [40, 45, 50, 55, 60]]
    
    for seed_folder in seed_folders:
        seed_path = search_path / seed_folder
        if seed_path.exists():
            csv_files.extend(list(seed_path.glob('*.csv')))
    
    return sorted(csv_files)


def load_and_crop_csv(csv_path, crop_duration=CROP_DURATION):
    """Load and crop CSV from encircle flag or first omega entry."""
    try:
        df = pd.read_csv(csv_path)
        start_idx = None
        
        # Try _encircle_data column
        if '_encircle_data' in df.columns:
            encircle_idx = df[df['_encircle_data'] == True].index
            if len(encircle_idx) > 0:
                start_idx = encircle_idx[0]
        
        # Fallback to first filtered omega
        if start_idx is None:
            omega_cols = [col for col in df.columns if 'omega' in col.lower() and 'filtered' in col.lower()]
            if len(omega_cols) == 0:
                return None
            
            for omega_col in omega_cols:
                non_nan_idx = df[omega_col].dropna().index
                if len(non_nan_idx) > 0:
                    if start_idx is None or non_nan_idx[0] < start_idx:
                        start_idx = non_nan_idx[0]
        
        if start_idx is None:
            return None
        
        # Get timestamp column
        timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'stamp' in col.lower()]
        if len(timestamp_cols) == 0:
            return df.iloc[start_idx:start_idx + int(crop_duration * 100)]
        
        time_col = timestamp_cols[0]
        start_time = df.loc[start_idx, time_col]
        end_time = start_time + crop_duration
        
        cropped_df = df[(df[time_col] >= start_time) & (df[time_col] <= end_time)].copy()
        cropped_df[time_col] = cropped_df[time_col] - start_time
        
        return cropped_df
    
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None


def _find_vicon_cols_for_drone(df, drone):
    """Find Vicon x, y, z position columns for a drone."""
    x_col = None
    y_col = None
    z_col = None
    
    for col in df.columns:
        if drone in col and 'vicon_position' in col.lower():
            if 'pos_x' in col.lower():
                x_col = col
            elif 'pos_y' in col.lower():
                y_col = col
            elif 'pos_z' in col.lower():
                z_col = col
    
    return x_col, y_col, z_col


def simulate_theoretical_trajectory(initial_phase, omega, dt, duration, embedding_fn, 
                                   z_offset=1.0, radius=RADIUS_NOMINAL):
    """
    Simulate theoretical trajectory for a drone.
    
    Args:
        initial_phase: Starting phase angle (radians)
        omega: Angular velocity (rad/s)
        dt: Time step (seconds)
        duration: Simulation duration (seconds)
        embedding_fn: Embedding function (omega_func_modelA or omega_func_modelC)
        z_offset: Vertical offset (meters)
        radius: Nominal radius (meters)
    
    Returns:
        trajectory: Nx3 array of (x, y, z) positions
        phases: N array of phases
        times: N array of times
    """
    num_steps = int(duration / dt)
    trajectory = np.zeros((num_steps, 3))
    phases = np.zeros(num_steps)
    times = np.zeros(num_steps)
    
    # Initialize
    e_x = np.asarray([[1.], [0.], [0.]])
    Rc = build_Rc(initial_phase)
    
    for i in range(num_steps):
        # Get current phase
        phase = get_phase(Rc)
        phases[i] = phase
        times[i] = i * dt
        
        # Compute embedding rotation
        Re = build_Re(embedding_fn, phase)
        
        # Compute 3D position
        q = (Re @ Rc @ (e_x * radius)).flatten()
        q[2] += z_offset
        trajectory[i] = q
        
        # Update phase
        Rc = exp_SO3(np.asarray([0., 0., omega * dt])) @ Rc
        Rc = orthonormalize(Rc)
    
    return trajectory, phases, times


def compute_initial_phases_mean(group, model, speed, drones_list=None):
    """
    Compute mean initial phases from all experiments for each drone.
    
    Returns:
        dict: {drone: initial_phase_rad}
    """
    if drones_list is None:
        drones_list = drones
    
    initial_phases = {drone: [] for drone in drones_list}
    
    csv_files = find_csv_files(group, model, speed)
    csv_files = [f for f in csv_files if 'processed' in f.name]
    
    for csv_file in csv_files:
        df = load_and_crop_csv(csv_file)
        if df is None or len(df) == 0:
            continue
        
        for drone in drones_list:
            meas_col = f"_{drone}_measured_phase"
            if meas_col in df.columns:
                phase_data = df[meas_col].dropna()
                if len(phase_data) > 0:
                    # Use first few samples to get initial phase
                    initial_phase = float(phase_data.iloc[0])
                    initial_phases[drone].append(initial_phase)
    
    # Compute mean initial phase for each drone
    mean_phases = {}
    for drone in drones_list:
        if len(initial_phases[drone]) > 0:
            phases_wrapped = [wrap_to_2pi(p) for p in initial_phases[drone]]
            # Compute circular mean
            mean_sin = np.mean(np.sin(phases_wrapped))
            mean_cos = np.mean(np.cos(phases_wrapped))
            mean_phases[drone] = np.arctan2(mean_sin, mean_cos)
            mean_phases[drone] = wrap_to_2pi(mean_phases[drone])
        else:
            mean_phases[drone] = 0.0
    
    return mean_phases


def compute_position_errors_single_run(csv_file, group, model, embedding_fn, z_offset, 
                                      initial_phases_mean, omega=0.2):
    """
    Compute position errors for a single experiment run.
    
    Returns:
        dict: {drone: {
                'time': times,
                'vicon_xyz': measured positions,
                'theoretical_xyz': theoretical positions,
                'error_xyz': position errors,
                'error_magnitude': sqrt(ex^2 + ey^2 + ez^2)
              }}
    """
    df = load_and_crop_csv(csv_file)
    if df is None or len(df) == 0:
        return None
    
    # Get time column
    timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'stamp' in col.lower()]
    time_col = timestamp_cols[0] if len(timestamp_cols) > 0 else None
    if time_col is None:
        return None
    
    time_data = df[time_col].values
    duration = float(time_data[-1]) if len(time_data) > 0 else CROP_DURATION
    
    results = {}
    
    for drone in drones:
        # Find Vicon columns
        x_col, y_col, z_col = _find_vicon_cols_for_drone(df, drone)
        if x_col is None or y_col is None:
            continue
        
        # Extract Vicon data
        vicon_valid = df[[time_col, x_col, y_col]].dropna()
        if z_col is not None:
            vicon_valid = df[[time_col, x_col, y_col, z_col]].dropna()
        
        if len(vicon_valid) == 0:
            continue
        
        vicon_times = vicon_valid[time_col].values
        vicon_x = vicon_valid[x_col].values
        vicon_y = vicon_valid[y_col].values
        vicon_z = vicon_valid[z_col].values if z_col is not None else np.zeros_like(vicon_x)
        
        # Simulate theoretical trajectory
        initial_phase = wrap_to_2pi(initial_phases_mean.get(drone, 0.0))
        theory_traj, theory_phases, theory_times = simulate_theoretical_trajectory(
            initial_phase, omega, DT, duration, embedding_fn, z_offset=z_offset
        )
        
        # Interpolate theoretical trajectory to match Vicon times
        theory_x_interp = np.interp(vicon_times, theory_times, theory_traj[:, 0])
        theory_y_interp = np.interp(vicon_times, theory_times, theory_traj[:, 1])
        theory_z_interp = np.interp(vicon_times, theory_times, theory_traj[:, 2])
        
        # Compute errors
        error_x = vicon_x - theory_x_interp
        error_y = vicon_y - theory_y_interp
        error_z = vicon_z - theory_z_interp
        error_mag = np.sqrt(error_x**2 + error_y**2 + error_z**2)
        
        results[drone] = {
            'time': vicon_times,
            'vicon_xyz': np.column_stack([vicon_x, vicon_y, vicon_z]),
            'theoretical_xyz': np.column_stack([theory_x_interp, theory_y_interp, theory_z_interp]),
            'error_xyz': np.column_stack([error_x, error_y, error_z]),
            'error_magnitude': error_mag
        }
    
    return results


def plot_position_errors_2x3(group_list=None, model_list=None, speed='0_2'):
    """
    Create 2x3 position error plots across groups and models.
    Rows: models
    Columns: groups (baseline, gps, relative)
    """
    if group_list is None:
        group_list = ['baseline', 'gps', 'relative']
    if model_list is None:
        model_list = ['modelA', 'modelC']
    
    n_models = len(model_list)
    n_groups = len(group_list)
    
    fig, axes = plt.subplots(n_models, n_groups, figsize=(18, 8), sharex=True, sharey=True)
    
    for i_model, model in enumerate(model_list):
        # Determine z_offset and embedding function
        z_offset = 0.8 if model == 'modelA' else 1.0
        embedding_fn = omega_func_modelA if model == 'modelA' else omega_func_modelC
        
        # Compute mean initial phases for this model
        mean_phases_dict = {}
        for group in group_list:
            mean_phases_dict[group] = compute_initial_phases_mean(group, model, speed)
        
        for j_group, group in enumerate(group_list):
            ax = axes[i_model, j_group] if n_models > 1 else axes[j_group]
            
            initial_phases_mean = mean_phases_dict[group]
            
            # Find all CSV files for this group/model
            csv_files = find_csv_files(group, model, speed)
            
            if len(csv_files) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=10, color='gray')
                continue
            
            # Process each drone and aggregate errors
            for drone_idx, drone in enumerate(drones):
                drone_color = colors[drone_idx]
                
                error_mags_all = []
                time_refs_all = []
                
                for csv_file in csv_files:
                    result = compute_position_errors_single_run(
                        csv_file, group, model, embedding_fn, z_offset,
                        initial_phases_mean, omega=float(speed.replace('_', '.'))
                    )
                    
                    if result is None or drone not in result:
                        continue
                    
                    error_mags_all.append(result[drone]['error_magnitude'])
                    time_refs_all.append(result[drone]['time'])
                
                # Plot aggregated statistics
                if len(error_mags_all) > 0:
                    longest_idx = np.argmax([len(t) for t in time_refs_all])
                    time_common = time_refs_all[longest_idx]
                    
                    errors_interp = [np.interp(time_common, t, e, left=np.nan, right=np.nan)
                                    for t, e in zip(time_refs_all, error_mags_all)]
                    stacked = np.array(errors_interp)
                    mean_err = np.nanmean(stacked, axis=0)
                    std_err = np.nanstd(stacked, axis=0)
                    
                    ax.plot(time_common, mean_err, '-', color=drone_color, linewidth=2.0,
                           label=labels[drone], zorder=3)
                    ax.fill_between(time_common, mean_err - std_err, mean_err + std_err,
                                   color=drone_color, alpha=0.15, zorder=2)
            
            # Configure plot
            ax.grid(True, linestyle=':')
            # ax.set_ylim(0, 0.5)  # Typical position error range (meters)
            ax.set_xlim(0, CROP_DURATION)
            
            # Column titles
            group_labels_map = {'baseline': 'Baseline', 'gps': 'Filter 1 (GPS)', 'relative': 'Filter 2 (Relative)'}
            if i_model == 0:
                ax.set_title(group_labels_map.get(group, group), fontweight='bold', fontsize=12)
            
            # Row labels
            if j_group == 0:
                model_label = model_labels[i_model]
                ax.set_ylabel(f'{model_label}\n\nError Mag (m)', fontsize=11)
            
            # Bottom labels
            if i_model == n_models - 1:
                ax.set_xlabel('Time (s)', fontsize=11)
            
            # Legend
            if i_model == 0 and j_group == 0:
                ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    output_path = plots_dir / 'position_errors_2x3.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPosition errors plot saved to: {output_path}")
    plt.close()


def plot_xyz_component_errors_2x3(group_list=None, model_list=None, speed='0_2', component='x'):
    """
    Plot individual position error components (x, y, or z).
    """
    if group_list is None:
        group_list = ['baseline', 'gps', 'relative']
    if model_list is None:
        model_list = ['modelA', 'modelC']
    
    n_models = len(model_list)
    n_groups = len(group_list)
    
    fig, axes = plt.subplots(n_models, n_groups, figsize=(18, 8), sharex=True, sharey=True)
    
    component_idx = {'x': 0, 'y': 1, 'z': 2}.get(component, 0)
    component_label = {'x': 'X', 'y': 'Y', 'z': 'Z'}.get(component, 'X')
    
    for i_model, model in enumerate(model_list):
        z_offset = 0.8 if model == 'modelA' else 1.0
        embedding_fn = omega_func_modelA if model == 'modelA' else omega_func_modelC
        
        mean_phases_dict = {}
        for group in group_list:
            mean_phases_dict[group] = compute_initial_phases_mean(group, model, speed)
        
        for j_group, group in enumerate(group_list):
            ax = axes[i_model, j_group] if n_models > 1 else axes[j_group]
            
            initial_phases_mean = mean_phases_dict[group]
            csv_files = find_csv_files(group, model, speed)
            
            if len(csv_files) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=10, color='gray')
                continue
            
            for drone_idx, drone in enumerate(drones):
                drone_color = colors[drone_idx]
                
                errors_component_all = []
                time_refs_all = []
                
                for csv_file in csv_files:
                    result = compute_position_errors_single_run(
                        csv_file, group, model, embedding_fn, z_offset,
                        initial_phases_mean, omega=float(speed.replace('_', '.'))
                    )
                    
                    if result is None or drone not in result:
                        continue
                    
                    component_error = result[drone]['error_xyz'][:, component_idx]
                    errors_component_all.append(component_error)
                    time_refs_all.append(result[drone]['time'])
                
                if len(errors_component_all) > 0:
                    longest_idx = np.argmax([len(t) for t in time_refs_all])
                    time_common = time_refs_all[longest_idx]
                    
                    errors_interp = [np.interp(time_common, t, e, left=np.nan, right=np.nan)
                                    for t, e in zip(time_refs_all, errors_component_all)]
                    stacked = np.array(errors_interp)
                    mean_err = np.nanmean(stacked, axis=0)
                    std_err = np.nanstd(stacked, axis=0)
                    
                    ax.plot(time_common, mean_err, '-', color=drone_color, linewidth=2.0,
                           label=labels[drone], zorder=3)
                    ax.fill_between(time_common, mean_err - std_err, mean_err + std_err,
                                   color=drone_color, alpha=0.15, zorder=2)
            
            # Configure plot
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
            ax.grid(True, linestyle=':')
            # ax.set_ylim(-0.3, 0.3)  # Allows both positive and negative errors
            ax.set_xlim(0, CROP_DURATION)
            
            group_labels_map = {'baseline': 'Baseline', 'gps': 'Filter 1 (GPS)', 'relative': 'Filter 2 (Relative)'}
            if i_model == 0:
                ax.set_title(group_labels_map.get(group, group), fontweight='bold', fontsize=12)
            
            if j_group == 0:
                model_label = model_labels[i_model]
                ax.set_ylabel(f'{model_label}\n\nError {component_label} (m)', fontsize=11)
            
            if i_model == n_models - 1:
                ax.set_xlabel('Time (s)', fontsize=11)
            
            if i_model == 0 and j_group == 0:
                ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    output_path = plots_dir / f'position_error_{component}_2x3.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPosition error {component_label} plot saved to: {output_path}")
    plt.close()


def plot_xyz_trajectory_components_2x3(group_list=None, model_list=None, speed='0_2', component='x'):
    """
    Plot XYZ trajectory components (measured vs theoretical) side-by-side.
    Rows: models
    Columns: groups (baseline, gps, relative)
    """
    if group_list is None:
        group_list = ['baseline', 'gps', 'relative']
    if model_list is None:
        model_list = ['modelA', 'modelC']
    
    n_models = len(model_list)
    n_groups = len(group_list)
    
    fig, axes = plt.subplots(n_models, n_groups, figsize=(18, 8), sharex=True, sharey=True)
    
    component_idx = {'x': 0, 'y': 1, 'z': 2}.get(component, 0)
    component_label = {'x': 'X', 'y': 'Y', 'z': 'Z'}.get(component, 'X')
    
    for i_model, model in enumerate(model_list):
        z_offset = 0.8 if model == 'modelA' else 1.0
        embedding_fn = omega_func_modelA if model == 'modelA' else omega_func_modelC
        
        mean_phases_dict = {}
        for group in group_list:
            mean_phases_dict[group] = compute_initial_phases_mean(group, model, speed)
        
        for j_group, group in enumerate(group_list):
            ax = axes[i_model, j_group] if n_models > 1 else axes[j_group]
            
            initial_phases_mean = mean_phases_dict[group]
            csv_files = find_csv_files(group, model, speed)
            
            if len(csv_files) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=10, color='gray')
                continue
            
            # Process each drone
            for drone_idx, drone in enumerate(drones):
                drone_color = colors[drone_idx]
                
                vicon_all = []
                theory_all = []
                time_refs_all = []
                
                for csv_file in csv_files:
                    result = compute_position_errors_single_run(
                        csv_file, group, model, embedding_fn, z_offset,
                        initial_phases_mean, omega=float(speed.replace('_', '.'))
                    )
                    
                    if result is None or drone not in result:
                        continue
                    
                    vicon_all.append(result[drone]['vicon_xyz'][:, component_idx])
                    theory_all.append(result[drone]['theoretical_xyz'][:, component_idx])
                    time_refs_all.append(result[drone]['time'])
                
                # Plot aggregated statistics
                if len(vicon_all) > 0:
                    longest_idx = np.argmax([len(t) for t in time_refs_all])
                    time_common = time_refs_all[longest_idx]
                    
                    # Interpolate vicon and theory to common time
                    vicon_interp = [np.interp(time_common, t, v, left=np.nan, right=np.nan)
                                   for t, v in zip(time_refs_all, vicon_all)]
                    theory_interp = [np.interp(time_common, t, th, left=np.nan, right=np.nan)
                                    for t, th in zip(time_refs_all, theory_all)]
                    
                    vicon_stacked = np.array(vicon_interp)
                    theory_stacked = np.array(theory_interp)
                    
                    vicon_mean = np.nanmean(vicon_stacked, axis=0)
                    vicon_std = np.nanstd(vicon_stacked, axis=0)
                    theory_mean = np.nanmean(theory_stacked, axis=0)
                    theory_std = np.nanstd(theory_stacked, axis=0)
                    
                    # Plot measured (solid line)
                    ax.plot(time_common, vicon_mean, '-', color=drone_color, linewidth=2.0,
                           label=f'{labels[drone]} (measured)', zorder=3)
                    ax.fill_between(time_common, vicon_mean - vicon_std, vicon_mean + vicon_std,
                                   color=drone_color, alpha=0.1, zorder=1)
                    
                    # Plot theoretical (dashed line)
                    ax.plot(time_common, theory_mean, '--', color=drone_color, linewidth=2.0,
                           label=f'{labels[drone]} (theory)', zorder=2)
                    ax.fill_between(time_common, theory_mean - theory_std, theory_mean + theory_std,
                                   color=drone_color, alpha=0.1, zorder=1)
            
            # Configure plot
            ax.grid(True, linestyle=':')
            ax.set_xlim(0, CROP_DURATION)
            
            group_labels_map = {'baseline': 'Baseline', 'gps': 'Filter 1 (GPS)', 'relative': 'Filter 2 (Relative)'}
            if i_model == 0:
                ax.set_title(group_labels_map.get(group, group), fontweight='bold', fontsize=12)
            
            if j_group == 0:
                model_label = model_labels[i_model]
                ax.set_ylabel(f'{model_label}\n\nPosition {component_label} (m)', fontsize=11)
            
            if i_model == n_models - 1:
                ax.set_xlabel('Time (s)', fontsize=11)
            
            if i_model == 0 and j_group == 0:
                ax.legend(loc='best', fontsize=9, framealpha=0.9, ncol=2)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    output_path = plots_dir / f'trajectory_{component}_components_2x3.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nTrajectory {component_label} component plot saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("POSITION ERROR ANALYSIS")
    print("=" * 80)
    print(f"Parameters: dt={DT}, k_phi={K_PHI}, radius={RADIUS_NOMINAL} m")
    
    # print("\nGenerating position error magnitude plots...")
    # plot_position_errors_2x3()
    
    # print("\nGenerating position error component plots...")
    # for component in ['x', 'y', 'z']:
    #     plot_xyz_component_errors_2x3(component=component)
    
    print("\nGenerating XYZ trajectory component plots (measured vs theoretical)...")
    for component in ['x', 'y', 'z']:
        plot_xyz_trajectory_components_2x3(component=component)
    
    print("\n" + "=" * 80)
    print("Position error analysis complete!")
    print("=" * 80)
