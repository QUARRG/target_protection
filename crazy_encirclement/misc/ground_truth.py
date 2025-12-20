import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import optimize
import warnings

from crazy_encirclement.filters import (
    build_Re,
    wrap_to_pi,
    wrap_to_2pi,
    phase_controller,
    omega_func_modelA,
    omega_func_modelC,
)

warnings.filterwarnings('ignore')

plt.rcParams.update({'text.usetex': True, 'font.size': 20, 'figure.dpi': 150})

# Configuration
base_dir = Path('/home/paulo/Documents/k_10/')
plots_dir = base_dir / 'plots'
plots_dir.mkdir(exist_ok=True)
groups = ['baseline',
          'gps',
          'relative',
        #   'combined',
        #   'combined_wind_mild',
          'combined_wind_strong',
        #   'total_outage',
        #   'total_outage_wind_mild',
          'total_outage_wind_strong']
models = ['modelA', 'modelC']
speeds = ['0_2']
k_phi = 10.0
radius_nominal = 1.0  # meters
drones = ['C14', 'C05', 'C04']
colormap_name = 'gist_rainbow'  # Can be changed to: 'plasma', 'inferno', 'magma', 'cividis', 'tab10', etc.
labels = {
    'C04': 'Quadcopter 3',
    'C05': 'Quadcopter 2',
    'C14': 'Quadcopter 1'
}

# Duration to crop after encircle flag (in seconds)
CROP_DURATION = 90.0

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
            # Exclude any previously generated processed.csv to avoid re-processing
            csv_files.extend([p for p in seed_path.glob('*.csv') if p.name != 'processed.csv'])
    
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
            
            # if len(omega_cols) == 0:
            #     # Try without 'filtered' keyword
            #     omega_cols = [col for col in df.columns if 'omega' in col.lower()]
            
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
        end_time = start_time + crop_duration
        
        # Crop the dataframe
        cropped_df = df[(df[time_col] >= start_time) & (df[time_col] <= end_time)].copy()
        
        # Reset time to start from 0
        cropped_df[time_col] = cropped_df[time_col] - start_time
        
        return cropped_df
    
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None


def _find_vicon_cols_for_drone(df, drone):
    """Try to find x,y,z vicon columns for a given drone in dataframe.

    Returns (x_col, y_col, z_col) where any can be None if not found.
    """
    cols = df.columns.tolist()

    def find_col(containing):
        for c in cols:
            low = c.lower()
            if drone.lower() in low and all(p in low for p in containing):
                return c
        return None

    # Common patterns used in other scripts: 'vicon_position_pos_x'
    x_col = find_col(['vicon', 'pos', 'x']) or find_col(['vicon', '_x']) or find_col(['_x'])
    y_col = find_col(['vicon', 'pos', 'y']) or find_col(['vicon', '_y']) or find_col(['_y'])
    z_col = find_col(['vicon', 'pos', 'z']) or find_col(['vicon', '_z']) or find_col(['_z'])

    return x_col, y_col, z_col


def objective(theta, q_meas, radius, embed_fn, reg, prev):
    '''Objective function for phase optimization.'''
    # ensure scalar
    th = float(theta)
    Re = build_Re(embed_fn, th)
    p = np.array([radius * np.cos(th), radius * np.sin(th), 0.0])
    q_est = Re.dot(p)
    err = np.linalg.norm(q_meas - q_est)
    # regularize deviation from previous phase (wrap to [-pi,pi])
    if reg > 0.0:
        d = np.arctan2(np.sin(th - prev), np.cos(th - prev))
        err = np.sqrt(err**2 + (reg * d)**2)
    return err
    

def compute_measured_phases(csv_path, z_offset, embedding_fn=None):
    """Compute measured phases from vicon positions and save processed CSV.

    Algorithm per drone:
    - Find vicon x,y,z columns for the drone in the CSV.
    - Compute initial phase from the first valid (x,y) point: phi0 = atan2(y0, x0).
    - Build rotation matrix Re = Rz(phi0).
    - For every 3D point q = [x,y,z]^T, compute p = Re.T @ q and phase = wrap_to_2pi(atan2(p[1], p[0])).
    - Store result in column '{drone}_measured_phase' (radians in [0,2pi)).

    The function writes a file named 'processed.csv' in the same folder as `csv_path`.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return False

    df = load_and_crop_csv(csv_path)
    if df is None or len(df) == 0:
        print(f"No data in {csv_path}")
        return False

    for drone in drones:
        measured_col = f"_{drone}_measured_phase"
        df[measured_col] = np.nan  # Initialize column

        # find timestamp column
        timestamp_cols = [c for c in df.columns if 'time' in c.lower() or 'stamp' in c.lower()]
        time_col = timestamp_cols[0] if len(timestamp_cols) > 0 else None

        # find vicon columns for this drone
        x_col, y_col, z_col = _find_vicon_cols_for_drone(df, drone)
        if x_col is None or y_col is None:
            print(f"Skipping {drone}: missing vicon x/y columns in {csv_path.name}")
            continue

        # build positions dataframe with rows where x and y are present
        cols_to_take = [x_col, y_col]
        if z_col is not None:
            cols_to_take.append(z_col)

        positions_df = df.loc[df[x_col].notna() & df[y_col].notna(), cols_to_take].copy()
        if positions_df.shape[0] == 0:
            print(f"No valid vicon samples for {drone} in {csv_path.name}")
            continue

        # initial phase from first valid point
        first_row = positions_df.iloc[0]
        prev_phase = np.arctan2(float(first_row[y_col]), float(first_row[x_col]))

        # store results in a dict keyed by original index
        phases_by_index = {}
        phases_by_index[positions_df.index[0]] = wrap_to_2pi(prev_phase)

        # optimizer regularization weight
        reg_weight = 0.0

        # iterate over subsequent rows and optimize phase
        for idx, row in positions_df.iloc[1:].iterrows():
            x_meas = float(row[x_col]); y_meas = float(row[y_col]); z_meas = float(row[z_col]) - z_offset if z_col is not None else 0.0
            q_meas = np.array([x_meas, y_meas, z_meas])

            # use a scalar bounded optimizer around previous phase to avoid costly gradients
            try:
                lower = prev_phase - 0.087 # ~5 degrees
                upper = prev_phase + 0.087 # ~5 degrees
                res = optimize.minimize_scalar(
                    lambda th: objective(theta=th, q_meas=q_meas, radius=radius_nominal, embed_fn=embedding_fn, reg=reg_weight, prev=prev_phase),
                    bounds=(lower, upper), method='bounded', options={'xatol': 1e-4}
                )
                theta_opt = float(res.x) if (hasattr(res, 'x') and res.success) else prev_phase
            except Exception as e:
                print(f"Optimizer error for {drone} at file {csv_path.name} index {idx}: {e}")
                theta_opt = prev_phase

            phases_by_index[idx] = wrap_to_2pi(theta_opt)
            prev_phase = theta_opt

        # assign computed phases back into original dataframe by index
        indices = sorted(phases_by_index.keys())
        df.loc[indices, measured_col] = [phases_by_index[i] for i in indices]
        # print(f"Computed measured phases for drone {drone} ({len(indices)} samples) in {csv_path.name}")

    # Save processed csv
    out_dir = csv_path.parent
    out_file = out_dir / 'processed.csv'
    try:
        df.to_csv(out_file, index=False)
        print(f"      Saved processed CSV to: {out_file}")
    except Exception as e:
        print(f"      Error saving processed CSV: {e}")
        return False

    return True


def process_all_experiments(groups_to_process=None, models_to_process=None, speeds_to_process=None, force=False):
    """Process all CSV files found under base_dir for selected groups/models/speeds.

    By default processes all configured groups/models/speeds.
    """
    groups_sel = groups_to_process if groups_to_process is not None else groups
    models_sel = models_to_process if models_to_process is not None else models
    speeds_sel = speeds_to_process if speeds_to_process is not None else speeds

    for group in groups_sel:
        print(f"\nProcessing group: {group}")
        for model in models_sel:
            print(f"  Model: {model}")
            embedding_fn = omega_func_modelA if model == 'modelA' else omega_func_modelC
            z_offset = 0.8 if model == 'modelA' else 1.0
            for speed in speeds_sel:
                print(f"    Speed: {speed}")
                csv_files = find_csv_files(group, model, speed)
                for csv in csv_files:
                    compute_measured_phases(csv, z_offset, embedding_fn)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("PROCESSING ALL EXPERIMENTS TO COMPUTE MEASURED PHASES")
    print("=" * 80)
    process_all_experiments()




