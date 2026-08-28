import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MultipleLocator
import sys
from pathlib import Path

# Add parent directory to path to import from filters.py
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import functions from filters.py
from filters import (
    REGISTRED_OMEGA_FUNCTIONS as REGISTERED_OMEGA_FUNCTIONS,
    exp_SO3,
    orthonormalize,
    build_Rc,
    build_Re,
    get_phase
)

plt.rcParams.update({'text.usetex': True, 'font.size': 24, 'figure.dpi': 150})
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Trajectory Simulation
# ----------------------------------------------------------------------
def simulate_embedding_trajectory(model_name: str, omega: float = 0.2, dt: float = 0.01, 
                                   radius: float = 2.0, duration: float = None, z_offset: float = 1.0):
    """
    Simulate the theoretical 3D trajectory for a given embedding model.
    
    Args:
        model_name: Name of the model ('modelA', 'modelB', 'modelC', 'modelD', 'modelE')
        omega: Angular velocity (rad/s)
        dt: Time step (seconds)
        radius: Nominal radius of the trajectory
        duration: Simulation duration (seconds). If None, computes one full loop (2*pi/omega)
        z_offset: Vertical offset to lift trajectory from ground (meters)
    
    Returns:
        trajectory: Nx3 array of (x, y, z) positions
        phases: N array of phase angles
        times: N array of time values
    """
    if model_name not in REGISTERED_OMEGA_FUNCTIONS:
        raise ValueError(f"Model '{model_name}' not found. Choose from: {list(REGISTERED_OMEGA_FUNCTIONS.keys())}")
    
    embedding_fn = REGISTERED_OMEGA_FUNCTIONS[model_name]
    
    # Compute duration for one full loop if not specified
    if duration is None:
        duration = 2 * np.pi / omega
    
    # Initialize
    e_x = np.asarray([[1.], [0.], [0.]])
    Rc = build_Rc(0.0)  # Start at phase = 0
    
    # Simulation arrays
    num_steps = int(duration / dt)
    trajectory = np.zeros((num_steps, 3))
    phases = np.zeros(num_steps)
    times = np.zeros(num_steps)
    
    # Simulate
    for i in range(num_steps):
        # Get current phase
        phase = get_phase(Rc)
        phases[i] = phase
        times[i] = i * dt
        
        # Compute embedding rotation
        Re = build_Re(embedding_fn, phase)
        
        # Compute 3D position: q = Re @ Rc @ (e_x * radius) + z_offset
        q = (Re @ Rc @ (e_x * radius)).flatten()
        q[2] += z_offset  # Add vertical offset
        trajectory[i] = q
        
        # Update phase (prediction step)
        Rc = exp_SO3(np.asarray([0., 0., omega * dt])) @ Rc
        Rc = orthonormalize(Rc)
    
    return trajectory, phases, times


def plot_embedding_trajectory_3d(model_name: str, omega: float = 0.2, dt: float = 0.01,
                                   radius: float = 2.0, duration: float = None, z_offset: float = 1.0,
                                   show_projection: bool = True, show_phase_color: bool = True):
    """
    Plot the theoretical 3D trajectory for a given embedding model.
    
    Args:
        model_name: Name of the model ('modelA', 'modelB', 'modelC', 'modelD', 'modelE')
        omega: Angular velocity (rad/s)
        dt: Time step (seconds)
        radius: Nominal radius of the trajectory
        duration: Simulation duration (seconds). If None, computes one full loop
        z_offset: Vertical offset to lift trajectory from ground (meters)
        show_projection: Whether to show projections on XY, XZ, YZ planes
        show_phase_color: Whether to color the trajectory by phase
    """
    # Simulate trajectory
    trajectory, phases, times = simulate_embedding_trajectory(
        model_name, omega, dt, radius, duration, z_offset
    )
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory
    if show_phase_color:
        # Color by phase
        scatter = ax.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                           c=np.degrees(phases), cmap='hsv', s=20, alpha=0.8)
        cbar = plt.colorbar(scatter, ax=ax, pad=0.02, shrink=0.8)
        cbar.ax.set_title('Phase (deg)', pad=10)
    else:
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
               'b-', linewidth=1.5, alpha=0.8)
    
    # Mark start and end points
    ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2],
              c='black', s=100, marker='o', label='Start/End', edgecolors='black', linewidths=2, zorder=10)
    # ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2],
    #           c='red', s=100, marker='s', label='End', edgecolors='black', linewidths=2, zorder=10)
    
    # Labels and title
    ax.set_xlabel('x (m)', labelpad=15)
    ax.set_ylabel('y (m)', labelpad=15)
    ax.set_zlabel('z (m)', labelpad=15)
    # ax.set_title(f'Theoretical 3D Trajectory - {model_name}\n'
    #             f'ω = {omega} rad/s, radius = {radius} m', 
    #             fontsize=14, fontweight='bold', pad=20)
    
    # Set orthographic projection
    ax.set_proj_type('ortho')
    
    # Equal aspect ratio
    max_range = np.array([trajectory[:, 0].max() - trajectory[:, 0].min(),
                         trajectory[:, 1].max() - trajectory[:, 1].min(),
                         trajectory[:, 2].max() - trajectory[:, 2].min()]).max() / 2.0
    
    mid_x = (trajectory[:, 0].max() + trajectory[:, 0].min()) * 0.5
    mid_y = (trajectory[:, 1].max() + trajectory[:, 1].min()) * 0.5
    mid_z = (trajectory[:, 2].max() + trajectory[:, 2].min()) * 0.5
    
    # Add margin to separate projections from trajectory
    margin = 0.25
    
    # Calculate axis limits that include margin
    x_min_limit = mid_x - max_range - margin
    y_min_limit = mid_y - max_range - margin
    z_min_limit = -margin
    
    # Plot projections at axis limits (on the planes)
    if show_projection:
        # XY plane projection - at minimum z limit
        ax.plot(trajectory[:, 0], trajectory[:, 1], 
               np.full_like(trajectory[:, 0], z_min_limit),
               'gray', alpha=0.3, linewidth=4)
        
        # XZ plane projection - at minimum y limit
        ax.plot(trajectory[:, 0], 
               np.full_like(trajectory[:, 1], y_min_limit),
               trajectory[:, 2],
               'gray', alpha=0.3, linewidth=4)
        
        # YZ plane projection - at minimum x limit
        ax.plot(np.full_like(trajectory[:, 0], x_min_limit),
               trajectory[:, 1], trajectory[:, 2],
               'gray', alpha=0.3, linewidth=4)
    
    # Set limits to include z=0 for ground plane projection
    ax.set_xlim(mid_x - max_range - margin, mid_x + max_range + margin)
    ax.set_ylim(mid_y - max_range - margin, mid_y + max_range + margin)
    ax.set_zlim(-margin, mid_z + max_range + margin)
    
    # Grid and legend
    ax.grid(True, alpha=0.3, linestyle=':')
    # ax.legend(loc='upper right', fontsize=10)
    
    # Set axis ticks to steps of 0.5
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.zaxis.set_major_locator(MultipleLocator(0.5))
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    # Ensure z-axis label is visible
    ax.zaxis.set_rotate_label(False)
    
    # plt.tight_layout(rect=[0, 0, 1.075, 1])
    
    return fig, ax, trajectory, phases, times


def plot_all_models_comparison(omega: float = 0.2, dt: float = 0.01, radius: float = 2.0, z_offset: float = 1.0):
    """
    Plot all embedding models in a comparison grid.
    
    Args:
        omega: Angular velocity (rad/s)
        dt: Time step (seconds)
        radius: Nominal radius of the trajectory
        z_offset: Vertical offset to lift trajectory from ground (meters)
    """
    models = ['modelA', 'modelB', 'modelC', 'modelD', 'modelE']
    
    fig = plt.figure(figsize=(18, 12))
    
    for i, model_name in enumerate(models):
        # Simulate trajectory
        trajectory, phases, times = simulate_embedding_trajectory(
            model_name, omega, dt, radius, z_offset=z_offset
        )
        
        # Create subplot
        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        
        # Plot trajectory colored by phase
        scatter = ax.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                           c=np.degrees(phases), cmap='hsv', s=2, alpha=0.8)
        
        # Mark start point
        ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2],
                  c='green', s=80, marker='o', edgecolors='black', linewidths=2)
        
        # Labels and title
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_zlabel('Z (m)', fontsize=10)
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
        
        # Set orthographic projection
        ax.set_proj_type('ortho')
        
        # Equal aspect ratio
        max_range = np.array([trajectory[:, 0].max() - trajectory[:, 0].min(),
                             trajectory[:, 1].max() - trajectory[:, 1].min(),
                             trajectory[:, 2].max() - trajectory[:, 2].min()]).max() / 2.0
        
        mid_x = (trajectory[:, 0].max() + trajectory[:, 0].min()) * 0.5
        mid_y = (trajectory[:, 1].max() + trajectory[:, 1].min()) * 0.5
        mid_z = (trajectory[:, 2].max() + trajectory[:, 2].min()) * 0.5
        
        # Add margin to separate projections from trajectory
        margin = 0.25
        
        ax.set_xlim(mid_x - max_range - margin, mid_x + max_range + margin)
        ax.set_ylim(mid_y - max_range - margin, mid_y + max_range + margin)
        ax.set_zlim(-margin, mid_z + max_range + margin)
        
        ax.grid(True, alpha=0.3)
        ax.view_init(elev=20, azim=45)
    
    # plt.suptitle(f'Theoretical 3D Trajectories - All Models (ω = {omega} rad/s)', 
    #             fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig
# ----------------------------------------------------------------------


if __name__ == '__main__':
    # Example 1: Plot single model
    print("Plotting modelA trajectory...")
    model_name = 'modelA'
    fig1, ax1, traj1, phases1, times1 = plot_embedding_trajectory_3d(
        model_name=model_name,
        omega=0.2,
        dt=0.01,
        radius=1.0,
        z_offset=1.0,
        show_projection=True,
        show_phase_color=True
    )
    plt.savefig(f'/home/paulo/Documents/k_10/plots/trajectory_{model_name}.png', dpi=150)
    print(f"Saved trajectory_{model_name}.png")
    
    # Example 2: Plot all models comparison
    print("\nPlotting all models comparison...")
    fig2 = plot_all_models_comparison(omega=0.2, dt=0.01, radius=2.0)
    plt.savefig('/home/paulo/Documents/k_10/plots/trajectories_all_models.png', dpi=150, bbox_inches='tight')
    print(f"Saved trajectories_all_models.png")
    
    # Print trajectory statistics
    print("\n" + "="*60)
    print(f"Trajectory Statistics for {model_name}:")
    print("="*60)
    print(f"Duration: {times1[-1]:.2f} seconds")
    print(f"Number of points: {len(traj1)}")
    print(f"X range: [{traj1[:, 0].min():.3f}, {traj1[:, 0].max():.3f}] m")
    print(f"Y range: [{traj1[:, 1].min():.3f}, {traj1[:, 1].max():.3f}] m")
    print(f"Z range: [{traj1[:, 2].min():.3f}, {traj1[:, 2].max():.3f}] m")
    print(f"Phase range: [{np.degrees(phases1.min()):.1f}, {np.degrees(phases1.max()):.1f}] degrees")
    
    plt.show()
