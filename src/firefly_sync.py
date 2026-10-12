"""
Decentralized Synchronization of Pulse-Coupled Spiking Oscillators

Models firefly-style swarm synchronization for autonomous drones with no
GPS (Global Positioning System) and no central controller. Agents fire
pulses based on the Mirollo-Strogatz pulse-coupled oscillator model, and
synchrony is tracked with the Kuramoto order parameter.

Three scenarios are covered: all-to-all coupling (upper bound baseline),
static decentralized coupling over a fixed random geometric graph, and
mobile decentralized coupling where agents move and the adjacency graph
is recomputed periodically.

Reproducibility: random seed 42.
"""

import os
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize

# Default simulation parameters
DEFAULT_B = 1.0          # Concave charging parameter
DEFAULT_EPS = 0.05       # Coupling strength (potential bump)
DEFAULT_DT = 0.01        # Timestep
DEFAULT_N = 100          # Swarm size
DEFAULT_AREA_SIZE = 10.0 # Domain [0, L] x [0, L]
DEFAULT_TMAX = 40.0      # Total simulation time


def charging_f(phi: np.ndarray, b: float = DEFAULT_B) -> np.ndarray:
    """
    Concave charging potential function f(phi) = (1/b) * ln(1 + (e^b - 1) * phi).
    Maps phase phi in [0, 1] to state potential x in [0, 1].
    """
    return (1.0 / b) * np.log(1.0 + (np.exp(b) - 1.0) * phi)


def inv_charging_f(x: np.ndarray, b: float = DEFAULT_B) -> np.ndarray:
    """
    Inverse charging function f_inv(x) = (e^(b*x) - 1) / (e^b - 1).
    Maps potential x in [0, 1] back to phase phi in [0, 1].
    """
    return (np.exp(b * x) - 1.0) / (np.exp(b) - 1.0)


def compute_order_parameter(phi: np.ndarray) -> float:
    """
    Computes Kuramoto-style order parameter r = |mean(exp(i * 2*pi * phi))|.
    r in [0, 1], where 0 indicates incoherent phase distribution and 1 indicates perfect synchrony.
    """
    return float(np.abs(np.mean(np.exp(1j * 2.0 * np.pi * phi))))


def compute_adjacency(positions: np.ndarray, comm_range: float) -> list[list[int]]:
    """
    Computes Euclidean adjacency list for N agents given comm_range.
    """
    n = positions.shape[0]
    # Compute pairwise Euclidean distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))

    adj_list = []
    for i in range(n):
        neighbors = [j for j in range(n) if i != j and dist[i, j] <= comm_range]
        adj_list.append(neighbors)
    return adj_list


def step_oscillators(
    phi: np.ndarray,
    adj_list: list[list[int]],
    eps: float = DEFAULT_EPS,
    b: float = DEFAULT_B,
    dt: float = DEFAULT_DT,
) -> np.ndarray:
    """
    Advances the phase of oscillators by dt and handles pulse coupling and recursive cascades.
    """
    phi = phi + dt
    n = len(phi)

    # Identify initial firers in this timestep
    fired_this_step = set()
    queue = deque()

    # Threshold check for natural firings
    for i in range(n):
        if phi[i] >= 1.0 - 1e-12:
            phi[i] = 0.0
            fired_this_step.add(i)
            queue.append(i)

    # Process recursive avalanche / cascade
    while queue:
        firer = queue.popleft()
        for neighbor in adj_list[firer]:
            if neighbor not in fired_this_step:
                # Potential bump
                current_pot = charging_f(phi[neighbor], b=b)
                new_pot = min(1.0, current_pot + eps)
                new_phi = inv_charging_f(new_pot, b=b)

                if new_pot >= 1.0 - 1e-12 or new_phi >= 1.0 - 1e-12:
                    phi[neighbor] = 0.0
                    fired_this_step.add(neighbor)
                    queue.append(neighbor)
                else:
                    phi[neighbor] = new_phi

    # Clamp phase bounds to [0, 1)
    phi = np.clip(phi, 0.0, 0.999999999)
    return phi


def simulate_all_to_all(
    n: int = DEFAULT_N,
    t_max: float = DEFAULT_TMAX,
    dt: float = DEFAULT_DT,
    eps: float = DEFAULT_EPS,
    b: float = DEFAULT_B,
    seed: int = 42,
) -> dict:
    """
    Scenario 1: All-to-all coupling baseline.
    """
    np.random.seed(seed)
    phi = np.random.uniform(0.0, 1.0, size=n)
    num_steps = int(np.round(t_max / dt))

    # All-to-all adjacency list
    adj_list = [[j for j in range(n) if j != i] for i in range(n)]

    time_series = np.linspace(0.0, t_max, num_steps + 1)
    order_params = np.zeros(num_steps + 1)
    order_params[0] = compute_order_parameter(phi)

    for step in range(1, num_steps + 1):
        phi = step_oscillators(phi, adj_list, eps=eps, b=b, dt=dt)
        order_params[step] = compute_order_parameter(phi)

    return {
        "time": time_series,
        "order_param": order_params,
        "final_phi": phi,
        "avg_degree": n - 1,
    }


def simulate_static(
    comm_range: float,
    n: int = DEFAULT_N,
    area_size: float = DEFAULT_AREA_SIZE,
    t_max: float = DEFAULT_TMAX,
    dt: float = DEFAULT_DT,
    eps: float = DEFAULT_EPS,
    b: float = DEFAULT_B,
    seed: int = 42,
) -> dict:
    """
    Scenario 2: Static decentralized coupling with fixed 2D random agent positions.
    """
    rng = np.random.RandomState(seed)
    positions = rng.uniform(0.0, area_size, size=(n, 2))
    phi = rng.uniform(0.0, 1.0, size=n)
    num_steps = int(np.round(t_max / dt))

    adj_list = compute_adjacency(positions, comm_range)
    avg_degree = float(np.mean([len(adj) for adj in adj_list]))

    time_series = np.linspace(0.0, t_max, num_steps + 1)
    order_params = np.zeros(num_steps + 1)
    order_params[0] = compute_order_parameter(phi)

    for step in range(1, num_steps + 1):
        phi = step_oscillators(phi, adj_list, eps=eps, b=b, dt=dt)
        order_params[step] = compute_order_parameter(phi)

    return {
        "time": time_series,
        "order_param": order_params,
        "final_phi": phi,
        "positions": positions,
        "avg_degree": avg_degree,
        "comm_range": comm_range,
    }


def simulate_mobile(
    speed: float,
    comm_range: float = 2.5,
    n: int = DEFAULT_N,
    area_size: float = DEFAULT_AREA_SIZE,
    t_max: float = DEFAULT_TMAX,
    dt: float = DEFAULT_DT,
    eps: float = DEFAULT_EPS,
    b: float = DEFAULT_B,
    recompute_interval: int = 5,
    seed: int = 42,
    record_trajectory: bool = False,
) -> dict:
    """
    Scenario 3: Mobile decentralized coupling with bouncing boundary conditions.
    Adjacency recomputed every recompute_interval timesteps.
    Uses separate RNG stream for headings to guarantee identical initial state to simulate_static.
    """
    # Main state RNG for positions and initial phases (exact match with simulate_static)
    rng_state = np.random.RandomState(seed)
    positions = rng_state.uniform(0.0, area_size, size=(n, 2))
    phi = rng_state.uniform(0.0, 1.0, size=n)

    # Separate, isolated RNG instance for velocity headings
    rng_motion = np.random.RandomState(seed + 1000)
    headings = rng_motion.uniform(0.0, 2.0 * np.pi, size=n)
    velocities = np.column_stack((speed * np.cos(headings), speed * np.sin(headings)))

    num_steps = int(np.round(t_max / dt))

    adj_list = compute_adjacency(positions, comm_range)
    degree_samples = [float(np.mean([len(adj) for adj in adj_list]))]

    time_series = np.linspace(0.0, t_max, num_steps + 1)
    order_params = np.zeros(num_steps + 1)
    order_params[0] = compute_order_parameter(phi)

    traj_positions = []
    traj_phases = []

    if record_trajectory:
        traj_positions.append(positions.copy())
        traj_phases.append(phi.copy())

    for step in range(1, num_steps + 1):
        # Update physical positions
        positions = positions + velocities * dt

        # Specular boundary reflection [0, area_size]
        for dim in range(2):
            # Lower boundary
            under = positions[:, dim] < 0.0
            positions[under, dim] = -positions[under, dim]
            velocities[under, dim] = -velocities[under, dim]

            # Upper boundary
            over = positions[:, dim] > area_size
            positions[over, dim] = 2.0 * area_size - positions[over, dim]
            velocities[over, dim] = -velocities[over, dim]

        # Recompute adjacency periodically
        if step % recompute_interval == 0:
            adj_list = compute_adjacency(positions, comm_range)
            degree_samples.append(float(np.mean([len(adj) for adj in adj_list])))

        # Step oscillator dynamics
        phi = step_oscillators(phi, adj_list, eps=eps, b=b, dt=dt)
        order_params[step] = compute_order_parameter(phi)

        if record_trajectory and (step % 4 == 0):  # Downsample for smooth animation
            traj_positions.append(positions.copy())
            traj_phases.append(phi.copy())

    return {
        "time": time_series,
        "order_param": order_params,
        "final_phi": phi,
        "avg_degree": float(np.mean(degree_samples)),
        "speed": speed,
        "traj_positions": traj_positions if record_trajectory else None,
        "traj_phases": traj_phases if record_trajectory else None,
    }


def find_sync_time(time_series: np.ndarray, order_param: np.ndarray, threshold: float = 0.9) -> str:
    """
    Finds the first time when the order parameter permanently or initially exceeds the threshold.
    """
    idx = np.where(order_param >= threshold)[0]
    if len(idx) > 0:
        return f"{time_series[idx[0]]:.2f}s"
    return "Did not reach"


def run_all_experiments():
    # Results are written to <repo_root>/results so this runs the same way
    # regardless of which machine or OS it's cloned onto.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(repo_root, "results")
    os.makedirs(output_dir, exist_ok=True)

    print("================================================================================")
    print(" Decentralized Pulse-Coupled Spiking Oscillators Simulation (Firefly Swarm)")
    print("================================================================================")

    # -------------------------------------------------------------------------
    # Scenario 1: All-to-all Baseline
    # -------------------------------------------------------------------------
    print("\n[1/4] Running Scenario 1: All-to-all coupling baseline...")
    res_s1 = simulate_all_to_all(n=100, t_max=35.0, dt=0.01, eps=0.05, b=1.0, seed=42)

    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(res_s1["time"], res_s1["order_param"], color="#1f77b4", lw=2, label="All-to-all (N=100)")
    plt.axhline(0.9, color="gray", linestyle="--", alpha=0.7, label="Sync threshold (r=0.9)")
    plt.title("Scenario 1: All-to-All Pulse-Coupled Oscillator Synchronization", fontsize=12, fontweight="bold")
    plt.xlabel("Time (s)", fontsize=11)
    plt.ylabel("Order Parameter $r(t)$", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scenario1_all_to_all.png"))
    plt.close()
    print("Saved Scenario 1 plot.")

    # -------------------------------------------------------------------------
    # Scenario 2: Static Decentralized Parameter Sweep
    # -------------------------------------------------------------------------
    print("\n[2/4] Running Scenario 2: Static decentralized coupling sweep across comm_range...")
    comm_ranges = [1.5, 2.2, 3.0, 4.5]
    res_s2_list = []

    plt.figure(figsize=(9, 5.5), dpi=300)
    colors_s2 = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8"]

    for r_comm, color in zip(comm_ranges, colors_s2):
        res = simulate_static(comm_range=r_comm, n=100, area_size=10.0, t_max=40.0, dt=0.01, eps=0.05, b=1.0, seed=42)
        res_s2_list.append(res)
        plt.plot(res["time"], res["order_param"], color=color, lw=1.8,
                 label=f"comm_range={r_comm:.1f} (⟨k⟩={res['avg_degree']:.1f})")

    plt.axhline(0.9, color="gray", linestyle="--", alpha=0.7, label="Sync threshold (r=0.9)")
    plt.title("Scenario 2: Static Swarm Synchronization vs Communication Range", fontsize=12, fontweight="bold")
    plt.xlabel("Time (s)", fontsize=11)
    plt.ylabel("Order Parameter $r(t)$", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scenario2_static_sweep.png"))
    plt.close()
    print("Saved Scenario 2 sweep plot.")

    # Phase cluster distribution for middling comm_range (e.g. 2.2)
    middling_res = res_s2_list[1]  # comm_range = 2.2
    sorted_phi = np.sort(middling_res["final_phi"])

    plt.figure(figsize=(8, 4.5), dpi=300)
    plt.plot(np.arange(1, 101), sorted_phi, marker="o", markersize=4, linestyle="-", color="#984ea3", lw=1.5)
    plt.title(f"Scenario 2: Final Sorted Phase Distribution (comm_range={middling_res['comm_range']}, ⟨k⟩={middling_res['avg_degree']:.1f})",
              fontsize=11, fontweight="bold")
    plt.xlabel("Agent Index (Sorted)", fontsize=11)
    plt.ylabel("Phase $\\phi \\in [0, 1)$", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.02, 1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scenario2_cluster_distribution.png"))
    plt.close()
    print("Saved Scenario 2 phase cluster distribution plot.")

    # -------------------------------------------------------------------------
    # Scenario 3: Mobile Decentralized Parameter Sweep
    # -------------------------------------------------------------------------
    print("\n[3/4] Running Scenario 3: Mobile decentralized coupling sweep across speed...")
    speeds = [0.0, 0.5, 1.5, 3.0]
    res_s3_list = []
    colors_s3 = ["#999999", "#386cb0", "#f0027f", "#7fc97f"]

    plt.figure(figsize=(9, 5.5), dpi=300)
    for spd, color in zip(speeds, colors_s3):
        res = simulate_mobile(speed=spd, comm_range=2.2, n=100, area_size=10.0, t_max=40.0, dt=0.01, eps=0.05, b=1.0, seed=42)
        res_s3_list.append(res)
        plt.plot(res["time"], res["order_param"], color=color, lw=1.8,
                 label=f"speed={spd:.1f} m/s (⟨k⟩={res['avg_degree']:.1f})")

    plt.axhline(0.9, color="gray", linestyle="--", alpha=0.7, label="Sync threshold (r=0.9)")
    plt.title("Scenario 3: Mobile Swarm Synchronization vs Drone Speed (comm_range=2.2)", fontsize=12, fontweight="bold")
    plt.xlabel("Time (s)", fontsize=11)
    plt.ylabel("Order Parameter $r(t)$", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scenario3_mobile_sweep.png"))
    plt.close()
    print("Saved Scenario 3 sweep plot.")

    # -------------------------------------------------------------------------
    # Scenario 3: 2D Spatial Scatter Animation of Swarm Synchronization
    # -------------------------------------------------------------------------
    print("\n[4/4] Generating 2D mobile swarm synchronization animation...")
    anim_res = simulate_mobile(speed=1.5, comm_range=2.5, n=100, area_size=10.0, t_max=35.0, dt=0.01, eps=0.05, b=1.0, seed=42, record_trajectory=True)

    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=120)
    norm = Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.get_cmap("twilight")

    pos0 = anim_res["traj_positions"][0]
    phi0 = anim_res["traj_phases"][0]

    scatter = ax.scatter(pos0[:, 0], pos0[:, 1], c=phi0, cmap=cmap, norm=norm, s=50, edgecolors="k", linewidth=0.5)
    ax.set_xlim(0, 10.0)
    ax.set_ylim(0, 10.0)
    ax.set_aspect("equal")
    ax.set_xlabel("X Position (m)", fontsize=10)
    ax.set_ylabel("Y Position (m)", fontsize=10)

    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Phase $\\phi \\in [0, 1)$", fontsize=10)

    title_text = ax.set_title("Mobile Swarm Sync: t=0.00s | r=0.00", fontsize=11, fontweight="bold")

    def update_frame(frame_idx):
        pos = anim_res["traj_positions"][frame_idx]
        phi = anim_res["traj_phases"][frame_idx]
        t_val = frame_idx * (0.01 * 4)  # Downsampling factor 4
        r_val = compute_order_parameter(phi)

        scatter.set_offsets(pos)
        scatter.set_array(phi)
        title_text.set_text(f"Mobile Swarm (speed=1.5, R=2.5)\nt={t_val:.2f}s | Order Parameter r={r_val:.3f}")
        return scatter, title_text

    num_frames = len(anim_res["traj_positions"])
    # Sample 140 frames evenly for smooth and compact GIF
    frame_indices = np.linspace(0, num_frames - 1, min(140, num_frames), dtype=int)

    anim_obj = animation.FuncAnimation(
        fig, update_frame, frames=frame_indices, interval=50, blit=False
    )

    anim_gif_path = os.path.join(output_dir, "scenario3_mobile_animation.gif")
    anim_obj.save(anim_gif_path, writer="pillow", fps=20)
    plt.close()
    print(f"Saved Scenario 3 animation to {anim_gif_path}")

    # -------------------------------------------------------------------------
    # Summary Metrics Table
    # -------------------------------------------------------------------------
    summary_rows = []

    # Scenario 1
    summary_rows.append({
        "Scenario": "1: All-to-All",
        "Parameter": "Baseline (N=100)",
        "Avg Degree": f"{res_s1['avg_degree']:.1f}",
        "Final r": f"{res_s1['order_param'][-1]:.4f}",
        "Time to r > 0.9": find_sync_time(res_s1["time"], res_s1["order_param"], 0.9),
    })

    # Scenario 2
    for res in res_s2_list:
        summary_rows.append({
            "Scenario": "2: Static Decentralized",
            "Parameter": f"comm_range={res['comm_range']:.1f}",
            "Avg Degree": f"{res['avg_degree']:.1f}",
            "Final r": f"{res['order_param'][-1]:.4f}",
            "Time to r > 0.9": find_sync_time(res["time"], res["order_param"], 0.9),
        })

    # Scenario 3
    for res in res_s3_list:
        summary_rows.append({
            "Scenario": "3: Mobile Decentralized",
            "Parameter": f"speed={res['speed']:.1f} m/s",
            "Avg Degree": f"{res['avg_degree']:.1f}",
            "Final r": f"{res['order_param'][-1]:.4f}",
            "Time to r > 0.9": find_sync_time(res["time"], res["order_param"], 0.9),
        })

    # Print Table
    print("\n" + "=" * 85)
    print(f"{'Scenario':<25} | {'Parameter':<18} | {'Avg Degree <k>':<14} | {'Final r':<10} | {'Time to r > 0.9':<15}")
    print("-" * 85)
    for r in summary_rows:
        print(f"{r['Scenario']:<25} | {r['Parameter']:<18} | {r['Avg Degree']:<14} | {r['Final r']:<10} | {r['Time to r > 0.9']:<15}")
    print("=" * 85 + "\n")

    # Save table to text file
    table_path = os.path.join(output_dir, "summary_table.txt")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(f"{'Scenario':<25} | {'Parameter':<18} | {'Avg Degree <k>':<14} | {'Final r':<10} | {'Time to r > 0.9':<15}\n")
        f.write("-" * 85 + "\n")
        for r in summary_rows:
            f.write(f"{r['Scenario']:<25} | {r['Parameter']:<18} | {r['Avg Degree']:<14} | {r['Final r']:<10} | {r['Time to r > 0.9']:<15}\n")

    print(f"All results and figures saved in {output_dir}/")


if __name__ == "__main__":
    run_all_experiments()
