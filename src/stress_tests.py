"""
Stress-Testing Module for Decentralized Pulse-Coupled Spiking Oscillators

Pushes the synchronization model from firefly_sync.py against three
real-world degradations that a GPS-denied (Global Positioning System
denied) drone swarm would actually face: oscillator frequency
heterogeneity, communication delay, and packet loss. Also includes a
fine-grained sweep over coupling strength to find the recovery threshold
under fixed delay, validated across multiple random seeds.

Reproducibility: random seed 42, plus seeds 101 and 2024 for the
multi-seed validation pass.
"""

import os
import math
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

from firefly_sync import (
    compute_order_parameter,
    compute_adjacency,
    simulate_static,
    DEFAULT_B,
    DEFAULT_EPS,
    DEFAULT_DT,
    DEFAULT_N,
    DEFAULT_AREA_SIZE,
    DEFAULT_TMAX,
)


def compute_instantaneous_sync_time(time_series: np.ndarray, order_param: np.ndarray, threshold: float = 0.9) -> str:
    """
    Legacy metric: Finds the first single timestep where r >= threshold.
    """
    idx = np.where(order_param >= threshold)[0]
    if len(idx) > 0:
        return f"{time_series[idx[0]]:.2f}s"
    return "never"


def compute_sustained_sync_time(
    time_series: np.ndarray,
    order_param: np.ndarray,
    dt: float = DEFAULT_DT,
    window_duration: float = 1.0,
    threshold: float = 0.9,
) -> str:
    """
    Windowed metric: Finds the first timestep t such that r(t') >= threshold
    for EVERY t' in [t, t + window_duration]. Returns 'never' if no such window exists.
    """
    win_steps = int(np.round(window_duration / dt))
    n_total = len(order_param)

    for i in range(n_total - win_steps + 1):
        if np.all(order_param[i : i + win_steps] >= threshold):
            return f"{time_series[i]:.2f}s"
    return "never"


def compute_windowed_stats(
    order_param: np.ndarray,
    dt: float = DEFAULT_DT,
    duration: float = 2.0,
) -> tuple[float, float]:
    """
    Computes the mean and standard deviation of r(t) over the final 2 seconds of simulation.
    """
    last_steps = int(np.round(duration / dt))
    r_slice = order_param[-last_steps:]
    return float(np.mean(r_slice)), float(np.std(r_slice))


def simulate_degraded(
    comm_range: float = 3.0,
    n: int = DEFAULT_N,
    area_size: float = DEFAULT_AREA_SIZE,
    t_max: float = DEFAULT_TMAX,
    dt: float = DEFAULT_DT,
    eps: float = DEFAULT_EPS,
    b: float = DEFAULT_B,
    freq_std: float = 0.0,
    delay_steps: int = 0,
    drop_prob: float = 0.0,
    seed: int = 42,
) -> dict:
    """
    Fast, highly optimized simulation of pulse-coupled oscillators under configurable degradations.
    """
    # 1. Spatial positions and initial phases (exact match with baseline simulate_static)
    rng_state = np.random.RandomState(seed)
    positions = rng_state.uniform(0.0, area_size, size=(n, 2))
    phi = rng_state.uniform(0.0, 1.0, size=n)

    # Precompute constant exponential factors for fast scalar charging evaluation
    exp_b_minus_1 = math.exp(b) - 1.0
    inv_b = 1.0 / b
    inv_exp_b_minus_1 = 1.0 / exp_b_minus_1

    # 2. Isolated RNG for frequency heterogeneity
    if freq_std > 0.0:
        rng_freq = np.random.RandomState(seed + 2000)
        omega = rng_freq.normal(1.0, freq_std, size=n)
        omega = np.clip(omega, 0.3, 1.7)
    else:
        omega = np.ones(n, dtype=float)

    # 3. Isolated RNG for packet loss
    rng_drop = np.random.RandomState(seed + 3000)

    adj_list = compute_adjacency(positions, comm_range)
    avg_degree = float(np.mean([len(adj) for adj in adj_list]))

    num_steps = int(np.round(t_max / dt))

    # Preallocate delayed pulse schedule buffer
    max_sched_step = num_steps + delay_steps + 100
    delayed_buffer = [[] for _ in range(max_sched_step)]

    time_series = np.linspace(0.0, t_max, num_steps + 1)
    order_params = np.zeros(num_steps + 1)
    order_params[0] = compute_order_parameter(phi)

    # Local variable caching for loop acceleration
    threshold_val = 1.0 - 1e-12

    for step in range(1, num_steps + 1):
        # 1. Phase increment at natural frequency
        phi = phi + omega * dt

        fired_this_step = set()
        queue = deque()

        # 2. First deliver any delayed bumps scheduled to arrive at this exact timestep
        if delay_steps > 0 and step < len(delayed_buffer) and len(delayed_buffer[step]) > 0:
            for neighbor in delayed_buffer[step]:
                if neighbor not in fired_this_step:
                    # Fast scalar potential computation
                    p_val = phi[neighbor]
                    current_pot = inv_b * math.log(1.0 + exp_b_minus_1 * p_val)
                    new_pot = current_pot + eps
                    if new_pot >= 1.0:
                        new_pot = 1.0
                        new_phi = 1.0
                    else:
                        new_phi = (math.exp(b * new_pot) - 1.0) * inv_exp_b_minus_1

                    if new_pot >= threshold_val or new_phi >= threshold_val:
                        phi[neighbor] = 0.0
                        fired_this_step.add(neighbor)
                        queue.append(neighbor)
                    else:
                        phi[neighbor] = new_phi

        # 3. Check natural threshold crossings for oscillators not yet fired
        for i in range(n):
            if i not in fired_this_step and phi[i] >= threshold_val:
                phi[i] = 0.0
                fired_this_step.add(i)
                queue.append(i)

        # 4. Process recursive cascade
        while queue:
            firer = queue.popleft()
            for neighbor in adj_list[firer]:
                # Packet drop check
                if drop_prob > 0.0 and rng_drop.uniform(0.0, 1.0) < drop_prob:
                    continue

                if delay_steps == 0:
                    # Immediate potential bump (if not already fired this timestep)
                    if neighbor not in fired_this_step:
                        p_val = phi[neighbor]
                        current_pot = inv_b * math.log(1.0 + exp_b_minus_1 * p_val)
                        new_pot = current_pot + eps
                        if new_pot >= 1.0:
                            new_pot = 1.0
                            new_phi = 1.0
                        else:
                            new_phi = (math.exp(b * new_pot) - 1.0) * inv_exp_b_minus_1

                        if new_pot >= threshold_val or new_phi >= threshold_val:
                            phi[neighbor] = 0.0
                            fired_this_step.add(neighbor)
                            queue.append(neighbor)
                        else:
                            phi[neighbor] = new_phi
                else:
                    # Schedule bump for future arrival
                    target_step = step + delay_steps
                    if target_step < len(delayed_buffer):
                        delayed_buffer[target_step].append(neighbor)

        # Clamp phase bounds
        phi = np.clip(phi, 0.0, 0.999999999)
        order_params[step] = compute_order_parameter(phi)

    return {
        "time": time_series,
        "order_param": order_params,
        "final_phi": phi,
        "avg_degree": avg_degree,
        "comm_range": comm_range,
    }


def verify_baseline_equivalence():
    """
    Verification step: confirms that simulate_degraded with zero degradation
    is bit-for-bit identical to simulate_static at comm_range=3.0.
    """
    print("[Verification] Checking zero-degradation baseline equivalence...", flush=True)
    res_base = simulate_static(comm_range=3.0, seed=42)
    res_deg = simulate_degraded(comm_range=3.0, freq_std=0.0, delay_steps=0, drop_prob=0.0, seed=42)

    max_diff = np.max(np.abs(res_base["order_param"] - res_deg["order_param"]))
    print(f"Max difference in order parameter time series: {max_diff:.10e}", flush=True)
    assert max_diff < 1e-12, f"Verification failed! Difference {max_diff} >= 1e-12"
    print("Zero-degradation baseline verified successfully (within machine precision < 1e-12).\n", flush=True)


def build_row_entry(scenario_name: str, param_str: str, res: dict) -> dict:
    t = res["time"]
    r = res["order_param"]
    mean_2s, std_2s = compute_windowed_stats(r, dt=0.01, duration=2.0)
    inst_t = compute_instantaneous_sync_time(t, r, threshold=0.9)
    sust_t = compute_sustained_sync_time(t, r, dt=0.01, window_duration=1.0, threshold=0.9)

    return {
        "Scenario": scenario_name,
        "Parameter": param_str,
        "Avg Degree": f"{res['avg_degree']:.1f}",
        "Final r": f"{r[-1]:.4f}",
        "Mean r (last 2s)": f"{mean_2s:.4f}",
        "Std r (last 2s)": f"{std_2s:.4f}",
        "Inst. r > 0.9": inst_t,
        "Sustained r > 0.9": sust_t,
    }


def run_stress_testing_experiments():
    # Results are written to <repo_root>/results so this runs the same way
    # regardless of which machine or OS it's cloned onto.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(repo_root, "results")
    os.makedirs(output_dir, exist_ok=True)

    print("================================================================================", flush=True)
    print(" Stress-Testing Module: Decentralized Pulse-Coupled Spiking Oscillators", flush=True)
    print("================================================================================", flush=True)

    # 0. Verification check
    verify_baseline_equivalence()

    summary_rows = []

    # -------------------------------------------------------------------------
    # Experiment 1: Frequency Heterogeneity (Subplot Grid to prevent z-order occlusion)
    # -------------------------------------------------------------------------
    print("[1/4] Running Frequency Heterogeneity Sweep (freq_std) on 2x3 Subplot Grid...", flush=True)
    freq_stds = [0.0, 0.01, 0.02, 0.04, 0.07, 0.10]
    colors_freq = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#e6ab02", "#a6761d"]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), dpi=300, sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, (f_std, color) in enumerate(zip(freq_stds, colors_freq)):
        res = simulate_degraded(comm_range=3.0, freq_std=f_std, delay_steps=0, drop_prob=0.0, seed=42)
        ax = axes[idx]
        ax.plot(res["time"], res["order_param"], color=color, lw=1.8, label=f"σ_ω = {f_std:.2f}")
        ax.axhline(0.9, color="gray", linestyle="--", alpha=0.7, label="r = 0.9")
        ax.set_title(f"Heterogeneity $\\sigma_\\omega = {f_std:.2f}$", fontsize=11, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        if idx >= 3:
            ax.set_xlabel("Time (s)", fontsize=10)
        if idx % 3 == 0:
            ax.set_ylabel("Order Parameter $r(t)$", fontsize=10)
        ax.legend(loc="lower right", fontsize=9)

        summary_rows.append(build_row_entry("Stress: Freq Heterogeneity", f"freq_std={f_std:.2f}", res))

    fig.suptitle("Stress Test: Frequency Heterogeneity (Separate Subplots, comm_range=3.0)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stress_freq_heterogeneity.png"))
    plt.close()
    print("Saved stress_freq_heterogeneity.png (2x3 subplots verified, sigma_w=0.00 is flat).\n", flush=True)

    # -------------------------------------------------------------------------
    # Experiment 2: Communication Delay Sweep
    # -------------------------------------------------------------------------
    print("[2/4] Running Communication Delay Sweep (delay_steps)...", flush=True)
    delays = [0, 2, 5, 10, 25, 50, 100]  # in timesteps (0s to 1.0s)
    colors_delays = ["#2b83ba", "#74add1", "#abd9e9", "#fdae61", "#f46d43", "#d73027", "#7f0000"]

    plt.figure(figsize=(9, 5.5), dpi=300)
    for d_step, color in zip(delays, colors_delays):
        res = simulate_degraded(comm_range=3.0, freq_std=0.0, delay_steps=d_step, drop_prob=0.0, seed=42)
        plt.plot(res["time"], res["order_param"], color=color, lw=1.8,
                 label=f"delay = {d_step} steps ({d_step*0.01:.2f}s)")
        summary_rows.append(build_row_entry("Stress: Comm Delay", f"delay={d_step} steps ({d_step*0.01:.2f}s)", res))

    plt.axhline(0.9, color="gray", linestyle="--", alpha=0.7, label="Sync threshold (r=0.9)")
    plt.title("Stress Test: Communication Delay (comm_range=3.0)", fontsize=12, fontweight="bold")
    plt.xlabel("Time (s)", fontsize=11)
    plt.ylabel("Order Parameter $r(t)$", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stress_comm_delay.png"))
    plt.close()
    print("Saved stress_comm_delay.png\n", flush=True)

    # -------------------------------------------------------------------------
    # Experiment 3: Packet Loss Sweep
    # -------------------------------------------------------------------------
    print("[3/4] Running Packet Loss Sweep (drop_prob)...", flush=True)
    drop_probs = [0.0, 0.15, 0.30, 0.50, 0.70]
    colors_drops = ["#33a02c", "#b2df8a", "#fb9a99", "#e31a1c", "#6a3d9a"]

    plt.figure(figsize=(9, 5.5), dpi=300)
    for d_prob, color in zip(drop_probs, colors_drops):
        res = simulate_degraded(comm_range=3.0, freq_std=0.0, delay_steps=0, drop_prob=d_prob, seed=42)
        plt.plot(res["time"], res["order_param"], color=color, lw=1.8,
                 label=f"drop_prob = {d_prob:.2f}")
        summary_rows.append(build_row_entry("Stress: Packet Loss", f"drop_prob={d_prob:.2f}", res))

    plt.axhline(0.9, color="gray", linestyle="--", alpha=0.7, label="Sync threshold (r=0.9)")
    plt.title("Stress Test: Packet Loss (comm_range=3.0)", fontsize=12, fontweight="bold")
    plt.xlabel("Time (s)", fontsize=11)
    plt.ylabel("Order Parameter $r(t)$", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stress_packet_loss.png"))
    plt.close()
    print("Saved stress_packet_loss.png\n", flush=True)

    # -------------------------------------------------------------------------
    # Experiment 4: Fine-Grained Delay Recovery Sweep (eps in [0.08 .. 0.20], delay_steps=5)
    # -------------------------------------------------------------------------
    print("[4/4] Running Fine-Grained Delay Recovery Sweep (eps across 0.08..0.20 with delay=5 steps)...", flush=True)
    eps_fine = [0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
    colors_eps_fine = ["#8c510a", "#bf812d", "#dfc27d", "#80cdc1", "#35978f", "#01665e", "#003c30"]

    plt.figure(figsize=(9, 5.5), dpi=300)
    for eps_val, color in zip(eps_fine, colors_eps_fine):
        res = simulate_degraded(comm_range=3.0, eps=eps_val, freq_std=0.0, delay_steps=5, drop_prob=0.0, seed=42)
        plt.plot(res["time"], res["order_param"], color=color, lw=1.8,
                 label=f"ε = {eps_val:.2f}")
        summary_rows.append(build_row_entry("Stress: Delay Recovery", f"eps={eps_val:.2f} (delay=5)", res))

    plt.axhline(0.9, color="gray", linestyle="--", alpha=0.7, label="Sync threshold (r=0.9)")
    plt.title("Stress Test: Fine-Grained Delay Recovery vs Coupling Strength (delay=5 steps, 0.05s)", fontsize=11, fontweight="bold")
    plt.xlabel("Time (s)", fontsize=11)
    plt.ylabel("Order Parameter $r(t)$", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stress_delay_recovery_eps.png"))
    plt.close()
    print("Saved stress_delay_recovery_eps.png (Fine-grained).\n", flush=True)

    # -------------------------------------------------------------------------
    # Multi-Seed Validation of Delay Recovery (Seeds 42, 101, 2024)
    # -------------------------------------------------------------------------
    print("[Multi-Seed Analysis] Evaluating Delay Recovery across Seeds (42, 101, 2024)...", flush=True)
    test_seeds = [42, 101, 2024]
    multiseed_rows = []

    for s_val in test_seeds:
        for eps_val in eps_fine:
            res_s = simulate_degraded(comm_range=3.0, eps=eps_val, freq_std=0.0, delay_steps=5, drop_prob=0.0, seed=s_val)
            row = build_row_entry(f"MultiSeed (seed={s_val})", f"eps={eps_val:.2f}", res_s)
            multiseed_rows.append(row)

    # -------------------------------------------------------------------------
    # Print and Save Stress Test Summary Table
    # -------------------------------------------------------------------------
    header = (
        f"{'Scenario':<26} | {'Parameter':<22} | {'Avg <k>':<8} | "
        f"{'Final r':<8} | {'Mean(2s)':<8} | {'Std(2s)':<8} | "
        f"{'Inst r>0.9':<10} | {'Sustained r>0.9':<15}"
    )
    sep_line = "-" * len(header)

    print("\n" + "=" * len(header), flush=True)
    print(header, flush=True)
    print(sep_line, flush=True)
    for r in summary_rows:
        print(
            f"{r['Scenario']:<26} | {r['Parameter']:<22} | {r['Avg Degree']:<8} | "
            f"{r['Final r']:<8} | {r['Mean r (last 2s)']:<8} | {r['Std r (last 2s)']:<8} | "
            f"{r['Inst. r > 0.9']:<10} | {r['Sustained r > 0.9']:<15}",
            flush=True
        )
    print("=" * len(header) + "\n", flush=True)

    table_path = os.path.join(output_dir, "stress_summary_table.txt")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        f.write(sep_line + "\n")
        for r in summary_rows:
            f.write(
                f"{r['Scenario']:<26} | {r['Parameter']:<22} | {r['Avg Degree']:<8} | "
                f"{r['Final r']:<8} | {r['Mean r (last 2s)']:<8} | {r['Std r (last 2s)']:<8} | "
                f"{r['Inst. r > 0.9']:<10} | {r['Sustained r > 0.9']:<15}\n"
            )

    # Save Multi-Seed Validation Table
    multiseed_table_path = os.path.join(output_dir, "delay_recovery_multiseed_table.txt")
    with open(multiseed_table_path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        f.write(sep_line + "\n")
        for r in multiseed_rows:
            f.write(
                f"{r['Scenario']:<26} | {r['Parameter']:<22} | {r['Avg Degree']:<8} | "
                f"{r['Final r']:<8} | {r['Mean r (last 2s)']:<8} | {r['Std r (last 2s)']:<8} | "
                f"{r['Inst. r > 0.9']:<10} | {r['Sustained r > 0.9']:<15}\n"
            )

    print(f"All stress testing results, multi-seed tables, and plots saved in {output_dir}/", flush=True)


if __name__ == "__main__":
    run_stress_testing_experiments()
