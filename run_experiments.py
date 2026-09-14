"""
NK-Virus Model: Experiment Runner

This script runs two paired simulations of the NK-virus agent-based model defined in `nk_virus_model.py` -> one with NK (Natural Killer) cells
present, one without, and produces a summary figure comparing the resulting tissue dynamics.

----- WHAT IT DOES -----
    1. Loads simulation parameters from a JSON config file.
    2. Runs the simulation once with the NK cells as configured, and once with NK-cell introduction pushed past `time_max` (i.e. effectively disabled).
    3. Saves both raw result sets to .npz files.
    4. Builds a 2x3 summary figure comparing the two runs:
        - population bar charts (healthy/infected/dead) at two checkpoint times (24h and 48h),
        - infection state histograms (alive vs. dead cells) at the same two checkpoint times,
        - dead-cell count over time,
        - active-infection count over time.

----- RUNNING -----
The typical entry point is `run_experiments_and_plot(config_file)`, which takes the path to a JSON parameter file (see `run_with_params` in `nk_virus_model.py`
for the required keys) and writes `results_with_nk.npz`, `results_without_nk.npz` and `simulation_summary.jpg` to disk.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from nk_virus_model import run_with_params


def get_closest_frame(stats, target_time):
    """Return the index of the saved frame whose simulation time is closest to `target_time`.

    Args:
        stats: (n_frames, ...) stats array as returned by `run_with_params`.
        target_time: time (in hours) to match.

    Returns:
        Index of the frame with the smallest |time - target_time|.
    """
    times = stats[:, 3]
    return np.argmin(np.abs(times - target_time))


def run_experiments_and_plot(config_file = "params.json", rnd_seed = 100):
    """Run the NK-cell/no-NK-cell comparison and save the resulting summary figure.

    Args:
        config_file: path to a JSON file holding the parameter dictionary expected by `run_with_params` (see its docstring in `nk_virus_model.py` for the 
        full list of required keys).

    Results:
        Writes `results_with_nk.npz`, `results_without_nk.npz` and `simulation_summary.jpg` to the current working directory.
    """

    #loading parameters from file
    with open(config_file, 'r') as f:
        params = json.load(f)

    # ----- WITH NK CELLS -----
    print("Starting simulation WITH NK cells.")
    np.random.seed(rnd_seed)
    res_nk = run_with_params(params)
    stats_nk = res_nk["stats_array"]

    # ----- WITHOUT NK CELLS -----
    print("Starting simulation WITHOUT NK cells.")
    params_no_nk = params.copy()
    params_no_nk["nk_delay"] = params["time_max"] + 999.0
    np.random.seed(rnd_seed)
    res_no_nk = run_with_params(params_no_nk)
    stats_no_nk = res_no_nk["stats_array"]

    #saving the results to .npz files
    np.savez_compressed("results_with_nk.npz", **res_nk)
    np.savez_compressed("results_without_nk.npz", **res_no_nk)
    print("Simulation results saved to .npz files.")

    # ----- PLOTTING THE RESULTS -----
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Effect of NK cells on the course of viral infection", fontsize=16)

    s_max = params["s_max"]
    states_x = np.arange(s_max + 1)

    times_to_plot = [24.0, 48.0]

    hist_axes = []
    hist_ymax = 0.0

    bar_axes = []
    bar_ymax = 0.0

    for i, t_target in enumerate(times_to_plot):
        idx_nk = get_closest_frame(stats_nk, t_target)
        idx_no_nk = get_closest_frame(stats_no_nk, t_target)

        # --- POPULATION BAR CHARTS ---
        ax_bar = axes[0, i]

        pop_nk = stats_nk[idx_nk, 0:3]              #cell count: [healthy, infected, dead]
        pop_no_nk = stats_no_nk[idx_no_nk, 0:3]

        bar_width = 0.35
        x = np.arange(3)

        ax_bar.bar(x - bar_width/2, pop_nk, bar_width, label = 'With NK cells', color = ['seagreen', 'orange', 'firebrick'])
        ax_bar.bar(x + bar_width/2, pop_no_nk, bar_width, label = 'Without NK cells', color = ['darkseagreen', 'wheat', 'salmon'])

        ax_bar.set_title(f"Tissue state at {t_target:.0f}h")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(['Healthy', 'Infected', 'Dead'])
        ax_bar.set_ylabel("Number of cells")
        ax_bar.grid(axis = 'y', linestyle = '--', alpha = 0.3)
        
        #legend only on the first plot to avoid clutter
        if i == 0:
            handles = [plt.Rectangle((0, 0), 1, 1, color = c) for c in ['seagreen', 'darkseagreen', 'orange', 'wheat', 'firebrick', 'salmon']]
            labels = ['Healthy (NK)', 'Healthy (no NK)', 'Infected (NK)', 'Infected (no NK)', 'Dead (NK)', 'Dead (no NK)']
            ax_bar.legend(handles, labels, fontsize = 'small', ncols = 2)

        bar_axes.append(ax_bar)
        bar_ymax = max(bar_ymax, np.max(pop_nk), np.max(pop_no_nk))

        # --- INFECTION STATE HISTOGRAM ---
        ax_hist = axes[1, i]

        alive_nk = stats_nk[idx_nk, 4 : 4 + s_max + 1].copy()
        dead_nk = stats_nk[idx_nk, 4 + s_max + 1:].copy()

        alive_no_nk = stats_no_nk[idx_no_nk, 4 : 4 + s_max + 1].copy()
        dead_no_nk = stats_no_nk[idx_no_nk, 4 + s_max + 1 :].copy()

        ax_hist.bar(states_x - bar_width/2, alive_nk, bar_width, label = 'Alive (NK)', color = 'steelblue')
        ax_hist.bar(states_x - bar_width/2, dead_nk, bar_width, bottom = alive_nk, label = 'Dead (NK)', color = 'steelblue', alpha = 0.5)

        ax_hist.bar(states_x + bar_width/2, alive_no_nk, bar_width, label = 'Alive (no NK)', color = 'palevioletred')
        ax_hist.bar(states_x + bar_width/2, dead_no_nk, bar_width, bottom = alive_no_nk, label = 'Dead (no NK)', color = 'palevioletred', alpha = 0.5)

        ax_hist.axvline(params["m_d"], color = 'gray', linestyle = '--', linewidth = 1, label = 'death threshold $m_d$' if i == 0 else None)
        
        ax_hist.set_title(f"Infection state distribution at {t_target:.0f}h")
        ax_hist.set_xlabel("Infection state s")
        ax_hist.set_ylabel("Number of living cells")
        ax_hist.set_xlim((0.5, params["s_max"] + 0.5))
        ax_hist.grid(axis = 'y', linestyle = '--', alpha = 0.3)

        if i == 0:
            ax_hist.legend(fontsize = 'small')

        #tracking this subplot's tallest stacked bar (alive + dead, for either NK series) to unify the y-axis scale across both checkpoint times
        hist_axes.append(ax_hist)
        hist_ymax = max(hist_ymax, np.max((alive_nk + dead_nk)[1:]), np.max((alive_no_nk + dead_no_nk)[1:]))

    for ax_hist in hist_axes:
        ax_hist.set_ylim(0, hist_ymax * 1.05)

    for ax_bar in bar_axes:
        ax_bar.set_ylim(0, bar_ymax * 1.05)

    # --- CUMULATIVE DEAD-CELL COUNT OVER TIME ---
    ax_dead_time = axes[0, 2]
    ax_dead_time.plot(stats_nk[:, 3], stats_nk[:, 2], label = 'With NK cells', color = 'firebrick', linewidth = 2)
    ax_dead_time.plot(stats_no_nk[:, 3], stats_no_nk[:, 2], label = 'Without NK cells', color = 'salmon', linewidth = 2)
    ax_dead_time.set_title("Cumulative dead-cell count over time")
    ax_dead_time.set_xlabel("Time [h]")
    ax_dead_time.set_ylabel("Dead cells")
    ax_dead_time.grid(linestyle = '--', alpha = 0.3)
    ax_dead_time.legend()

    # --- ACTIVE INFECTIONS OVER TIME ---
    ax_inf_time = axes[1, 2]
    ax_inf_time.plot(stats_nk[:, 3], stats_nk[:, 1], label = 'With NK cells', color = 'orange', linewidth = 2)
    ax_inf_time.plot(stats_no_nk[:, 3], stats_no_nk[:, 1], label = 'Without NK cells', color = 'wheat', linewidth = 2)
    ax_inf_time.set_title("Active infections over time")
    ax_inf_time.set_xlabel("Time [h]")
    ax_inf_time.set_ylabel("Infected cells (alive)")
    ax_inf_time.grid(linestyle = '--', alpha = 0.3)
    ax_inf_time.legend()

    plt.tight_layout(rect = [0, 0.03, 1, 0.95])
    plt.savefig("simulation_summary.jpg", dpi = 300)
    print("Plots saved as simulation_summary.jpg.")

if __name__ == "__main__":
    run_experiments_and_plot()