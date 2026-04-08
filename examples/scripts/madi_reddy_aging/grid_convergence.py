"""
Grid convergence study for the Madi-Reddy SEI model.

Runs the model at N = 10, 20, 40, 80 and compares final SEI thickness
and charge loss to check that the method-of-lines discretisation is
converged.
"""

import numpy as np
import pybamm
import matplotlib.pyplot as plt
from madi_reddy_sei_model import MadiReddySEIModel, param, graphite_ocv


def run_convergence():
    N_values = [10, 20, 40, 80]
    t_eval = np.linspace(0, 365 * 24 * 3600, 400)
    solver = pybamm.CasadiSolver(mode="safe")

    results = {}

    for N in N_values:
        print(f"Solving N = {N} ...")
        model = MadiReddySEIModel(N=N)
        sim = pybamm.Simulation(model, parameter_values=param, solver=solver)
        sol = sim.solve(t_eval)

        results[N] = {
            "time_days": sol.t / 86400,
            "sei_nm": sol["SEI thickness [nm]"].entries,
            "q_loss": sol["Charge loss per area [A.h.m-2]"].entries,
            "soc": sol["SoC [-]"].entries,
        }
        print(f"  N={N}: final SEI = {results[N]['sei_nm'][-1]:.4f} nm, "
              f"Q_loss = {results[N]['q_loss'][-1]:.4e} A·h·m⁻²")

    # Plot convergence
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="white")
    fig.suptitle("Grid Convergence Study", fontsize=14, fontweight="bold")

    cmap = plt.cm.plasma
    for idx, N in enumerate(N_values):
        color = cmap(idx / (len(N_values) - 1))
        r = results[N]
        axes[0].plot(r["time_days"], r["sei_nm"], color=color, linewidth=1.5, label=f"N={N}")
        axes[1].plot(r["time_days"], r["q_loss"], color=color, linewidth=1.5, label=f"N={N}")
        axes[2].plot(r["time_days"], r["soc"], color=color, linewidth=1.5, label=f"N={N}")

    axes[0].set_xlabel("Time [days]")
    axes[0].set_ylabel("SEI thickness [nm]")
    axes[0].set_title("SEI Thickness")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Time [days]")
    axes[1].set_ylabel("Charge loss [A·h·m⁻²]")
    axes[1].set_title("Charge Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel("Time [days]")
    axes[2].set_ylabel("SoC [-]")
    axes[2].set_title("State of Charge")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = "grid_convergence.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\nConvergence plot saved to: {output_path}")
    plt.close(fig)

    # Print convergence table
    print("\n" + "=" * 55)
    print(f"{'N':>5} | {'Final SEI [nm]':>15} | {'Final Q_loss [A·h/m²]':>22}")
    print("-" * 55)
    for N in N_values:
        r = results[N]
        print(f"{N:>5} | {r['sei_nm'][-1]:>15.6f} | {r['q_loss'][-1]:>22.6e}")
    print("=" * 55)


if __name__ == "__main__":
    run_convergence()
