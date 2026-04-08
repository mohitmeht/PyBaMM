"""
Compare SEI growth at initial SoC = 0.3 vs 0.9 (25 °C).
Includes cell voltage (LCO/graphite) over 360 days.
Milestones at 30, 60, 90, 120, 180, 360 days.
"""

import numpy as np
import pybamm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from madi_reddy_sei_model import MadiReddySEIModel, param


# =========================================================================
# Positive electrode OCV: LiCoO2 (standard Dualfoil parametrisation)
# =========================================================================
def lco_ocp(y):
    """LiCoO2 OCP [V] vs Li/Li+. y = stoichiometry (Li fraction)."""
    y = np.clip(y, 0.01, 0.99)
    return (
        4.04596
        + np.exp(-42.30027 * y + 16.56714)
        - 0.04880 * np.arctan(50.01833 * y - 26.48897)
        - 0.05447 * np.arctan(18.99678 * y - 12.32362)
        - np.exp(78.24095 * y - 78.68074)
    )


def graphite_ocp_np(x):
    """Graphite OCP [V] vs Li/Li+ (Appendix A, numpy version)."""
    x = np.clip(x, 1e-6, 0.999999)
    return (
        0.7222
        + 0.1387 * x
        + 0.029 * np.sqrt(x)
        - 0.0172 / x
        + 0.0019 / x**1.5
        + 0.2802 * np.exp(0.9 - 15 * x)
        - 0.7984 * np.exp(0.4465 * x - 0.4108)
    )


# =========================================================================
# Cell design: stoichiometry windows for Sony 18650 (LCO/graphite)
# =========================================================================
# At full cell SoC=1 (charged): x_neg=0.9, y_pos=0.50
# At full cell SoC=0 (discharged): x_neg=0.01, y_pos=0.99
X_100, X_0 = 0.9, 0.01      # negative electrode stoich at SoC=1, SoC=0
Y_100, Y_0 = 0.50, 0.99     # positive electrode stoich at SoC=1, SoC=0


def x_to_y_pos(x_neg):
    """Map negative stoichiometry to positive stoichiometry (initial balance)."""
    soc_cell = (x_neg - X_0) / (X_100 - X_0)
    return Y_0 + soc_cell * (Y_100 - Y_0)


def run_soc_comparison():
    soc_values = [0.3, 0.9]
    colors = {"0.3": "#e63946", "0.9": "#1d3557"}
    milestone_days = [30, 60, 90, 120, 180, 360]
    t_eval = np.linspace(0, 360 * 24 * 3600, 800)
    solver = pybamm.CasadiSolver(mode="safe")
    N = 40

    results = {}
    for z0 in soc_values:
        print(f"Solving SoC₀ = {z0} ...")
        model = MadiReddySEIModel(N=N)
        p = pybamm.ParameterValues(dict(param))
        p["Initial SoC [-]"] = z0
        sim = pybamm.Simulation(model, parameter_values=p, solver=solver)
        sol = sim.solve(t_eval)

        time_days = sol.t / 86400
        x_neg = sol["SoC [-]"].entries          # negative stoichiometry over time
        y_pos_init = x_to_y_pos(z0)             # positive stoich (fixed during storage)

        # Cell voltage = U_pos(y_fixed) - U_neg(x(t))
        v_cell = lco_ocp(y_pos_init) - graphite_ocp_np(x_neg)

        results[z0] = {
            "time_days": time_days,
            "sei_nm": sol["SEI thickness [nm]"].entries,
            "soc": x_neg,
            "ocv": sol["Negative OCV [V]"].entries,
            "jsei": sol["SEI current density [A.m-2]"].entries,
            "q_loss": sol["Charge loss per area [A.h.m-2]"].entries,
            "eta": sol["SEI overpotential [V]"].entries,
            "v_cell": v_cell,
            "y_pos": y_pos_init,
        }
        r = results[z0]
        print(f"  y_pos = {y_pos_init:.4f} (fixed)")
        print(f"  V_cell: {v_cell[0]:.4f} → {v_cell[-1]:.4f} V")
        print(f"  Final SEI = {r['sei_nm'][-1]:.2f} nm, "
              f"Final SoC = {r['soc'][-1]:.4f}")

    # =====================================================================
    # Milestone table
    # =====================================================================
    print("\n" + "=" * 105)
    print(f"{'Days':>6} | {'SEI (0.3)':>10} {'SEI (0.9)':>10} | "
          f"{'SoC (0.3)':>9} {'SoC (0.9)':>9} | "
          f"{'V_cell (0.3)':>12} {'V_cell (0.9)':>12} | "
          f"{'Q_loss (0.3)':>12} {'Q_loss (0.9)':>12}")
    print("-" * 105)
    for d in milestone_days:
        row = []
        for z0 in soc_values:
            r = results[z0]
            idx = np.argmin(np.abs(r["time_days"] - d))
            row.append((r["sei_nm"][idx], r["soc"][idx],
                        r["v_cell"][idx], r["q_loss"][idx]))
        print(f"{d:>6} | {row[0][0]:>10.2f} {row[1][0]:>10.2f} | "
              f"{row[0][1]:>9.4f} {row[1][1]:>9.4f} | "
              f"{row[0][2]:>12.4f} {row[1][2]:>12.4f} | "
              f"{row[0][3]:>12.4e} {row[1][3]:>12.4e}")
    print("=" * 105)

    # =====================================================================
    # Plot: 4×2 grid
    # =====================================================================
    fig = plt.figure(figsize=(16, 16), facecolor="white")
    fig.suptitle(
        "SEI Growth & Cell Voltage — SoC₀ = 0.3 vs 0.9 (25 °C, 360 Days)",
        fontsize=14, fontweight="bold", y=0.98,
    )

    gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.30,
                  left=0.08, right=0.95, top=0.94, bottom=0.04)

    panels = [
        ("v_cell", "Cell voltage [V]", "(a) Cell voltage during storage", False),
        ("sei_nm", "SEI thickness [nm]", "(b) SEI thickness growth", False),
        ("soc", "SoC (neg. stoich.) [-]", "(c) Negative electrode stoichiometry", False),
        ("ocv", "Negative OCV [V]", "(d) Negative electrode OCV", False),
        ("jsei_abs", "|J_SEI| [A·m⁻²]", "(e) SEI interfacial current density", True),
        ("q_loss", "Charge loss [A·h·m⁻²]", "(f) Cumulative charge loss", False),
        ("eta_mV", "η_SEI [mV]", "(g) SEI overpotential", False),
    ]

    for idx, (key, ylabel, title, use_log) in enumerate(panels):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        for z0 in soc_values:
            r = results[z0]
            t = r["time_days"]
            c = colors[str(z0)]
            label = f"SoC₀ = {z0}"

            if key == "jsei_abs":
                ax.semilogy(t, np.abs(r["jsei"]), color=c, linewidth=2, label=label)
            elif key == "eta_mV":
                ax.plot(t, r["eta"] * 1e3, color=c, linewidth=2, label=label)
            else:
                ax.plot(t, r[key], color=c, linewidth=2, label=label)

        # Milestone markers
        for d in milestone_days:
            ax.axvline(d, color="gray", linewidth=0.5, alpha=0.3, linestyle=":")

        ax.set_xlabel("Time [days]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 360)

    # Fill the last subplot with annotation
    ax_info = fig.add_subplot(gs[3, 1])
    ax_info.axis("off")
    info_text = (
        "Cell Design: LiCoO₂ / Graphite (Sony 18650)\n\n"
        "Stoichiometry windows:\n"
        f"  Neg: x ∈ [{X_0}, {X_100}]\n"
        f"  Pos: y ∈ [{Y_100}, {Y_0}]\n\n"
        "During OCV storage:\n"
        "  • Positive stoich. y_pos = fixed\n"
        "  • Negative stoich. x_neg ↓ (Li consumed by SEI)\n"
        "  • V_cell = U_pos(y) − U_neg(x)\n\n"
        f"SoC₀=0.3: y_pos = {results[0.3]['y_pos']:.4f}\n"
        f"SoC₀=0.9: y_pos = {results[0.9]['y_pos']:.4f}"
    )
    ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
                 fontsize=11, verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.8))

    output_path = "compare_soc_03_09.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\nFigure saved to: {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    run_soc_comparison()
