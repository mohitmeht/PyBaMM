"""
Madi-Reddy Mixed-Mode SEI Growth Model Under OCV Storage
=========================================================

Standalone PyBaMM implementation of the transient solvent-diffusion PDE
inside the SEI on a moving domain, using the Landau transform (ξ = x/L(t))
for a fixed computational domain.

Governing equations (Landau-transformed):
  ∂c_s/∂t = (D_eff/L²) ∂²c_s/∂ξ² + (ξ/L)(dL/dt) ∂c_s/∂ξ
  dc_Li/dt = a · J_SEI / (n·F)
  dL/dt    = -J_SEI / (n·F·c_p)

  J_SEI = -J_SEI,0 · (c_s(1,t)/c_s,max) · (c_Li/c_Li,max)² · exp(-α·n·F·η_SEI/(R·T))
  J_SEI,0 = n·F·k_SEI · c_Li,max^(n·(1-α))
  η_SEI = U_OCV(z) - U_SEI,   z = c_Li / c_Li,max

BCs:
  c_s(ξ=0, t) = ε_SEI · c_s,bulk              (Dirichlet)
  -(D_eff/L)(∂c_s/∂ξ)|_{ξ=1} = -J_SEI/(n·F)  (flux / Neumann)

Reference: Madi & Reddy paper, Tables I-III, Appendix A (OCV) & B (fixed domain).
25 °C fitted parameters from Table II.
"""

import numpy as np
import pybamm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# =========================================================================
# Graphite OCV (Appendix A)
# =========================================================================
def graphite_ocv(z):
    """
    Appendix A in the paper.
    z = SoC = c_Li / c_Li,max
    """
    z = pybamm.maximum(pybamm.minimum(z, 0.999999), 1e-6)
    return (
        0.7222
        + 0.1387 * z
        + 0.029 * pybamm.sqrt(z)
        - 0.0172 / z
        + 0.0019 / (z ** 1.5)
        + 0.2802 * pybamm.exp(0.9 - 15 * z)
        - 0.7984 * pybamm.exp(0.4465 * z - 0.4108)
    )


# =========================================================================
# Model Class
# =========================================================================
class MadiReddySEIModel(pybamm.BaseModel):
    """
    Standalone mixed-mode SEI growth model under OCV storage.

    States:
        c_li(t)              lithium concentration in negative electrode
        L(t)                 SEI thickness
        q_loss(t)            cumulative charge loss per area [C.m-2]
        c_i(t), i=1..N       solvent concentration at ξ_i in fixed domain ξ=x/L
    """

    def __init__(self, N=40):
        super().__init__(name="Madi-Reddy mixed-mode SEI under OCV storage")
        if N < 2:
            raise ValueError("Use N >= 2")

        # -----------------------------------------------------------------
        # Parameters
        # -----------------------------------------------------------------
        F = pybamm.Parameter("Faraday constant [C.mol-1]")
        R = pybamm.Parameter("Gas constant [J.mol-1.K-1]")
        T = pybamm.Parameter("Temperature [K]")

        n = pybamm.Parameter("SEI electrons transferred [-]")
        alpha = pybamm.Parameter("SEI transfer coefficient [-]")
        a = pybamm.Parameter("Specific surface area [m-1]")

        c_li_max = pybamm.Parameter("Maximum lithium concentration [mol.m-3]")
        c_s_bulk = pybamm.Parameter("Bulk solvent concentration [mol.m-3]")
        c_s_max = pybamm.Parameter("Maximum solvent concentration [mol.m-3]")

        eps_sei = pybamm.Parameter("SEI porosity [-]")
        D_eff = pybamm.Parameter("Effective solvent diffusivity in SEI [m2.s-1]")
        k_sei = pybamm.Parameter("SEI kinetic rate constant [m.s-1]")
        U_sei = pybamm.Parameter("SEI equilibrium potential [V]")
        c_p = pybamm.Parameter("SEI product concentration [mol.m-3]")

        L0 = pybamm.Parameter("Initial SEI thickness [m]")
        z0 = pybamm.Parameter("Initial SoC [-]")

        # -----------------------------------------------------------------
        # Main lumped states
        # -----------------------------------------------------------------
        c_li = pybamm.Variable("Lithium concentration [mol.m-3]")
        L = pybamm.Variable("SEI thickness [m]")
        q_loss = pybamm.Variable("Charge loss per area [C.m-2]")

        # -----------------------------------------------------------------
        # Fixed-domain solvent variables
        # ξ_i = i/N, i=1,...,N
        # ξ=0 is Dirichlet boundary (not a state variable)
        # ξ=1 is included as the last state
        # -----------------------------------------------------------------
        h = 1 / N
        xi_nodes = np.arange(1, N + 1) / N
        c = [
            pybamm.Variable(f"Solvent concentration node {i} [mol.m-3]")
            for i in range(1, N + 1)
        ]

        c_left = eps_sei * c_s_bulk   # Dirichlet at ξ=0
        c_right = c[-1]               # interfacial value at ξ=1

        # -----------------------------------------------------------------
        # OCV, overpotential, interfacial current
        # -----------------------------------------------------------------
        z = c_li / c_li_max
        U_ocv = graphite_ocv(z)
        eta_sei = U_ocv - U_sei

        j0_sei = n * F * k_sei * (c_li_max ** (n * (1 - alpha)))
        j_sei = (
            -j0_sei
            * (c_right / c_s_max)
            * ((c_li ** 2) / (c_li_max ** 2))
            * pybamm.exp(-alpha * n * F * eta_sei / (R * T))
        )

        # -----------------------------------------------------------------
        # Thickness evolution: dL/dt = -j_sei / (n·F·c_p)
        # -----------------------------------------------------------------
        Ldot = -j_sei / (n * F * c_p)

        # -----------------------------------------------------------------
        # Lithium balance:
        # (1/a) dc_li/dt = j_sei / (n·F)  =>  dc_li/dt = a · j_sei / (n·F)
        # -----------------------------------------------------------------
        self.rhs[c_li] = a * j_sei / (n * F)

        # -----------------------------------------------------------------
        # Cumulative lost charge:
        # dq_loss/dt = |j_sei|  (j_sei is negative, so q_loss grows)
        # -----------------------------------------------------------------
        self.rhs[q_loss] = -j_sei

        # -----------------------------------------------------------------
        # SEI thickness ODE
        # -----------------------------------------------------------------
        self.rhs[L] = Ldot

        # -----------------------------------------------------------------
        # Flux BC at ξ = 1:
        # -(D_eff/L) dc/dξ|_(ξ=1) = -j_sei/(n·F)
        # => dc/dξ|_(ξ=1) = (L/D_eff) · j_sei/(n·F)
        # -----------------------------------------------------------------
        dcdxi_right = (L / D_eff) * (j_sei / (n * F))

        # Ghost node for second-order closure at ξ=1:
        # (c_ghost - c_{N-1})/(2h) = dcdxi_right
        c_ghost = c[-2] + 2 * h * dcdxi_right

        # -----------------------------------------------------------------
        # Fixed-domain PDE:
        # dc/dt = (D_eff/L²) d²c/dξ² + (ξ/L)·Ldot·dc/dξ
        # -----------------------------------------------------------------
        for i in range(N):
            c_im1 = c_left if i == 0 else c[i - 1]
            c_i = c[i]
            c_ip1 = c_ghost if i == N - 1 else c[i + 1]

            dc_dxi = (c_ip1 - c_im1) / (2 * h)
            d2c_dxi2 = (c_ip1 - 2 * c_i + c_im1) / (h ** 2)

            self.rhs[c_i] = (
                (D_eff / (L ** 2)) * d2c_dxi2
                + (xi_nodes[i] * Ldot / L) * dc_dxi
            )

        # -----------------------------------------------------------------
        # Initial conditions
        # -----------------------------------------------------------------
        self.initial_conditions[c_li] = z0 * c_li_max
        self.initial_conditions[L] = L0
        self.initial_conditions[q_loss] = pybamm.Scalar(0)

        for ci in c:
            self.initial_conditions[ci] = c_left

        # -----------------------------------------------------------------
        # Output variables
        # -----------------------------------------------------------------
        _vars = {
            "SEI thickness [m]": L,
            "SEI thickness [nm]": 1e9 * L,
            "Lithium concentration [mol.m-3]": c_li,
            "SoC [-]": z,
            "Negative OCV [V]": U_ocv,
            "SEI overpotential [V]": eta_sei,
            "SEI current density [A.m-2]": j_sei,
            "Charge loss per area [C.m-2]": q_loss,
            "Charge loss per area [A.h.m-2]": q_loss / 3600,
        }

        # Register all solvent nodes first (so node N exists under its own name)
        for i, ci in enumerate(c, start=1):
            _vars[f"Solvent concentration node {i} [mol.m-3]"] = ci

        # Now add the alias for the interfacial node (node N)
        _vars["Interfacial solvent concentration [mol.m-3]"] = c_right

        self.variables = _vars


# =========================================================================
# Parameter set: 25 °C — Table I + MATLAB fit from Table II
# =========================================================================
param = pybamm.ParameterValues(
    {
        "Faraday constant [C.mol-1]": 96485.0,
        "Gas constant [J.mol-1.K-1]": 8.3145,
        "Temperature [K]": 298.15,
        "SEI electrons transferred [-]": 2.0,
        "SEI transfer coefficient [-]": 0.5,
        "Specific surface area [m-1]": 3.0e6,
        "Maximum lithium concentration [mol.m-3]": 3.056e4,
        "Bulk solvent concentration [mol.m-3]": 4541.0,
        "Maximum solvent concentration [mol.m-3]": 4541.0,
        "SEI porosity [-]": 0.05,
        "Effective solvent diffusivity in SEI [m2.s-1]": 1.4029e-19,
        "SEI kinetic rate constant [m.s-1]": 1.6434e-17,
        "SEI equilibrium potential [V]": 0.4,
        "SEI product concentration [mol.m-3]": 28556.0,
        "Initial SEI thickness [m]": 1e-12,
        "Initial SoC [-]": 1.0,
    }
)


# =========================================================================
# Build, solve, and plot
# =========================================================================
def main():
    N = 40
    model = MadiReddySEIModel(N=N)
    solver = pybamm.CasadiSolver(mode="safe")
    sim = pybamm.Simulation(model, parameter_values=param, solver=solver)

    # Simulate 1 year of OCV storage
    t_eval = np.linspace(0, 365 * 24 * 3600, 800)

    print("Solving Madi-Reddy SEI model (N={}, 1 year OCV storage)...".format(N))
    sol = sim.solve(t_eval)
    print("  Done. Final time: {:.1f} days".format(sol.t[-1] / 86400))

    # =====================================================================
    # Extract solution data
    # =====================================================================
    time_days = sol.t / 86400
    sei_nm = sol["SEI thickness [nm]"].entries
    ocv = sol["Negative OCV [V]"].entries
    soc = sol["SoC [-]"].entries
    jsei = sol["SEI current density [A.m-2]"].entries
    q_loss_ah_m2 = sol["Charge loss per area [A.h.m-2]"].entries
    eta = sol["SEI overpotential [V]"].entries
    c_int = sol["Interfacial solvent concentration [mol.m-3]"].entries

    # Solvent concentration profiles at selected times
    n_snapshots = 6
    snapshot_indices = np.linspace(0, len(time_days) - 1, n_snapshots, dtype=int)
    xi_nodes = np.arange(1, N + 1) / N

    # =====================================================================
    # Plot: 6-panel figure
    # =====================================================================
    fig = plt.figure(figsize=(16, 12), facecolor="white")
    fig.suptitle(
        "Madi-Reddy Mixed-Mode SEI Growth — 25 °C OCV Storage (1 Year)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.30,
                  left=0.08, right=0.95, top=0.93, bottom=0.06)

    # --- Panel 1: SEI thickness ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time_days, sei_nm, color="#2563eb", linewidth=2)
    ax1.set_xlabel("Time [days]")
    ax1.set_ylabel("SEI thickness [nm]")
    ax1.set_title("(a) SEI thickness growth")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, time_days[-1])

    # --- Panel 2: Negative OCV & SoC ---
    ax2a = fig.add_subplot(gs[0, 1])
    color_ocv = "#dc2626"
    color_soc = "#059669"
    ax2a.plot(time_days, ocv, color=color_ocv, linewidth=2, label="OCV")
    ax2a.set_xlabel("Time [days]")
    ax2a.set_ylabel("Negative OCV [V]", color=color_ocv)
    ax2a.tick_params(axis="y", labelcolor=color_ocv)
    ax2a.set_title("(b) Negative electrode OCV & SoC")
    ax2a.grid(True, alpha=0.3)
    ax2a.set_xlim(0, time_days[-1])

    ax2b = ax2a.twinx()
    ax2b.plot(time_days, soc, color=color_soc, linewidth=2, linestyle="--", label="SoC")
    ax2b.set_ylabel("SoC [-]", color=color_soc)
    ax2b.tick_params(axis="y", labelcolor=color_soc)

    lines1, labels1 = ax2a.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2a.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    # --- Panel 3: SEI current density ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.semilogy(time_days, np.abs(jsei), color="#7c3aed", linewidth=2)
    ax3.set_xlabel("Time [days]")
    ax3.set_ylabel("|J_SEI| [A·m⁻²]")
    ax3.set_title("(c) SEI interfacial current density")
    ax3.grid(True, which="both", alpha=0.3)
    ax3.set_xlim(0, time_days[-1])

    # --- Panel 4: Charge loss ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(time_days, q_loss_ah_m2, color="#d97706", linewidth=2)
    ax4.set_xlabel("Time [days]")
    ax4.set_ylabel("Charge loss [A·h·m⁻²]")
    ax4.set_title("(d) Cumulative charge loss per area")
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, time_days[-1])

    # --- Panel 5: Solvent concentration profile snapshots ---
    ax5 = fig.add_subplot(gs[2, 0])
    cmap = plt.cm.viridis
    for idx, si in enumerate(snapshot_indices):
        c_profile = np.array([
            sol[f"Solvent concentration node {j} [mol.m-3]"].entries[si]
            for j in range(1, N + 1)
        ])
        color = cmap(idx / (n_snapshots - 1))
        label_txt = f"t = {time_days[si]:.0f} d"
        ax5.plot(xi_nodes, c_profile, color=color, linewidth=1.5, label=label_txt)

    ax5.set_xlabel("ξ = x / L(t)")
    ax5.set_ylabel("Solvent concentration [mol·m⁻³]")
    ax5.set_title("(e) Solvent conc. profiles in SEI")
    ax5.legend(fontsize=8, loc="best")
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0, 1)

    # --- Panel 6: SEI overpotential ---
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(time_days, eta * 1e3, color="#0891b2", linewidth=2)
    ax6.set_xlabel("Time [days]")
    ax6.set_ylabel("η_SEI [mV]")
    ax6.set_title("(f) SEI overpotential")
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(0, time_days[-1])

    # Save figure
    output_path = "madi_reddy_sei_results.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"  Figure saved to: {output_path}")
    plt.close(fig)

    # =====================================================================
    # Print summary
    # =====================================================================
    print("\n" + "=" * 60)
    print("Solution Summary")
    print("=" * 60)
    print(f"  Initial SoC:       {soc[0]:.4f}")
    print(f"  Final SoC:         {soc[-1]:.4f}")
    print(f"  Initial SEI:       {sei_nm[0]:.4f} nm")
    print(f"  Final SEI:         {sei_nm[-1]:.2f} nm")
    print(f"  OCV (t=0):         {ocv[0]:.4f} V")
    print(f"  OCV (t=365d):      {ocv[-1]:.4f} V")
    print(f"  Total charge loss: {q_loss_ah_m2[-1]:.4e} A·h·m⁻²")
    print(f"  |J_SEI| at t=0:    {np.abs(jsei[0]):.4e} A·m⁻²")
    print(f"  |J_SEI| at t=365d: {np.abs(jsei[-1]):.4e} A·m⁻²")
    print("=" * 60)

    # =====================================================================
    # Also save raw data to CSV for further analysis
    # =====================================================================
    csv_path = "madi_reddy_sei_timeseries.csv"
    data = np.column_stack([
        time_days, sei_nm, ocv, soc, jsei, q_loss_ah_m2, eta, c_int
    ])
    header = (
        "Time_days,SEI_thickness_nm,Negative_OCV_V,SoC,"
        "J_SEI_A_per_m2,Q_loss_Ah_per_m2,eta_SEI_V,c_s_interface_mol_per_m3"
    )
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="")
    print(f"  Time-series data saved to: {csv_path}")


if __name__ == "__main__":
    main()
