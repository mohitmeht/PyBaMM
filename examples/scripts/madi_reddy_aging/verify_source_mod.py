import pybamm
import matplotlib.pyplot as plt
import numpy as np

def verify_madi_reddy():
    # 1. Setup model with 'madi-reddy' SEI
    options = {"SEI": "madi-reddy"}
    model = pybamm.lithium_ion.DFN(options=options)

    # 2. Parameters (Mohtat 2020 + Madi-Reddy sync)
    params = pybamm.ParameterValues("Mohtat2020")
    
    # Paper uses a = 3e6. Mohtat uses eps_s = 0.61. 
    # Match a = 3*eps_s/R -> R = 3*0.61/3e6 = 6.1e-7
    params.update({
        "Negative particle radius [m]": 6.1e-7,
        "Negative electrode maximum concentration [mol.m-3]": 3.056e4,
        "Ratio of lithium moles to SEI moles": 2.0,
        "SEI kinetic rate constant [m.s-1]": 1.6434e-17,
        "EC diffusivity [m2.s-1]": 1.4029e-19,
        "EC initial concentration in electrolyte [mol.m-3]": 4541.0,
        "Negative electrode SEI porosity": 0.05,
        "Initial SEI thickness [m]": 5e-9,
        "SEI open-circuit potential [V]": 0.4,
        "SEI partial molar volume [m3.mol-1]": 9.585e-05,
        "SEI growth transfer coefficient": 0.5,
    }, check_already_exists=False)
    
    # 3. Solve (OCV rest)
    sim = pybamm.Simulation(model, parameter_values=params)
    
    solver = pybamm.CasadiSolver(mode="safe")
    t_eval = np.linspace(0, 30 * 24 * 3600, 100)
    
    print("Solving DFN with 'madi-reddy' SEI model (Mohtat2022/2020)...")
    sim = pybamm.Simulation(model, parameter_values=params, solver=solver)
    sol = sim.solve(t_eval)

    # 4. Verify results
    L_sei = sol["X-averaged negative SEI thickness [m]"].data
    j_sei = sol["X-averaged negative electrode SEI interfacial current density [A.m-2]"].data
    t_days = sol.t / (24 * 3600)
    
    print(f"Simulation successful!")
    print(f"Initial SEI thickness: {L_sei[0]*1e9:.2f} nm")
    print(f"Final SEI thickness (end of sim): {L_sei[-1]*1e9:.2f} nm")
    print(f"Initial SEI current density: {j_sei[0]:.2e} A.m-2")
    print(f"Min j_sei: {np.min(j_sei):.2e} A.m-2")
    print(f"Max j_sei: {np.max(j_sei):.2e} A.m-2")
    
    Q_loss = sol["Loss of lithium to negative SEI [mol]"].data
    print(f"Initial Li loss: {Q_loss[0]:.2e} mol")
    print(f"Final Li loss: {Q_loss[-1]:.2e} mol")
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    ax[0, 0].plot(t_days, L_sei * 1e9)
    ax[0, 0].set_xlabel("Time [days]")
    ax[0, 0].set_ylabel("SEI thickness [nm]")
    ax[0, 0].set_title("SEI Growth")
    ax[0, 0].grid(True)

    ax[0, 1].plot(t_days, sol["Terminal voltage [V]"].data)
    ax[0, 1].set_xlabel("Time [days]")
    ax[0, 1].set_ylabel("Voltage [V]")
    ax[0, 1].set_title("Cell Voltage")
    ax[0, 1].grid(True)
    
    ax[1, 0].plot(t_days, j_sei)
    ax[1, 0].set_xlabel("Time [days]")
    ax[1, 0].set_ylabel("j_sei [A.m-2]")
    ax[1, 0].set_title("SEI Current Density")
    ax[1, 0].grid(True)
    
    delta_phi = sol["X-averaged negative electrode surface potential difference [V]"].data
    ax[1, 1].plot(t_days, delta_phi)
    ax[1, 1].set_xlabel("Time [days]")
    ax[1, 1].set_ylabel("Delta phi [V]")
    ax[1, 1].set_title("Surface Potential Diff")
    ax[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig("verification_source_mod.png")
    print("Verification plot saved to verification_source_mod.png")

if __name__ == "__main__":
    verify_madi_reddy()
