import pybamm as pb
import numpy as np
import matplotlib.pyplot as plt

def run_comparison():
    pb.set_logging_level("INFO")
    
    # 1. Models
    # Madi-Reddy (Mixed-Mode)
    options_mr = {"SEI": "madi-reddy"}
    model_mr = pb.lithium_ion.DFN(options_mr)
    
    # Mohtat-style (Pure Solvent Diffusion)
    options_sd = {"SEI": "solvent-diffusion limited"}
    model_sd = pb.lithium_ion.DFN(options_sd)
    
    # 2. Parameters
    # Madi-Reddy Base (OKane2022 + MR parameters)
    param_mr = pb.ParameterValues("OKane2022")
    eps_s = param_mr["Negative electrode active material volume fraction"]
    param_mr.update({
        "Negative particle radius [m]": 3.0 * eps_s / 3.0e6,
        "Negative electrode maximum concentration [mol.m-3]": 3.056e4,
        "Ratio of lithium moles to SEI moles": 2.0,
        "SEI kinetic rate constant [m.s-1]": 1.6434e-18,  # Our reduced rate
        "EC diffusivity [m2.s-1]": 1.4029e-19,
        "EC initial concentration in electrolyte [mol.m-3]": 4541.0,
        "Negative electrode SEI porosity": 0.05,
        "Initial SEI thickness [m]": 5e-9,  
        "SEI open-circuit potential [V]": 0.4,
        "SEI partial molar volume [m3.mol-1]": 9.585e-05,
        "SEI growth transfer coefficient": 0.5,
        "Current function [A]": 0,
    }, check_already_exists=False)

    # Mohtat2020 parameters (Pure Diffusion)
    param_sd = pb.ParameterValues("Mohtat2020")
    param_sd.update({
        "Current function [A]": 0,
        "Initial SEI thickness [m]": 5e-9, # matching start
    }, check_already_exists=False)

    # 3. Solve
    t_eval = np.linspace(0, 365 * 24 * 3600, 200)
    
    print("Solving Madi-Reddy (1 year, SOC 0.9)...")
    sim_mr = pb.Simulation(model_mr, parameter_values=param_mr)
    sol_mr = sim_mr.solve(t_eval=t_eval, initial_soc=0.9)
    
    print("Solving Mohtat2020 Diffusion Limited (1 year, SOC 0.9)...")
    sim_sd = pb.Simulation(model_sd, parameter_values=param_sd)
    sol_sd = sim_sd.solve(t_eval=t_eval, initial_soc=0.9)

    # 4. Plotting
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # SEI Thickness
    axes[0].plot(sol_mr["Time [h]"].entries, sol_mr["X-averaged negative SEI thickness [m]"].entries * 1e9, 
                 label="Madi-Reddy (Mixed-Mode, kr=1.6e-18)", color="#1d3557")
    axes[0].plot(sol_sd["Time [h]"].entries, sol_sd["X-averaged negative SEI thickness [m]"].entries * 1e9, 
                 label="Mohtat2020 (Pure Diffusion)", color="#e63946", linestyle="--")
    axes[0].set_ylabel("SEI thickness [nm]")
    axes[0].set_title("1-Year Comparison: Madi-Reddy vs Mohtat2020 (SoC 0.9)")
    axes[0].legend()
    axes[0].grid(True)

    # LLI
    axes[1].plot(sol_mr["Time [h]"].entries, sol_mr["Loss of lithium inventory [%]"].entries, 
                 label="Madi-Reddy (LLI)", color="#1d3557")
    axes[1].plot(sol_sd["Time [h]"].entries, sol_sd["Loss of lithium inventory [%]"].entries, 
                 label="Mohtat2020 (LLI)", color="#e63946", linestyle="--")
    axes[1].set_ylabel("LLI [%]")
    axes[1].set_xlabel("Time [h]")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("madi_vs_mohtat_1yr.png")
    print("Results saved to madi_vs_mohtat_1yr.png")

if __name__ == "__main__":
    run_comparison()
