import pybamm as pb
import numpy as np
import matplotlib.pyplot as plt

def run_hybrid_sim():
    pb.set_logging_level("INFO")
    
    # 1. Model & Options
    options = {"SEI": "madi-reddy"}
    model = pb.lithium_ion.DFN(options)
    
    # 2. Optimized Parameters (Final Tuned Set)
    param = pb.ParameterValues("OKane2022")
    eps_s = param["Negative electrode active material volume fraction"]
    param.update({
        "Negative particle radius [m]": 3.0 * eps_s / 3.0e6,
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
        "Current function [A]": 0,
    }, check_already_exists=False)

    # Mesh resolution (Colab standard)
    var_pts = {"x_n": 30, "x_s": 30, "x_p": 30, "r_n": 50, "r_p": 50}

    # =====================================================================
    # PHASE 1: 14-Day OCV Storage
    # =====================================================================
    print("Starting Phase 1: 14-day OCV storage...")
    storage_time = 14 * 24 * 3600
    t_eval = np.linspace(0, storage_time, 100)
    
    sim_storage = pb.Simulation(model, parameter_values=param, var_pts=var_pts)
    sol_storage = sim_storage.solve(t_eval=t_eval, initial_soc=0.9)

    # =====================================================================
    # PHASE 2: 3-Cycle Experiment
    # =====================================================================
    print("\nStarting Phase 2: 3 cycles...")
    # Define the experiment based on calendar_ageing_and_cycling.py
    experiment = pb.Experiment([
        (
            "Discharge at C/9 until 3.2 V",
            "Rest for 15 minutes",
            "Charge at C/7 until 4.1 V",
            "Hold at 4.1 V until C/37",
            "Rest for 15 minutes",
            "Discharge at C/4 for 5s",
            "Rest for 15 minutes",
        )
    ] * 3)

    # Create a new simulation starting from the end of storage
    # We pass the ending solution as the starting point
    sim_cycling = pb.Simulation(model, parameter_values=param, var_pts=var_pts, experiment=experiment)
    sol_cycling = sim_cycling.solve(starting_solution=sol_storage)

    # =====================================================================
    # Combine and Plot
    # =====================================================================
    # For plotting, we can just use sol_cycling which contains the full history 
    # if solved correctly with starting_solution
    # =====================================================================
    # Plot
    # =====================================================================
    variables = [
        "Terminal voltage [V]",
        "X-averaged negative SEI thickness [m]",
        "Loss of lithium inventory [%]",
        "X-averaged negative electrode interfacial current density [A.m-2]"
    ]
    
    n_vars = len(variables)
    fig, axes = plt.subplots(n_vars, 1, figsize=(10, 3 * n_vars))
    
    for i, var in enumerate(variables):
        t_s = sol_storage["Time [h]"].entries
        v_s = sol_storage[var].entries
        t_c = sol_cycling["Time [h]"].entries
        v_c = sol_cycling[var].entries
        
        axes[i].plot(t_s, v_s, color="#1d3557", linewidth=1.5, label="Storage")
        axes[i].plot(t_c, v_c, color="#e63946", linewidth=1.5, label="Cycling")
        
        axes[i].set_title(var)
        axes[i].set_xlabel("Time [h]")
        axes[i].grid(True, alpha=0.3)
        axes[i].axvline(14*24, color="gray", linestyle="--", alpha=0.5)
        axes[i].legend()

    plt.tight_layout()
    plt.savefig("madi_14d_3cyc_results.png")
    print("\nDone. Results saved to madi_14d_3cyc_results.png")

if __name__ == "__main__":
    run_hybrid_sim()
