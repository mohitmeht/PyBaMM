import pybamm as pb
import numpy as np
import matplotlib.pyplot as plt

def run_continuous_hybrid():
    pb.set_logging_level("INFO")
    
    # 1. Models & Options
    options = {"SEI": "madi-reddy"}
    model = pb.lithium_ion.DFN(options)
    
    # 2. Original Paper Parameters
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

    var_pts = {"x_n": 30, "x_s": 30, "x_p": 30, "r_n": 50, "r_p": 50}

    # 3. Define the experiment sequence
    cycle_steps = (
        "Discharge at C/9 until 3.2 V",
        "Rest for 15 minutes",
        "Charge at C/7 until 4.1 V",
        "Hold at 4.1 V until C/37",
        "Rest for 15 minutes",
        "Discharge at C/4 for 5s",
        "Rest for 15 minutes",
    )
    
    durations = [7, 14, 21, 30, 90]
    full_steps = []
    
    for days in durations:
        full_steps.append(f"Rest for {days * 24} hours")
        for _ in range(3):
            full_steps.append(cycle_steps)

    experiment = pb.Experiment(full_steps)

    print(f"Starting continuous hybrid experiment at 30% SoC with storage steps: {durations}...")
    solver = pb.CasadiSolver(mode="safe", atol=1e-6, rtol=1e-6)
    sim = pb.Simulation(model, parameter_values=param, var_pts=var_pts, experiment=experiment, solver=solver)
    sol = sim.solve(initial_soc=0.3, calc_esoh=False)

    # 5. Plotting
    variables = [
        "Terminal voltage [V]",
        "X-averaged negative SEI thickness [m]",
        "Loss of lithium inventory [%]",
        "X-averaged negative electrode interfacial current density [A.m-2]"
    ]
    
    n_vars = len(variables)
    fig, axes = plt.subplots(n_vars, 1, figsize=(12, 3 * n_vars), sharex=True)
    
    time_h = sol["Time [h]"].entries
    
    for i, var in enumerate(variables):
        axes[i].plot(time_h, sol[var].entries, color="#457b9d", linewidth=1.5)
        axes[i].set_ylabel(var)
        axes[i].grid(True, alpha=0.3)
        
        temp_time = 0
        for days in durations:
            axes[i].axvline(temp_time, color="#e63946", linestyle="--", alpha=0.4)
            temp_time += days*24 + 3*20 

    axes[-1].set_xlabel("Time [h]")
    plt.suptitle("Continuous Hybrid SEI Aging Experiment at 30% SoC")
    plt.tight_layout()
    plt.savefig("continuous_hybrid_30soc_experiment.png")
    print("\nExperiment complete. Results saved to continuous_hybrid_30soc_experiment.png")

if __name__ == "__main__":
    run_continuous_hybrid()
