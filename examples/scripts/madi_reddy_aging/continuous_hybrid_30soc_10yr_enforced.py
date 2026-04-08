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
    
    # Initial 10-year Phase: (87600 hours, 6mo period)
    full_steps = [pb.step.string("Rest for 87600 hours", period="4380 hours")]
    
    for days in durations:
        # Storage segment with daily sampling
        full_steps.append(pb.step.string(f"Rest for {days * 24} hours", period="24 hours"))
        
        # Testing segment (3 cycles)
        for _ in range(3):
            full_steps.append(cycle_steps)
            
        # NEW: Enforce return to 30% SoC before the NEXT storage period
        # For this DFN model, ~3.6V corresponds to roughly 30% SoC
        full_steps.append("Discharge at C/3 until 3.6 V")
        full_steps.append("Rest for 1 hour")

    experiment = pb.Experiment(full_steps)

    print(f"Starting experiment with ENFORCED 30% SoC storage plateau...")
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
        axes[i].plot(time_h, sol[var].entries, color="#2a9d8f", linewidth=1.5)
        axes[i].set_ylabel(var)
        axes[i].grid(True, alpha=0.3)
        
        # Highlight the 10-year transition
        axes[i].axvline(87600, color="#e9c46a", linestyle="-", linewidth=2.0)

    axes[-1].set_xlabel("Time [h]")
    plt.suptitle("Aging Experiment: True 30% SoC Storage (10yr Pre-Aging + Hybrid Cycling)")
    plt.tight_layout()
    plt.savefig("continuous_hybrid_30soc_10yr_enforced.png")
    print("\nExperiment complete. Results saved to continuous_hybrid_30soc_10yr_enforced.png")

if __name__ == "__main__":
    run_continuous_hybrid()
