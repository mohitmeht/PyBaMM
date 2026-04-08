import pybamm as pb
import numpy as np
import matplotlib.pyplot as plt

def run_zoomed_plot():
    pb.set_logging_level("INFO")
    
    # 1. Models & Options
    options = {"SEI": "madi-reddy"}
    model = pb.lithium_ion.DFN(options)
    
    # 2. Parameters (Enforced 30% SoC settings)
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

    # 3. Experiment Sequence (Enforced 30% Storage)
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
    full_steps = [pb.step.string("Rest for 87600 hours", period="4380 hours")]
    for days in durations:
        full_steps.append(pb.step.string(f"Rest for {days * 24} hours", period="24 hours"))
        for _ in range(3):
            full_steps.append(cycle_steps)
        full_steps.append("Discharge at C/3 until 3.6 V")
        full_steps.append("Rest for 1 hour")

    experiment = pb.Experiment(full_steps)

    # 4. Solve
    solver = pb.CasadiSolver(mode="safe", atol=1e-6, rtol=1e-6)
    sim = pb.Simulation(model, parameter_values=param, var_pts=var_pts, experiment=experiment, solver=solver)
    sol = sim.solve(initial_soc=0.3, calc_esoh=False)

    # 5. Targeted Plotting (Post 10-year only)
    variables = [
        "Terminal voltage [V]",
        "Loss of lithium inventory [%]"
    ]
    
    time_h = sol["Time [h]"].entries
    mask = time_h >= 87600
    
    # Filter the data
    time_zoomed = time_h[mask] - 87600 # Offset to 0 for the testing phase
    
    n_vars = len(variables)
    fig, axes = plt.subplots(n_vars, 1, figsize=(12, 4 * n_vars), sharex=True)
    
    for i, var in enumerate(variables):
        var_data = sol[var].entries[mask]
        axes[i].plot(time_zoomed, var_data, color="#1d3557", linewidth=2.0)
        axes[i].set_ylabel(var)
        axes[i].grid(True, alpha=0.3)
        
    axes[-1].set_xlabel("Time since Start of Testing [h]")
    plt.suptitle("Simplified View: Voltage & LLI during Hybrid Testing Phase (30% SoC Enforced)")
    plt.tight_layout()
    plt.savefig("testing_phase_only_30soc.png")
    print("\nZoomed plot complete. Results saved to testing_phase_only_30soc.png")

if __name__ == "__main__":
    run_zoomed_plot()
