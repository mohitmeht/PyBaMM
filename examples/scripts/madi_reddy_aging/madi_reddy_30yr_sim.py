import numpy as np
import pybamm as pb
import matplotlib.pyplot as plt

pb.set_logging_level("INFO")

def run_30yr_madi_reddy():
    pb.set_logging_level("INFO")

    # 1. Model Selection
    options = {"SEI": "madi-reddy"}
    model = pb.lithium_ion.DFN(options)

    # 2. Base parameters
    parameter_values = pb.ParameterValues("OKane2022")
    eps_s = parameter_values["Negative electrode active material volume fraction"]

    parameter_values.update(
        {
            "Negative particle radius [m]": 3.0 * eps_s / 3.0e6,
            "Negative electrode maximum concentration [mol.m-3]": 3.056e4,
            "Ratio of lithium moles to SEI moles": 2.0,
            "SEI kinetic rate constant [m.s-1]": 1.6434e-17,
            "EC diffusivity [m2.s-1]": 1.4029e-19,
            "EC initial concentration in electrolyte [mol.m-3]": 4541.0,
            "Negative electrode SEI porosity": 0.05,
            "Initial SEI thickness [m]": 5e-9,  # 5 nm initial SEI
            "SEI open-circuit potential [V]": 0.4,
            "SEI partial molar volume [m3.mol-1]": 9.585e-05,
            "SEI growth transfer coefficient": 0.5,
            "Current function [A]": 0,
        },
        check_already_exists=False,
    )

    # 3. Setup
    solver = pb.IDAKLUSolver()

    # Time configuration
    days = 365
    hours = days * 24
    minutes = hours * 60
    seconds = minutes * 60

    t_eval = np.linspace(0, seconds, hours + 1)

    # 4. Solves
    # Use increased mesh resolution from Colab code
    var_pts = {"x_n": 30, "x_s": 30, "x_p": 30, "r_n": 50, "r_p": 50}
    
    print(f"Starting {days}-day (1-year) simulation with 'madi-reddy' model (N=100) at SoC 0.9...")
    sim_90 = pb.Simulation(
        model, 
        parameter_values=parameter_values, 
        var_pts=var_pts
    )
    sol_90 = sim_90.solve(t_eval=t_eval, solver=solver, initial_soc=0.9)
    # Using sol_90 directly as solve returns the solution

    print(f"Starting {days}-day (1-year) simulation at SoC 0.3...")
    sim_30 = pb.Simulation(
        model, 
        parameter_values=parameter_values, 
        var_pts=var_pts
    )
    sol_30 = sim_30.solve(t_eval=t_eval, solver=pb.IDAKLUSolver(), initial_soc=0.3)

    # 5. Plotting (Save to file instead of dynamic)
    variables = [
        "Voltage [V]",
        "X-averaged negative particle surface concentration [mol.m-3]",
        "X-averaged electrolyte concentration [mol.m-3]",
        "X-averaged negative SEI concentration [mol.m-3]",
        "X-averaged negative SEI thickness [m]",
        "Sum of x-averaged negative electrode volumetric "
        "interfacial current densities [A.m-3]",
        "Loss of lithium inventory [%]",
        "Total lithium lost [mol]", 
        "Loss of lithium to negative SEI [mol]"
    ]
    
    # Create static plot for reporting
    n_vars = len(variables)
    fig, axes = plt.subplots(n_vars, 1, figsize=(10, 4 * n_vars))
    
    time_unit = "Time [h]"
    
    for i, var_name in enumerate(variables):
        ax = axes[i]
        
        ax.plot(sol_90[time_unit].entries, sol_90[var_name].entries, label="SOC 0.9", color="#1d3557")
        ax.plot(sol_30[time_unit].entries, sol_30[var_name].entries, label="SOC 0.3", color="#e63946")
        
        ax.set_title(str(var_name))
        ax.set_xlabel(time_unit)
        ax.grid(True)
        ax.legend()
        
        # Disable offset notation so values like 999.9 don't display as -0.1 with a +1e3 offset
        ax.yaxis.get_major_formatter().set_useOffset(False)

    plt.tight_layout()
    plt.savefig("madi_reddy_1year_results.png")
    print("Results saved to madi_reddy_1year_results.png")

if __name__ == "__main__":
    run_30yr_madi_reddy()
