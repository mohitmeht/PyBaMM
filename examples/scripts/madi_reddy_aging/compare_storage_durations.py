import pybamm as pb
import numpy as np
import matplotlib.pyplot as plt

def run_storage_sweep():
    pb.set_logging_level("ERROR") # Reduced logging for cleaner output
    
    # 1. Parameter Set (Original Paper Values)
    param = pb.ParameterValues("OKane2022")
    eps_s = param["Negative electrode active material volume fraction"]
    param.update({
        "Negative particle radius [m]": 3.0 * eps_s / 3.0e6,
        "Negative electrode maximum concentration [mol.m-3]": 3.056e4,
        "Ratio of lithium moles to SEI moles": 2.0,
        "SEI kinetic rate constant [m.s-1]": 1.6434e-17, # Paper
        "EC diffusivity [m2.s-1]": 1.4029e-19,         # Paper
        "EC initial concentration in electrolyte [mol.m-3]": 4541.0,
        "Negative electrode SEI porosity": 0.05,
        "Initial SEI thickness [m]": 5e-9,              # Paper
        "SEI open-circuit potential [V]": 0.4,
        "SEI partial molar volume [m3.mol-1]": 9.585e-05,
        "SEI growth transfer coefficient": 0.5,
        "Current function [A]": 0,
    }, check_already_exists=False)

    options = {"SEI": "madi-reddy"}
    model = pb.lithium_ion.DFN(options)
    var_pts = {"x_n": 30, "x_s": 30, "x_p": 30, "r_n": 50, "r_p": 50}
    
    durations = [7, 14, 21, 30, 90]
    results = []

    # Experiment Definition (3 Cycles)
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

    for days in durations:
        print(f"Running simulation for {days} days storage...")
        storage_time = days * 24 * 3600
        t_eval = np.linspace(0, storage_time, 100)
        
        sim_storage = pb.Simulation(model, parameter_values=param, var_pts=var_pts)
        sol_storage = sim_storage.solve(t_eval=t_eval, initial_soc=0.9)
        
        sim_cycling = pb.Simulation(model, parameter_values=param, var_pts=var_pts, experiment=experiment)
        sol_cycling = sim_cycling.solve(starting_solution=sol_storage)
        
        v_start = sol_storage["Terminal voltage [V]"].entries[0]
        v_end = sol_storage["Terminal voltage [V]"].entries[-1]
        
        results.append({
            "days": days,
            "lli_end": sol_cycling["Loss of lithium inventory [%]"].entries[-1],
            "v_drop_mv": (v_start - v_end) * 1000,
            "sei_end": sol_cycling["X-averaged negative SEI thickness [m]"].entries[-1] * 1e9
        })

    # Summary Table and Plot
    print("\n--- Summary Results ---")
    print(f"{'Days':<10} {'LLI [%]':<15} {'V-Drop [mV]':<15} {'SEI [nm]':<15}")
    for res in results:
        print(f"{res['days']:<10} {res['lli_end']:<15.3f} {res['v_drop_mv']:<15.2f} {res['sei_end']:<15.2f}")

    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 10))
    days_list = [r["days"] for r in results]
    lli_list = [r["lli_end"] for r in results]
    v_drop_list = [r["v_drop_mv"] for r in results]
    sei_list = [r["sei_end"] for r in results]

    # Subplot 1: LLI and SEI
    ax1.plot(days_list, lli_list, 'o-', color='#1d3557', label='Final LLI [%]')
    ax1.set_xlabel('Storage Duration [days]')
    ax1.set_ylabel('Loss of Lithium Inventory [%]', color='#1d3557')
    ax1.tick_params(axis='y', labelcolor='#1d3557')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(days_list, sei_list, 's--', color='#e63946', label='Final SEI [nm]')
    ax2.set_ylabel('Final SEI Thickness [nm]', color='#e63946')
    ax2.tick_params(axis='y', labelcolor='#e63946')
    ax1.set_title('Aging Comparison (Madi-Reddy Paper Values)')

    # Subplot 2: Voltage Drop
    ax3.plot(days_list, v_drop_list, 'd-', color='#457b9d', label='Voltage Drop [mV]')
    ax3.set_xlabel('Storage Duration [days]')
    ax3.set_ylabel('Voltage Drop [mV]', color='#457b9d')
    ax3.tick_params(axis='y', labelcolor='#457b9d')
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Storage Potential Decay (OCV Phase)')

    fig.tight_layout()
    plt.savefig("storage_comparison_results.png")
    print("\nComparison results saved to storage_comparison_results.png")

if __name__ == "__main__":
    run_storage_sweep()
