import pybamm as pb
import numpy as np
from scipy.optimize import brentq

def get_voltage_at_1yr(scale_factor):
    # Setup model
    options = {"SEI": "madi-reddy"}
    model = pb.lithium_ion.DFN(options)
    
    # Base Parameters (Original Madi-Reddy)
    param = pb.ParameterValues("OKane2022")
    eps_s = param["Negative electrode active material volume fraction"]
    
    # Scale both k and D by the same factor
    k_base = 1.6434e-17
    D_base = 1.4029e-19
    
    param.update({
        "Negative particle radius [m]": 3.0 * eps_s / 3.0e6,
        "Negative electrode maximum concentration [mol.m-3]": 3.056e4,
        "Ratio of lithium moles to SEI moles": 2.0,
        "SEI kinetic rate constant [m.s-1]": k_base * scale_factor,
        "EC diffusivity [m2.s-1]": D_base * scale_factor,
        "EC initial concentration in electrolyte [mol.m-3]": 4541.0,
        "Negative electrode SEI porosity": 0.05,
        "Initial SEI thickness [m]": 2e-9,  
        "SEI open-circuit potential [V]": 0.4,
        "SEI partial molar volume [m3.mol-1]": 9.585e-05,
        "SEI growth transfer coefficient": 0.5,
        "Current function [A]": 0,
    }, check_already_exists=False)

    # Solve for 1 year
    t_eval = np.linspace(0, 365 * 24 * 3600, 50)
    sim = pb.Simulation(model, parameter_values=param)
    try:
        sol = sim.solve(t_eval, initial_soc=0.9)
        final_v = sol["Voltage [V]"].entries[-1]
        return final_v
    except:
        return 0.0 # return 0 if fails

def objective(scale_factor):
    voltage = get_voltage_at_1yr(scale_factor)
    print(f"Scale: {scale_factor:.2e} -> Final Voltage: {voltage:.4f}V")
    return voltage - 3.9

print("Finding optimal scale factor for 3.9V final voltage...")
# Range: we need significantly more LLI to hit 3.9V.
optimal_scale = brentq(objective, 0.1, 100.0, xtol=1e-2)

print(f"\nOptimal Scale Factor: {optimal_scale:.6e}")
print(f"New k_sei: {1.6434e-17 * optimal_scale:.6e}")
print(f"New D_eff: {1.4029e-19 * optimal_scale:.6e}")
