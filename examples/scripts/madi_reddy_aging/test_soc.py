import pybamm as pb
import numpy as np

for soc in [0.9, 0.3]:
    options = {"SEI": "madi-reddy"}
    model = pb.lithium_ion.DFN(options)
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
        "Current function [A]": 0
    }, check_already_exists=False)

    sim = pb.Simulation(model, parameter_values=param)
    sim.solve(t_eval=[0, 3600*24*1], initial_soc=soc) # 1 day
    sol = sim.solution
    L_sei = sol["X-averaged negative SEI thickness [m]"].entries[-1]
    print(f"SOC {soc}: L_sei = {L_sei}")
