import pybamm as pb
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def run_plotly_viz():
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

    # 3. Experiment Sequence
    cycle_steps = (
        "Discharge at C/9 until 3.2 V",
        "Rest for 15 minutes",
        "Charge at C/7 until 4.1 V",
        "Hold at 4.1 V until C/37",
        "Rest for 15 minutes",
        pb.step.string("Discharge at C/4 for 5s", period="0.05 seconds"),
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

    # 5. Interactive Plotting (Post 10-year only)
    time_h = sol["Time [h]"].entries
    mask = time_h >= 87600
    time_zoomed = time_h[mask] - 87600
    
    voltage = sol["Terminal voltage [V]"].entries[mask]
    lli = sol["Loss of lithium inventory [%]"].entries[mask]
    
    # Create Subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Terminal Voltage [V]", "Loss of Lithium Inventory [%]"))

    fig.add_trace(go.Scatter(x=time_zoomed, y=voltage, name="Voltage", 
                             line=dict(color='#1d3557', width=2),
                             hovertemplate='Time: %{x:.1f} h<br>Voltage: %{y:.4f} V'), row=1, col=1)

    fig.add_trace(go.Scatter(x=time_zoomed, y=lli, name="LLI Loss", 
                             line=dict(color='#e63946', width=2),
                             hovertemplate='Time: %{x:.1f} h<br>LLI: %{y:.2f} %'), row=2, col=1)

    fig.update_xaxes(title_text="Time since Start of Testing [h]", row=2, col=1)
    fig.update_layout(height=800, title_text="Interactive Testing Phase Analysis (30% SoC Enforced)", 
                      showlegend=False, template="plotly_white")
    
    output_file = "testing_phase_interactive.html"
    fig.write_html(output_file)
    print(f"\nInteractive plot complete. Results saved to {output_file}")

if __name__ == "__main__":
    run_plotly_viz()
