#
# Madi-Reddy SEI growth model
#
import pybamm
from .base_sei import BaseModel


class MadiReddySEI(BaseModel):
    """
    Class for Madi-Reddy mixed-mode SEI growth.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain to use for this submodel ("negative" or "positive")
    options : dict
        A dictionary of options to be passed to the model.
    phase : str, optional
        Phase of the particle (default is "primary")
    cracks : bool, optional
        Whether this is a submodel for standard SEI or SEI on cracks
    """

    def __init__(self, param, domain, options, phase="primary", cracks=False):
        super().__init__(param, domain, options=options, phase=phase, cracks=cracks)
        if self.options.electrode_types[domain] == "planar":
            self.reaction_loc = "interface"
        elif self.options["x-average side reactions"] == "true":
            self.reaction_loc = "x-average"
        else:
            self.reaction_loc = "full electrode"
        self.N = 100  # Increased number of solvent diffusion nodes for Colab-level precision
        
        # Register citation
        pybamm.citations.register(
            "S.K. Madi Reddy, W. Shang, R.E. White, Mathematical Model for SEI Growth under Open-Circuit Conditions, J. Electrochem. Soc. 169 (2022) 090505. https://doi.org/10.1149/1945-7111/ac8ee5."
        )

    def get_fundamental_variables(self):
        domain, Domain = self.domain_Domain
        phase_param = self.phase_param

        # Standard SEI concentration (drives thickness)
        L_sei_0 = phase_param.L_sei_0
        V_bar_sei = phase_param.V_bar_sei
        
        # c_sei is an interfacial quantity [mol.m-2] for interface reactions
        # but in porous electrodes it is a bulk quantity [mol.m-3]
        if self.reaction_loc == "interface":
            scale = L_sei_0 / V_bar_sei
            c_sei = pybamm.Variable(
                f"{Domain} {self.reaction_name}concentration [mol.m-2]",
                domain="current collector",
                scale=scale,
            )
        else:
            scale = L_sei_0 * phase_param.a_typ / V_bar_sei
            c_sei = pybamm.Variable(
                f"{Domain} {self.reaction_name}concentration [mol.m-3]",
                domain=f"{domain} electrode",
                auxiliary_domains={"secondary": "current collector"},
                scale=scale,
            )

        # Solvent concentration nodes (Landau domain xi = x/L)
        # We add N solvent nodes as fundamental variables
        c_s_bulk = phase_param.c_ec_0
        variables = {}
        # Domain for variables
        if self.reaction_loc == "x-average":
            node_domain = []
        else:
            node_domain = [f"{domain} electrode"]
            
        for i in range(1, self.N):
            node_name = f"{Domain} {self.reaction_name}solvent concentration node {i} [mol.m-3]"
            c_node = pybamm.Variable(
                node_name, 
                domain=node_domain, 
                auxiliary_domains={"secondary": "current collector"} if self.reaction_loc != "interface" else None,
                scale=c_s_bulk
            )
            variables[node_name] = c_node

        variables.update(self._get_standard_concentration_variables(c_sei))
        return variables

    def get_coupled_variables(self, variables):
        domain, Domain = self.domain_Domain
        phase_param = self.phase_param
        
        # OCP, Potential, and Temperature
        if self.reaction_loc == "interface":
            delta_phi = variables["Lithium metal interface surface potential difference [V]"]
            T = variables[f"{Domain} electrode temperature [K]"]
            T = pybamm.boundary_value(T, "right")
        elif self.reaction_loc == "x-average":
            delta_phi = variables[f"X-averaged {domain} electrode surface potential difference [V]"]
            T = variables[f"X-averaged {domain} electrode temperature [K]"]
        else:
            delta_phi = variables[f"{Domain} electrode surface potential difference [V]"]
            T = variables[f"{Domain} electrode temperature [K]"]

        L_sei = variables[f"{Domain} {self.reaction_name}thickness [m]"]
        
        # Paper parameters
        k_sei = phase_param.k_sei
        D_s_eff = phase_param.D_ec
        c_s_bulk = phase_param.c_ec_0
        epsilon_sei = pybamm.Parameter(f"{Domain} electrode SEI porosity") # Custom parameter
        U_sei = phase_param.U_sei
        alpha_sei = phase_param.alpha_SEI
        F = self.param.F
        R = self.param.R
        F_RT = F / (R * T)

        # Overpotential
        eta_sei = delta_phi - U_sei

        # Interfacial solvent concentration (xi = 1, node N-1)
        c_s_inter = variables[f"{Domain} {self.reaction_name}solvent concentration node {self.N-1} [mol.m-3]"]
        
        # Tafel law (mixed-mode)
        # j_sei = -F * k_sei * (c_s_inter)        # Surface concentrations and factors
        c_n_surf = variables[f"{Domain} {self.phase_name}particle surface concentration [mol.m-3]"]
        c_n_max = phase_param.c_max
        z_n = pybamm.maximum(c_n_surf / c_n_max, 1e-6)
        c_s_inter_safe = pybamm.maximum(c_s_inter, 1e-12)
        
        # Butler-Volmer / Tafel factors
        n_sei = phase_param.z_sei
        j0_sei = n_sei * F * k_sei * (c_n_max ** (n_sei * (1.0 - alpha_sei)))
        
        j_sei = (
            -j0_sei
            * (c_s_inter_safe / c_s_bulk)
            * (z_n**2)
            * pybamm.exp(-alpha_sei * n_sei * F_RT * eta_sei)
        )

        # Arrhenius enhancement (PyBaMM standard)
        arrhenius = pybamm.exp(phase_param.E_sei / R * (1/self.param.T_ref - 1/T))
        j_sei = arrhenius * j_sei

        # Register standard variables
        variables.update(self._get_standard_reaction_variables(j_sei))
        variables.update(self._get_standard_volumetric_current_density_variables(variables))
        
        # Solvent fluxes (Landau transformed)
        # dL/dt = - j_sei * V_bar / (n_sei * F)
        dLdt = -j_sei * phase_param.V_bar_sei / (n_sei * F)
        
        variables.update(super().get_coupled_variables(variables))
        return variables

    def set_rhs(self, variables):
        domain, Domain = self.domain_Domain
        phase_param = self.phase_param
        
        # Standard SEI concentration RHS
        c_sei = variables[f"{Domain} {self.reaction_name}concentration [mol.m-3]" if self.reaction_loc != "interface" else f"{Domain} {self.reaction_name}concentration [mol.m-2]"]
        j_sei = variables[f"{Domain} electrode {self.reaction_name}interfacial current density [A.m-2]"]
        
        if self.reaction_loc == "interface":
            a = 1
        else:
            a = variables[f"{Domain} electrode {self.phase_name}surface area to volume ratio [m-1]"]
            
        n_sei = phase_param.z_sei
        dcdt_sei = a * j_sei / (self.param.F * n_sei)
        self.rhs = {c_sei: -dcdt_sei}

        # Solvent diffusion RHS
        L_sei = variables[f"{Domain} {self.reaction_name}thickness [m]"]
        dLdt = -j_sei * phase_param.V_bar_sei / (n_sei * self.param.F)
        D_s_eff = phase_param.D_ec
        epsilon_sei = pybamm.Parameter(f"{Domain} electrode SEI porosity")
        
        dxi = 1.0 / (self.N - 1)
        
        n_sei = phase_param.z_sei
        for i in range(1, self.N):
            c_node = variables[f"{Domain} {self.reaction_name}solvent concentration node {i} [mol.m-3]"]
            xi = i * dxi
            
            # Left boundary (bulk solvent Dirichlet)
            if i == 1:
                c_prev = phase_param.c_ec_0 * epsilon_sei
            else:
                c_prev = variables[f"{Domain} {self.reaction_name}solvent concentration node {i-1} [mol.m-3]"]
                
            if i == self.N - 1:
                # xi = 1 (interface)
                # dc/dxi = (L * j_sei) / (n * F * D_eff)
                grad_xi = (L_sei * j_sei) / (n_sei * self.param.F * D_s_eff)
                
                # c_xixi = 2 * (c_prev - c_node + dxi * grad_xi) / dxi^2
                c_xixi = 2 * (c_prev - c_node + dxi * grad_xi) / (dxi**2)
                
                conv = (xi / L_sei) * dLdt * grad_xi
                diff = (D_s_eff / (L_sei**2)) * c_xixi
                self.rhs[c_node] = conv + diff
            else:
                c_next = variables[f"{Domain} {self.reaction_name}solvent concentration node {i+1} [mol.m-3]"]
                
                c_xi = (c_next - c_prev) / (2 * dxi)
                c_xixi = (c_next - 2 * c_node + c_prev) / (dxi**2)
                
                conv = (xi / L_sei) * dLdt * c_xi
                diff = (D_s_eff / (L_sei**2)) * c_xixi
                self.rhs[c_node] = conv + diff

    def set_initial_conditions(self, variables):
        domain, Domain = self.domain_Domain
        phase_param = self.phase_param
        L_sei_0 = phase_param.L_sei_0
        V_bar_sei = phase_param.V_bar_sei
        
        # Standard SEI IC
        c_sei = variables[f"{Domain} {self.reaction_name}concentration [mol.m-3]" if self.reaction_loc != "interface" else f"{Domain} {self.reaction_name}concentration [mol.m-2]"]
        if self.reaction_loc == "interface":
            c_sei_0 = L_sei_0 / V_bar_sei
        else:
            c_sei_0 = L_sei_0 * phase_param.a_typ / V_bar_sei
        
        self.initial_conditions = {c_sei: c_sei_0}
        
        # Solvent nodes IC (constant bulk for simplicity)
        epsilon_sei = pybamm.Parameter(f"{Domain} electrode SEI porosity")
        for i in range(1, self.N):
            c_node = variables[f"{Domain} {self.reaction_name}solvent concentration node {i} [mol.m-3]"]
            self.initial_conditions[c_node] = phase_param.c_ec_0 * epsilon_sei
