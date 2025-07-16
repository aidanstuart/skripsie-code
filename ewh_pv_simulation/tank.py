"""
tank.py

Two-node stratified electric water heater tank.
"""
import numpy as np

class StratifiedTank:
    """
    Two-node stratified tank.

    Attributes
    ----------
    top_temp : float
    bottom_temp : float
    volume : float
    c : float
        Specific heat capacity of water (J/kg.K).
    rho : float
        Water density (kg/m3).
    R_th : float
        Thermal resistance to ambient (K/W).
    element_rating : float
        Heating element power (kW).
    dt : float
        Timestep (s).
    """
    def __init__(self, volume_l: float, c: float, rho: float,
                 R_th: float, element_rating_kw: float, dt_s: float, **kwargs):
        self.volume = float(volume_l) / 1000  # L -> m3
        self.c = float(c)
        self.rho = float(rho)
        self.R_th = float(R_th)
        self.element_rating = float(element_rating_kw)
        self.dt = float(dt_s)
        
        # Split volume equally between top and bottom nodes
        self.v_top = self.volume / 2
        self.v_bot = self.volume / 2
        
        # Mass of water in each node
        self.mass_top = self.rho * self.v_top
        self.mass_bot = self.rho * self.v_bot
        
        # Initialize temperatures
        self.top_temp = None
        self.bottom_temp = None

    def initialize(self, T0: float):
        """Initialize both nodes to T0 (°C)."""
        self.top_temp = float(T0)
        self.bottom_temp = float(T0)

    def step(self, power_kw: float, draw_kwh: float, T_amb: float):
        """
        Advance tank state one timestep.

        Parameters
        ----------
        power_kw : float
            Heating element input (kW).
        draw_kwh : float
            Energy removed by draw event (kWh).
        T_amb : float
            Ambient temperature (°C).

        Returns
        -------
        tuple
            (top_temp, bottom_temp) in °C
        """
        # Ensure all inputs are float
        power_kw = float(power_kw)
        draw_kwh = float(draw_kwh)
        T_amb = float(T_amb)
        
        # Convert power to energy over timestep
        Q_in = power_kw * 1000 * self.dt  # J
        
        # Calculate ambient losses
        T_mean = (self.top_temp + self.bottom_temp) / 2
        Q_loss = (T_mean - T_amb) * self.dt / self.R_th  # J
        
        # Assume heating element is in bottom node and heat distributes evenly
        # (this is a simplification - real elements might be positioned differently)
        Q_heat_top = Q_in * 0.3  # 30% to top
        Q_heat_bot = Q_in * 0.7  # 70% to bottom
        
        # Calculate energy content of each node
        E_top = self.mass_top * self.c * self.top_temp + Q_heat_top - Q_loss/2
        E_bot = self.mass_bot * self.c * self.bottom_temp + Q_heat_bot - Q_loss/2
        
        # Handle draw events (removes energy from top node first)
        E_draw = draw_kwh * 3600 * 1000  # Convert kWh to J
        
        if E_draw > 0:
            if E_draw <= E_top:
                # Draw only from top node
                E_top -= E_draw
                # Add cold water to replace what was drawn
                # (simplified - assumes instantaneous mixing)
                if self.top_temp > 0:
                    E_top += (E_draw * T_amb / self.top_temp)
                else:
                    E_top += E_draw * T_amb / 20  # Avoid division by zero
            else:
                # Draw from both nodes
                E_draw_remaining = E_draw - E_top
                E_top = self.mass_top * self.c * T_amb  # Replace with cold water
                E_bot = max(0, E_bot - E_draw_remaining)
                # Add cold water to bottom node too
                if self.bottom_temp > 0:
                    E_bot += (E_draw_remaining * T_amb / self.bottom_temp)
                else:
                    E_bot += E_draw_remaining * T_amb / 20  # Avoid division by zero
        
        # Calculate new temperatures
        if self.mass_top * self.c > 0:
            self.top_temp = max(0, E_top / (self.mass_top * self.c))
        else:
            self.top_temp = T_amb
            
        if self.mass_bot * self.c > 0:
            self.bottom_temp = max(0, E_bot / (self.mass_bot * self.c))
        else:
            self.bottom_temp = T_amb
        
        # Prevent temperatures from going too high (safety limit)
        self.top_temp = min(float(self.top_temp), 90.0)
        self.bottom_temp = min(float(self.bottom_temp), 90.0)
        
        # Ensure temperatures are not NaN
        if np.isnan(self.top_temp):
            self.top_temp = T_amb
        if np.isnan(self.bottom_temp):
            self.bottom_temp = T_amb
        
        return float(self.top_temp), float(self.bottom_temp)