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
                 R_th: float, element_rating_kw: float, dt_s: float):
        self.volume = volume_l / 1000  # L -> m3
        self.c = c
        self.rho = rho
        self.R_th = R_th
        self.element_rating = element_rating_kw
        self.dt = dt_s
        # split volume
        self.v_top = self.volume / 2
        self.v_bot = self.volume / 2
        # init temperatures
        self.top_temp = None
        self.bottom_temp = None

    def initialize(self, T0: float):
        """Initialize both nodes to T0 (°C)."""
        self.top_temp = T0
        self.bottom_temp = T0

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
        (top_temp, bottom_temp)
        """
        # convert to J
        Q_in = power_kw * 1000 * self.dt
        # ambient losses
        T_mean = (self.top_temp + self.bottom_temp) / 2
        Q_loss = (T_mean - T_amb) * self.dt / self.R_th
        # update energy
        E_top = self.c * self.rho * self.v_top * self.top_temp + Q_in/2 - Q_loss/2
        E_bot = self.c * self.rho * self.v_bot * self.bottom_temp + Q_in/2 - Q_loss/2
        # draws remove from top first
        E_draw = draw_kwh * 3600 * 1000
        if E_draw <= E_top:
            E_top -= E_draw
        else:
            E_draw -= E_top
            E_top = 0
            E_bot = max(0, E_bot - E_draw)
        # compute new temps
        self.top_temp = E_top / (self.c * self.rho * self.v_top)
        self.bottom_temp = E_bot / (self.c * self.rho * self.v_bot)
        return self.top_temp, self.bottom_temp
