"""
demand.py

Load and process hot-water draw profiles.
"""
import pandas as pd

class DemandProfile:
    """
    Convert volume profiles to energy draws.
    """
    def __init__(self, profile_path: str, tank_setpoint: float, temp_in: float):
        self.df = pd.read_csv(profile_path, index_col=0, parse_dates=True)
        season_cols = [
            'Summer_Water_Consumption',
            'Autumn_Water_Consumption',
            'Winter_Water_Consumption',
            'Spring_Water_Consumption'
        ]
        self.df['volume_l'] = self.df[season_cols].sum(axis=1)
        self.tank_setpoint = tank_setpoint
        self.temp_in = temp_in

    def get_draw_energy(self) -> pd.Series:
        """
        Return energy drawn (kWh) per timestep.
        """
        # volume in L per interval -> mass kg
        mass = self.df['volume_l'] / 1000 * 1000
        # sum seasonal volumes into one series (L)
        volume = self.df[['Summer_Water_Consumption',
                          'Autumn_Water_Consumption',
                          'Winter_Water_Consumption',
                          'Spring_Water_Consumption']].sum(axis=1)
        mass = volume / 1000 * 1000   # convert L → m³ → kg via density
        # energy = m*c*DeltaT
        c = 4184
        dT = self.tank_setpoint - self.temp_in
        E_j = mass * c * dT
        return E_j / (3600 * 1000)
