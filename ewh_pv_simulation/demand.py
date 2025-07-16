"""
demand.py

Load and process hot-water draw profiles.
"""
import pandas as pd
import numpy as np

class DemandProfile:
    """
    Convert volume profiles to energy draws.
    """
    def __init__(self, profile_path: str, tank_setpoint: float, temp_in: float):
        try:
            print(f"    Loading demand profile: {profile_path}")
            # Read the CSV file
            self.df = pd.read_csv(profile_path)
            
            # Check if the data has the expected columns
            season_cols = [
                'Summer_Water_Consumption',
                'Autumn_Water_Consumption', 
                'Winter_Water_Consumption',
                'Spring_Water_Consumption'
            ]
            
            # Verify columns exist
            missing_cols = [col for col in season_cols if col not in self.df.columns]
            if missing_cols:
                print(f"    Available columns: {list(self.df.columns)}")
                raise ValueError(f"Missing columns in {profile_path}: {missing_cols}")
            
            # Create a proper datetime index
            # Match the irradiance data timeframe - 2024 data with 5-minute intervals
            start_date = '2024-01-01'
            periods = len(self.df)
            self.df.index = pd.date_range(start=start_date, periods=periods, freq='5min')
            
            # Sum seasonal volumes into one series
            self.df['volume_l'] = self.df[season_cols].sum(axis=1)
            
            # Debug info
            print(f"    Demand profile loaded: {periods} points, {self.df['volume_l'].sum():.1f}L total")
            print(f"    Time range: {self.df.index.min()} to {self.df.index.max()}")
            
            self.tank_setpoint = tank_setpoint
            self.temp_in = temp_in
            
        except Exception as e:
            print(f"    ERROR loading demand profile from {profile_path}: {e}")
            raise

    def get_draw_energy(self) -> pd.Series:
        """
        Return energy drawn (kWh) per timestep.
        """
        # Convert volume in L to mass in kg (density = 1000 kg/m³)
        volume_l = self.df['volume_l']
        mass_kg = volume_l  # 1L of water = 1kg
        
        # Calculate energy required to heat water
        # E = m * c * ΔT
        c = 4184  # J/(kg·K) - specific heat capacity of water
        delta_T = self.tank_setpoint - self.temp_in
        energy_J = mass_kg * c * delta_T
        
        # Convert from J to kWh
        energy_kWh = energy_J / (3600 * 1000)
        
        return energy_kWh