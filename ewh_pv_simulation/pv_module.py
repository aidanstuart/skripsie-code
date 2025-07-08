"""
pv_module.py

PV power modeling using pvlib ModelChain.
"""
import pandas as pd
import pvlib
from pvlib.location import Location
from pvlib.pvsystem import PVSystem
from pvlib.modelchain import ModelChain
import warnings

class PVModule:
    """
    Wraps pvlib's ModelChain for rooftop PV power simulation.

    Parameters
    ----------
    module_params : dict
        PV module datasheet parameters (from pvlib.pvsystem.retrieve_sam).
    system_params : dict
        System configuration, requires keys:
        - tilt: surface tilt in degrees
        - azimuth: surface azimuth in degrees
        - inverter: dict of inverter parameters
        - latitude: site latitude
        - longitude: site longitude
        - timezone: site timezone string
    """
    def __init__(self, module_params: dict, system_params: dict):
        self.module = module_params
        
        # Unpack system parameters
        tilt = system_params['tilt']
        azimuth = system_params['azimuth']
        inv_params = system_params.get('inverter', {})
        lat = system_params['latitude']
        lon = system_params['longitude']
        tz = system_params['timezone']

        # Build PVSystem and ModelChain
        try:
            pv_sys = PVSystem(
                module_parameters=module_params,
                inverter_parameters=inv_params,
                surface_tilt=tilt,
                surface_azimuth=azimuth
            )
            
            location = Location(latitude=lat, longitude=lon, tz=tz)
            
            # Suppress warnings about missing parameters
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.mc = ModelChain(
                    pv_sys,
                    location,
                    aoi_model='sapm',
                    spectral_model='sapm',
                    temperature_model='sapm'
                )
        except Exception as e:
            print(f"Error initializing PV system: {e}")
            raise

    def get_power(self, meteo: pd.DataFrame) -> pd.Series:
        """
        Compute DC power (p_mp) from meteorological inputs.

        Parameters
        ----------
        meteo : DataFrame
            Must include columns ['dni','ghi','dhi','temp_air','wind_speed'] indexed by timestamp.

        Returns
        -------
        pd.Series
            DC power output (kW) at each timestamp.
        """
        try:
            # Ensure we have the required columns
            required_cols = ['dni', 'ghi', 'dhi', 'temp_air', 'wind_speed']
            missing_cols = [col for col in required_cols if col not in meteo.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Create weather dataframe with proper column names
            weather = meteo[required_cols].copy()
            
            # Run the model
            self.mc.run_model(weather)
            
            # Return DC power in kW
            return self.mc.results.dc['p_mp'] / 1000.0
            
        except Exception as e:
            print(f"Error computing PV power: {e}")
            # Return zero power series as fallback
            return pd.Series(0.0, index=meteo.index)