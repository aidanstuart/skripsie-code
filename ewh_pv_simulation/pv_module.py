"""
pv_module.py

PV power modeling using pvlib ModelChain.
"""
import pandas as pd
import pvlib
from pvlib.location import Location
from pvlib.pvsystem import PVSystem
from pvlib.modelchain import ModelChain

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
        # unpack system parameters
        tilt = system_params['tilt']
        azimuth = system_params['azimuth']
        inv_params = system_params.get('inverter', {})
        lat = system_params['latitude']
        lon = system_params['longitude']
        tz = system_params['timezone']

        # build PVSystem and ModelChain
        pv_sys = PVSystem(
            module_parameters=module_params,
            inverter_parameters=inv_params,
            surface_tilt=tilt,
            surface_azimuth=azimuth
        )
        location = Location(latitude=lat, longitude=lon, tz=tz)
        self.mc = ModelChain(
            pv_sys,
            location,
            aoi_model='sapm',
            spectral_model='sapm',
            temperature_model='sapm'
        )

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
        weather = meteo.rename(columns={
            'dni': 'dni',
            'ghi': 'ghi',
            'dhi': 'dhi',
            'temp_air': 'temp_air',
            'wind_speed': 'wind_speed'
        })
        self.mc.run_model(weather)
        # p_mp in W, convert to kW
        return self.mc.results.dc['p_mp'] / 1000.0
