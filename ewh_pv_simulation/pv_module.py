import pandas as pd
import pvlib
from pvlib.location import Location
from pvlib.pvsystem import PVSystem
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
import warnings
import numpy as np

class PVModule:
    """
    Wraps pvlib's ModelChain for rooftop PV power simulation with enhanced debugging.
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
        racking_model = system_params.get('racking_model', 'open_rack_cell_glassback')

        try:
            print(f"Initializing PVSystem with tilt={tilt}°, azimuth={azimuth}°, location=({lat}, {lon})")
            if racking_model not in TEMPERATURE_MODEL_PARAMETERS['sapm']:
                print(f"Warning: Invalid racking model '{racking_model}', using default.")
                racking_model = 'open_rack_cell_glassback'

            temp_params = dict(TEMPERATURE_MODEL_PARAMETERS['sapm'][racking_model])

            pv_sys = PVSystem(
                module_parameters=self.module,
                inverter_parameters=inv_params,
                surface_tilt=tilt,
                surface_azimuth=azimuth,
                racking_model=racking_model,
                temperature_model_parameters=temp_params
            )

            location = Location(latitude=lat, longitude=lon, tz=tz)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.mc = ModelChain(
                    pv_sys,
                    location,
                    aoi_model='no_loss',
                    spectral_model='no_loss',
                    temperature_model='sapm'
                )

            print("✅ ModelChain initialized successfully")

        except Exception as e:
            print(f"❌ Error initializing PV system: {e}")
            import traceback
            traceback.print_exc()
            raise

    def get_power(self, meteo: pd.DataFrame) -> pd.Series:
        """
        Compute DC power (p_mp) from meteorological inputs.
        """
        try:
            required_cols = ['dni', 'ghi', 'dhi', 'temp_air', 'wind_speed']
            missing_cols = [col for col in required_cols if col not in meteo.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            weather = meteo[required_cols].copy()

            for col in required_cols:
                if weather[col].dtype == 'object':
                    weather[col] = pd.to_numeric(weather[col], errors='coerce')
                weather[col] = weather[col].replace([np.inf, -np.inf], 0).fillna(0)

            if weather.index.tz is not None:
                weather.index = weather.index.tz_localize(None)

            self.mc.run_model(weather)

            if not hasattr(self.mc, 'results') or self.mc.results is None:
                return pd.Series(0.0, index=meteo.index)

            if not hasattr(self.mc.results, 'dc') or self.mc.results.dc is None:
                return pd.Series(0.0, index=meteo.index)

            if 'p_mp' not in self.mc.results.dc:
                return pd.Series(0.0, index=meteo.index)

            return self.mc.results.dc['p_mp'] / 1000.0  # kW

        except Exception as e:
            print(f"❌ Error computing PV power: {e}")
            import traceback
            traceback.print_exc()
            return pd.Series(0.0, index=meteo.index)
