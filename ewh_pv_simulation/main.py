"""
main.py

Run solar-medium strategy simulations in parallel
and compute costs under a TOU tariff.
"""
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import pvlib
from pv_module import PVModule
from tank import StratifiedTank
from config import TANK_PARAMS, SIM_PARAMS
from demand import DemandProfile
import glob

# === Simulation parameters ===
SYSTEM_PARAMS = {
    'tilt': 30,
    'azimuth': 180,
    'inverter': {'pdc0': 3000,       # inverter DC capacity in W
                 'eta_inv_nom': 0.96 # nominal efficiency},
                },
    'latitude': -33.934,          # e.g. Stellenbosch latitude
    'longitude': 18.860,          # Stellenbosch longitude
    'timezone': 'Africa/Johannesburg',
    'racking_model': 'open_rack_glass_glass',
}
TANK_PARAMS = {
    'volume_l': 150,  # litres
    'c': 4184,        # J/(kg·K)
    'rho': 1000,      # kg/m3
    'R_th': 0.4807,   # K/W
    'element_rating_kw': 3,  # kW
    'dt_s': 300,      # 5-min intervals
    'setpoint': 60,   # °C
    'min_usage_temperature': 50,  # °C
}
MODULE_NAME = 'Canadian_Solar_Inc__CS5P_220M'  # default CEC module

# === Helpers ===
def load_irradiance(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        parse_dates=['period_end'],
        index_col='period_end'
    )
    df = df.rename(columns={
        'gti':          'poa_global',
        'air_temp':     'temp_air',
        'wind_speed_10m':'wind_speed',
        'dni':          'dni',
        'ghi':          'ghi',
        'dhi':          'dhi',
    })

    # drop everything but the six columns we need:
    df = df[['poa_global','temp_air','wind_speed','dni','ghi','dhi']]

    # *new* — strip tz and sort for monotonicity
    df.index = df.index.tz_localize(None)
    df.sort_index(inplace=True)

    return df



def load_tariff(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def map_season(ts: pd.Timestamp) -> str:
    return 'high' if ts.month in [11,12,1,2,3,4] else 'low'

def map_day_type(ts: pd.Timestamp) -> str:
    return 'weekend' if ts.weekday() >= 5 else 'weekday'

def get_hourly_rate(ts: pd.Timestamp, tariff: pd.DataFrame) -> float:
    season = map_season(ts)
    day_type = map_day_type(ts)
    hour = ts.hour
    row = tariff[
        (tariff['season']==season) &
        (tariff['day_type']==day_type) &
        (tariff['start_hour']<=hour) &
        (tariff['end_hour']>hour)
    ]
    if row.empty:
        raise ValueError(f"No tariff for {ts}")
    return float(row['rate_R_per_kWh'].iloc[0])

# === Core simulation ===
def simulate_household(profile_path: str,
                       irr_df: pd.DataFrame,
                       tariff: pd.DataFrame) -> dict:
    inlet_temp = SIM_PARAMS["cold_event_temperature"]
    # demand
    dp = DemandProfile(profile_path, TANK_PARAMS['setpoint'], inlet_temp)
    # get demand energy, sort it, and forward-fill it onto irr_df’s index
    energy = dp.get_draw_energy().sort_index()
    draw   = energy.reindex(irr_df.index, method='ffill').fillna(0)
    # PV system
    cec_mod = pvlib.pvsystem.retrieve_sam('CECMod')
    module_params = cec_mod[MODULE_NAME]
    pv_sys = PVModule(module_params, SYSTEM_PARAMS)
    pv_power = pv_sys.get_power(irr_df)
    # tank
    tank = StratifiedTank(**TANK_PARAMS)
    tank.initialize(TANK_PARAMS['setpoint'])
    dt_h = TANK_PARAMS['dt_s'] / 3600
    records = []
    for ts in irr_df.index:
        p_pv = pv_power.loc[ts]
        p_pv_used = min(p_pv, tank.element_rating)
        p_grid = max(0, tank.element_rating - p_pv)
        t_amb = irr_df.loc[ts, 'temp_air']
        d     = draw.loc[ts]
        top, bot = tank.step(p_grid, d, t_amb)
        records.append((ts, top, bot, p_grid*dt_h, p_pv_used*dt_h))
    df = pd.DataFrame(records, columns=['ts','top_T','bot_T','grid_kwh','pv_kwh']).set_index('ts')
    # hourly aggregation
    hourly = df.resample('H').sum()
    meanT = df.resample('H')[['top_T','bot_T']].mean()
    hourly = hourly.join(meanT)
    hourly['solar_fraction'] = hourly['pv_kwh'] / (hourly['pv_kwh'] + hourly['grid_kwh']).replace(0, np.nan)
    # KPIs
    kpis = {
        'annual_grid_kwh': hourly['grid_kwh'].sum(),
        'annual_solar_kwh': hourly['pv_kwh'].sum(),
        'cold_draw_pct': (hourly['top_T'] < TANK_PARAMS['min_usage_temperature']).mean() * 100,
    }
    # cost
    hourly['rate_R'] = hourly.index.to_series().apply(lambda x: get_hourly_rate(x, tariff))
    hourly['cost_R'] = hourly['grid_kwh'] * hourly['rate_R']
    total_cost = hourly['cost_R'].sum()
    return {'profile': profile_path, 'hourly': hourly, 'kpis': kpis, 'cost_R': total_cost}

# === Entry point ===
if __name__ == '__main__':
    irr = load_irradiance('ewh_pv_simulation/solar_data/Stellenbosch/irradiance_data/solcast_2024_whole_year.csv')
    tariff = load_tariff('ewh_pv_simulation/tou_tariff.csv')
    profiles = sorted(glob.glob('ewh_pv_simulation/user_data/Source_Data/ewh_profile*.csv'))
    results = Parallel(n_jobs=-1)(
        delayed(simulate_household)(f, irr, tariff) for f in profiles
    )
    df_kpi = pd.DataFrame([
        {**r['kpis'], 'cost_R': r['cost_R'], 'profile': r['profile']} for r in results
    ])
    df_kpi.to_csv('results_kpis.csv', index=False)
    print("Done: results_kpis.csv generated.")
    
    import pvlib

