import os
import pandas as pd
import numpy as np
import pvlib
from tank import StratifiedTank
from config import TANK_PARAMS, SIM_PARAMS, SYSTEM_PARAMS, MODULE_NAME
from demand import DemandProfile
from pv_module import PVModule
from utils import get_hourly_rate, CHUNK_SIZE


def simulate_household_efficient(profile_path: str, irr_df: pd.DataFrame, tariff: pd.DataFrame) -> dict:
    """Memory-efficient simulation of a single household with solar tracking only when heating is needed."""
    try:
        household_id = os.path.basename(profile_path)
        print(f"Processing {household_id}...")

        inlet_temp = float(SIM_PARAMS["cold_event_temperature"])
        dp = DemandProfile(profile_path, float(TANK_PARAMS['setpoint']), inlet_temp)
        energy_demand = dp.get_draw_energy().iloc[::4]
        draw = energy_demand.reindex(irr_df.index, method='nearest').fillna(0)

        try:
            cec_mod = pvlib.pvsystem.retrieve_sam('CECMod')
            if MODULE_NAME in cec_mod:
                module_params = cec_mod[MODULE_NAME]
            else:
                print(f"⚠️ MODULE_NAME '{MODULE_NAME}' not found. Using fallback module.")
                module_params = cec_mod.iloc[:, 0]  # First column as fallback
            pv_sys = PVModule(module_params, SYSTEM_PARAMS)
            pv_power = pv_sys.get_power(irr_df)
            pv_power = pd.to_numeric(pv_power, errors='coerce').fillna(0)
        except Exception as e:
            print(f"Error initializing PV system: {e}")
            pv_power = pd.Series(0.0, index=irr_df.index)

        tank = StratifiedTank(**TANK_PARAMS)
        tank.initialize(float(TANK_PARAMS['setpoint']))
        dt_h = (float(TANK_PARAMS['dt_s']) * 4) / 3600  # 20-min step

        chunk_results = []
        timestamps = irr_df.index.tolist()
        solar_used_when_needed_kwh = 0.0
        grid_used_when_needed_kwh = 0.0
        heating_event_count = 0
        solar_event_count = 0

        for i in range(0, len(timestamps), CHUNK_SIZE):
            chunk_end = min(i + CHUNK_SIZE, len(timestamps))
            chunk_ts = timestamps[i:chunk_end]
            chunk_records = []

            for ts in chunk_ts:
                try:
                    p_pv = float(pv_power.get(ts, 0.0))
                    p_pv_used = min(p_pv, float(tank.element_rating))
                    p_grid = max(0, float(tank.element_rating) - p_pv_used)
                    t_amb = float(irr_df.loc[ts, 'temp_air'])
                    demand = float(draw.get(ts, 0.0))
                    top_temp, bot_temp = tank.step(p_grid + p_pv_used, demand, t_amb)

                    if p_pv_used + p_grid > 0:
                        heating_event_count += 1
                        solar_used_when_needed_kwh += p_pv_used * dt_h
                        grid_used_when_needed_kwh += p_grid * dt_h
                        if p_pv_used > 0:
                            solar_event_count += 1

                    chunk_records.append({
                        'grid_kwh': p_grid * dt_h,
                        'pv_kwh': p_pv_used * dt_h,
                        'demand_kwh': demand,
                        'top_T': float(top_temp),
                        'hour': int(ts.hour),
                        'month': int(ts.month),
                        'weekday': int(ts.weekday())
                    })
                except:
                    continue

            if chunk_records:
                chunk_df = pd.DataFrame(chunk_records)
                chunk_df['rate_R'] = chunk_df.apply(
                    lambda row: get_hourly_rate(
                        pd.Timestamp(year=2024, month=int(row['month']), day=1, hour=int(row['hour'])),
                        tariff
                    ), axis=1)
                chunk_df['cost_R'] = chunk_df['grid_kwh'] * chunk_df['rate_R']

                chunk_summary = {
                    'grid_kwh': float(chunk_df['grid_kwh'].sum()),
                    'pv_kwh': float(chunk_df['pv_kwh'].sum()),
                    'demand_kwh': float(chunk_df['demand_kwh'].sum()),
                    'cost_R': float(chunk_df['cost_R'].sum()),
                    'cold_draws': int((chunk_df['top_T'] < float(TANK_PARAMS['min_usage_temperature'])).sum()),
                    'total_points': len(chunk_df),
                    'temp_sum': float(chunk_df['top_T'].sum()),
                    'solar_used_when_needed_kwh': solar_used_when_needed_kwh,
                    'grid_used_when_needed_kwh': grid_used_when_needed_kwh,
                    'heating_event_count': heating_event_count,
                    'solar_event_count': solar_event_count
                }

                chunk_results.append(chunk_summary)
                gc.collect()

        if chunk_results:
            total_grid = sum(c['grid_kwh'] for c in chunk_results)
            total_pv = sum(c['pv_kwh'] for c in chunk_results)
            total_demand = sum(c['demand_kwh'] for c in chunk_results)
            total_cost = sum(c['cost_R'] for c in chunk_results)
            total_cold = sum(c['cold_draws'] for c in chunk_results)
            total_points = sum(c['total_points'] for c in chunk_results)
            temp_sum = sum(c['temp_sum'] for c in chunk_results)
            total_solar_needed = sum(c['solar_used_when_needed_kwh'] for c in chunk_results)
            total_grid_needed = sum(c['grid_used_when_needed_kwh'] for c in chunk_results)
            total_heating_events = sum(c['heating_event_count'] for c in chunk_results)
            total_solar_events = sum(c['solar_event_count'] for c in chunk_results)

            kpis = {
                'annual_grid_kwh': total_grid,
                'annual_solar_kwh': total_pv,
                'annual_demand_kwh': total_demand,
                'solar_fraction': total_pv / (total_grid + total_pv) if (total_grid + total_pv) > 0 else 0,
                'cold_draw_pct': (total_cold / total_points * 100) if total_points > 0 else 0,
                'avg_temp': temp_sum / total_points if total_points > 0 else 0,
                'solar_used_when_needed_kwh': total_solar_needed,
                'grid_used_when_needed_kwh': total_grid_needed,
                'solar_heating_energy_fraction': total_solar_needed / (total_solar_needed + total_grid_needed) if (total_solar_needed + total_grid_needed) > 0 else 0,
                'solar_heating_event_fraction': total_solar_events / total_heating_events if total_heating_events > 0 else 0
            }

            return {
                'profile': household_id,
                'kpis': kpis,
                'cost_R': total_cost,
                'success': True
            }

        else:
            return {'profile': household_id, 'kpis': {}, 'cost_R': 0, 'success': False, 'error': 'No data'}

    except Exception as e:
        return {'profile': os.path.basename(profile_path), 'kpis': {}, 'cost_R': 0, 'success': False, 'error': str(e)}
