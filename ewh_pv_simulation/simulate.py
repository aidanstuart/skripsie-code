import os
import pandas as pd
import numpy as np
import pvlib
import gc
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

        # Load demand profile
        inlet_temp = float(SIM_PARAMS["cold_event_temperature"])
        dp = DemandProfile(profile_path, float(TANK_PARAMS['setpoint']), inlet_temp)
        energy_demand = dp.get_draw_energy()
        
        # Debug: Print data info
        print(f"  Energy demand shape: {energy_demand.shape}")
        print(f"  Irradiance data shape: {irr_df.shape}")
        print(f"  Energy demand index range: {energy_demand.index.min()} to {energy_demand.index.max()}")
        print(f"  Irradiance index range: {irr_df.index.min()} to {irr_df.index.max()}")
        
        # Align time series properly
        common_index = irr_df.index.intersection(energy_demand.index)
        if len(common_index) == 0:
            print(f"  ERROR: No common time indices between irradiance and demand data")
            return {'profile': household_id, 'kpis': {}, 'cost_R': 0, 'success': False, 'error': 'No common time indices'}
        
        print(f"  Common time indices: {len(common_index)}")
        
        # Use common index
        irr_aligned = irr_df.loc[common_index]
        draw = energy_demand.loc[common_index]

        # Initialize PV system
        try:
            cec_mod = pvlib.pvsystem.retrieve_sam('CECMod')
            if MODULE_NAME in cec_mod:
                module_params = cec_mod[MODULE_NAME]
                print(f"  Using module: {MODULE_NAME}")
            else:
                print(f"  WARNING: MODULE_NAME '{MODULE_NAME}' not found. Using fallback module.")
                module_params = cec_mod.iloc[0].to_dict()  # First module as fallback
            
            pv_sys = PVModule(module_params, SYSTEM_PARAMS)
            pv_power = pv_sys.get_power(irr_aligned)
            pv_power = pd.to_numeric(pv_power, errors='coerce').fillna(0)
            print(f"  PV power mean: {pv_power.mean():.3f} kW")
            
        except Exception as e:
            print(f"  ERROR initializing PV system: {e}")
            pv_power = pd.Series(0.0, index=irr_aligned.index)

        # Initialize tank
        tank = StratifiedTank(**TANK_PARAMS)
        tank.initialize(float(TANK_PARAMS['setpoint']))
        dt_h = float(TANK_PARAMS['dt_s']) / 3600  # Convert seconds to hours
        
        print(f"  Starting simulation with {len(common_index)} time steps...")

        # Simulation tracking variables
        solar_used_when_needed_kwh = 0.0
        grid_used_when_needed_kwh = 0.0
        heating_event_count = 0
        solar_event_count = 0
        total_solar_savings_R = 0.0
        
        chunk_results = []
        timestamps = common_index.tolist()

        for i in range(0, len(timestamps), CHUNK_SIZE):
            chunk_end = min(i + CHUNK_SIZE, len(timestamps))
            chunk_ts = timestamps[i:chunk_end]
            chunk_records = []

            for ts in chunk_ts:
                try:
                    p_pv = float(pv_power.loc[ts])
                    p_pv_used = min(p_pv, float(tank.element_rating))
                    p_grid = max(0, float(tank.element_rating) - p_pv_used)
                    t_amb = float(irr_aligned.loc[ts, 'temp_air'])
                    demand = float(draw.loc[ts])
                    
                    # Step tank simulation
                    top_temp, bot_temp = tank.step(p_grid + p_pv_used, demand, t_amb)

                    # Calculate electricity rate for this timestep
                    current_rate = get_hourly_rate(ts, tariff)
                    
                    # Calculate cost savings from solar
                    solar_savings_this_step = p_pv_used * dt_h * current_rate
                    
                    # Track heating events
                    if p_pv_used + p_grid > 0:
                        heating_event_count += 1
                        solar_used_when_needed_kwh += p_pv_used * dt_h
                        grid_used_when_needed_kwh += p_grid * dt_h
                        total_solar_savings_R += solar_savings_this_step
                        if p_pv_used > 0:
                            solar_event_count += 1

                    chunk_records.append({
                        'grid_kwh': p_grid * dt_h,
                        'pv_kwh': p_pv_used * dt_h,
                        'demand_kwh': demand,
                        'top_T': float(top_temp),
                        'hour': int(ts.hour),
                        'month': int(ts.month),
                        'weekday': int(ts.weekday()),
                        'rate_R': current_rate,
                        'solar_savings_R': solar_savings_this_step
                    })
                    
                except Exception as e:
                    print(f"    Error at timestamp {ts}: {e}")
                    continue

            if chunk_records:
                chunk_df = pd.DataFrame(chunk_records)
                chunk_df['cost_R'] = chunk_df['grid_kwh'] * chunk_df['rate_R']

                chunk_summary = {
                    'grid_kwh': float(chunk_df['grid_kwh'].sum()),
                    'pv_kwh': float(chunk_df['pv_kwh'].sum()),
                    'demand_kwh': float(chunk_df['demand_kwh'].sum()),
                    'cost_R': float(chunk_df['cost_R'].sum()),
                    'solar_savings_R': float(chunk_df['solar_savings_R'].sum()),
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

        # Aggregate results
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
            total_solar_savings = sum(c['solar_savings_R'] for c in chunk_results)
            
            # Calculate what the cost would have been without solar
            total_cost_without_solar = total_cost + total_solar_savings
            savings_percentage = (total_solar_savings / total_cost_without_solar * 100) if total_cost_without_solar > 0 else 0
            
            # Calculate heating time percentages
            total_heating_kwh = total_solar_needed + total_grid_needed
            solar_heating_time_percentage = (total_solar_needed / total_heating_kwh * 100) if total_heating_kwh > 0 else 0
            grid_heating_time_percentage = (total_grid_needed / total_heating_kwh * 100) if total_heating_kwh > 0 else 0

            print(f"  Results - Grid: {total_grid:.1f} kWh, Solar: {total_pv:.1f} kWh, Cost: R{total_cost:.2f}")
            print(f"  Cost savings - Solar saved: R{total_solar_savings:.2f} ({savings_percentage:.1f}%)")
            print(f"  Heating energy - Solar: {solar_heating_time_percentage:.1f}%, Grid: {grid_heating_time_percentage:.1f}%")

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
                'solar_heating_event_fraction': total_solar_events / total_heating_events if total_heating_events > 0 else 0,
                'annual_solar_savings_R': total_solar_savings,
                'cost_without_solar_R': total_cost_without_solar,
                'savings_percentage': savings_percentage,
                'solar_heating_time_percentage': solar_heating_time_percentage,
                'grid_heating_time_percentage': grid_heating_time_percentage
            }

            return {
                'profile': household_id,
                'kpis': kpis,
                'cost_R': total_cost,
                'success': True
            }

        else:
            print(f"  ERROR: No chunk results generated")
            return {'profile': household_id, 'kpis': {}, 'cost_R': 0, 'success': False, 'error': 'No chunk results'}

    except Exception as e:
        print(f"  ERROR in simulate_household_efficient: {e}")
        import traceback
        traceback.print_exc()
        return {'profile': os.path.basename(profile_path), 'kpis': {}, 'cost_R': 0, 'success': False, 'error': str(e)}