"""
main.py

Memory-efficient solar-medium strategy simulations
and compute costs under a TOU tariff.
"""
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import pvlib
from pv_module import PVModule
from tank import StratifiedTank
from config import TANK_PARAMS, SIM_PARAMS, SYSTEM_PARAMS, MODULE_NAME
from demand import DemandProfile
import glob
import os
import warnings
import gc
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

# Memory optimization settings
CHUNK_SIZE = 1000  # Process 1000 timesteps at a time
MAX_WORKERS = 4    # Reduce parallel workers to prevent memory overload

def load_irradiance_chunked(path: str, chunk_size: int = CHUNK_SIZE) -> pd.DataFrame:
    """Load irradiance data in chunks to save memory."""
    try:
        print(f"Loading irradiance data from: {path}")
        
        # First pass: get basic info about the file
        df_info = pd.read_csv(path, nrows=5)
        print(f"Columns available: {list(df_info.columns)}")
        
        # Load only required columns
        required_cols = ['period_end', 'gti', 'air_temp', 'wind_speed_10m', 'dni', 'ghi', 'dhi']
        df = pd.read_csv(path, usecols=required_cols, parse_dates=['period_end'])
        
        # Set index and rename columns
        df.set_index('period_end', inplace=True)
        
        column_mapping = {
            'gti': 'poa_global',
            'air_temp': 'temp_air',
            'wind_speed_10m': 'wind_speed',
        }
        df = df.rename(columns=column_mapping)
        
        # Remove timezone info and sort
        df.index = df.index.tz_localize(None)
        df.sort_index(inplace=True)
        
        # Sample data to reduce memory usage (every 4th point = 20min intervals instead of 5min)
        df = df.iloc[::4].copy()
        
        print(f"Loaded and sampled {len(df)} irradiance records")
        return df
        
    except Exception as e:
        print(f"Error loading irradiance data: {e}")
        raise

def load_tariff(path: str) -> pd.DataFrame:
    """Load TOU tariff data."""
    try:
        df = pd.read_csv(path)
        print(f"Loaded {len(df)} tariff records")
        return df
    except Exception as e:
        print(f"Error loading tariff data: {e}")
        raise

def map_season(ts: pd.Timestamp) -> str:
    """Map timestamp to season for tariff lookup."""
    return 'high' if ts.month in [11, 12, 1, 2, 3, 4] else 'low'

def map_day_type(ts: pd.Timestamp) -> str:
    """Map timestamp to day type for tariff lookup."""
    return 'weekend' if ts.weekday() >= 5 else 'weekday'

def get_hourly_rate(ts: pd.Timestamp, tariff: pd.DataFrame) -> float:
    """Get electricity rate for a given timestamp."""
    try:
        season = map_season(ts)
        day_type = map_day_type(ts)
        hour = ts.hour
        
        filtered_tariff = tariff[
            (tariff['season'] == season) &
            (tariff['day_type'] == day_type)
        ]
        
        for _, row in filtered_tariff.iterrows():
            start_hour = int(row['start_hour'])
            end_hour = int(row['end_hour'])
            
            if start_hour > end_hour:
                if hour >= start_hour or hour < end_hour:
                    return float(row['rate_R_per_kWh'])
            else:
                if start_hour <= hour < end_hour:
                    return float(row['rate_R_per_kWh'])
        
        return 1.5  # Default rate
        
    except Exception as e:
        return 1.5

def simulate_household_efficient(profile_path: str, irr_df: pd.DataFrame, tariff: pd.DataFrame) -> dict:
    """Memory-efficient simulation of a single household."""
    try:
        household_id = os.path.basename(profile_path)
        
        # Load demand profile with memory optimization
        inlet_temp = SIM_PARAMS["cold_event_temperature"]
        dp = DemandProfile(profile_path, TANK_PARAMS['setpoint'], inlet_temp)
        
        # Sample demand data to match irradiance sampling
        energy_demand = dp.get_draw_energy()
        energy_demand = energy_demand.iloc[::4]  # Match irradiance sampling
        
        # Align demand with irradiance timeline
        draw = energy_demand.reindex(irr_df.index, method='nearest').fillna(0)
        
        # Initialize PV system (only once per household)
        pv_power = None
        try:
            cec_mod = pvlib.pvsystem.retrieve_sam('CECMod')
            module_params = cec_mod[MODULE_NAME]
            pv_sys = PVModule(module_params, SYSTEM_PARAMS)
            pv_power = pv_sys.get_power(irr_df)
        except Exception as e:
            pv_power = pd.Series(0.0, index=irr_df.index)
        
        # Initialize tank
        tank = StratifiedTank(**TANK_PARAMS)
        tank.initialize(TANK_PARAMS['setpoint'])
        
        # Simulation timestep (adjusted for sampling)
        dt_h = (TANK_PARAMS['dt_s'] * 4) / 3600  # 20-minute timesteps
        
        # Process in chunks to save memory
        chunk_results = []
        timestamps = irr_df.index.tolist()
        
        for i in range(0, len(timestamps), CHUNK_SIZE):
            chunk_end = min(i + CHUNK_SIZE, len(timestamps))
            chunk_ts = timestamps[i:chunk_end]
            
            chunk_records = []
            for ts in chunk_ts:
                try:
                    # Get PV power available
                    p_pv = pv_power.loc[ts] if ts in pv_power.index else 0.0
                    
                    # Solar-medium strategy
                    p_pv_used = min(p_pv, tank.element_rating)
                    p_grid = max(0, tank.element_rating - p_pv_used)
                    
                    # Get ambient temperature and demand
                    t_amb = irr_df.loc[ts, 'temp_air']
                    demand = draw.loc[ts] if ts in draw.index else 0.0
                    
                    # Update tank state
                    top_temp, bot_temp = tank.step(p_grid + p_pv_used, demand, t_amb)
                    
                    # Only store essential data
                    chunk_records.append({
                        'grid_kwh': p_grid * dt_h,
                        'pv_kwh': p_pv_used * dt_h,
                        'demand_kwh': demand,
                        'top_T': top_temp,
                        'hour': ts.hour,
                        'month': ts.month,
                        'weekday': ts.weekday()
                    })
                    
                except Exception as e:
                    continue
            
            # Process chunk results immediately
            if chunk_records:
                chunk_df = pd.DataFrame(chunk_records)
                
                # Calculate costs for this chunk
                chunk_df['rate_R'] = chunk_df.apply(
                    lambda row: get_hourly_rate(
                        pd.Timestamp(year=2024, month=row['month'], day=1, hour=row['hour']),
                        tariff
                    ), axis=1
                )
                chunk_df['cost_R'] = chunk_df['grid_kwh'] * chunk_df['rate_R']
                
                # Aggregate chunk results
                chunk_summary = {
                    'grid_kwh': chunk_df['grid_kwh'].sum(),
                    'pv_kwh': chunk_df['pv_kwh'].sum(),
                    'demand_kwh': chunk_df['demand_kwh'].sum(),
                    'cost_R': chunk_df['cost_R'].sum(),
                    'cold_draws': (chunk_df['top_T'] < TANK_PARAMS['min_usage_temperature']).sum(),
                    'total_points': len(chunk_df),
                    'temp_sum': chunk_df['top_T'].sum()
                }
                
                chunk_results.append(chunk_summary)
                
                # Clean up
                del chunk_df, chunk_records
                gc.collect()
        
        # Aggregate all chunks
        if chunk_results:
            total_grid = sum(c['grid_kwh'] for c in chunk_results)
            total_pv = sum(c['pv_kwh'] for c in chunk_results)
            total_demand = sum(c['demand_kwh'] for c in chunk_results)
            total_cost = sum(c['cost_R'] for c in chunk_results)
            total_cold = sum(c['cold_draws'] for c in chunk_results)
            total_points = sum(c['total_points'] for c in chunk_results)
            temp_sum = sum(c['temp_sum'] for c in chunk_results)
            
            kpis = {
                'annual_grid_kwh': total_grid,
                'annual_solar_kwh': total_pv,
                'annual_demand_kwh': total_demand,
                'solar_fraction': total_pv / (total_grid + total_pv) if (total_grid + total_pv) > 0 else 0,
                'cold_draw_pct': (total_cold / total_points * 100) if total_points > 0 else 0,
                'avg_temp': temp_sum / total_points if total_points > 0 else 0,
            }
            
            return {
                'profile': household_id,
                'kpis': kpis,
                'cost_R': total_cost,
                'success': True
            }
        else:
            return {
                'profile': household_id,
                'kpis': {},
                'cost_R': 0,
                'success': False,
                'error': 'No data processed'
            }
        
    except Exception as e:
        return {
            'profile': os.path.basename(profile_path),
            'kpis': {},
            'cost_R': 0,
            'success': False,
            'error': str(e)
        }

def main():
    """Main execution function with memory optimization."""
    try:
        print("Starting memory-efficient simulation...")
        
        # Load input data
        irr_path = 'ewh_pv_simulation/solar_data/Stellenbosch/irradiance_data/solcast_2024_whole_year.csv'
        tariff_path = 'ewh_pv_simulation/tou_tariff.csv'
        
        if not os.path.exists(irr_path):
            raise FileNotFoundError(f"Irradiance data not found: {irr_path}")
        if not os.path.exists(tariff_path):
            raise FileNotFoundError(f"Tariff data not found: {tariff_path}")
        
        # Load data with memory optimization
        irr = load_irradiance_chunked(irr_path)
        tariff = load_tariff(tariff_path)
        
        # Force garbage collection
        gc.collect()
        
        # Find profile files
        profile_pattern = 'ewh_pv_simulation/user_data/Source_Data/ewh_profile*.csv'
        profiles = sorted(glob.glob(profile_pattern))
        
        if not profiles:
            raise FileNotFoundError(f"No profile files found with pattern: {profile_pattern}")
        
        print(f"Found {len(profiles)} profile files")
        print(f"Using {MAX_WORKERS} parallel workers")
        
        # Process households in smaller batches to prevent memory overload
        batch_size = 10  # Process 10 households at a time
        all_results = []
        
        for i in range(0, len(profiles), batch_size):
            batch_end = min(i + batch_size, len(profiles))
            batch_profiles = profiles[i:batch_end]
            
            print(f"Processing batch {i//batch_size + 1}/{(len(profiles)-1)//batch_size + 1} ({len(batch_profiles)} households)")
            
            # Run simulations for this batch
            batch_results = Parallel(n_jobs=MAX_WORKERS)(
                delayed(simulate_household_efficient)(f, irr, tariff) for f in batch_profiles
            )
            
            all_results.extend(batch_results)
            
            # Force garbage collection between batches
            gc.collect()
        
        # Separate successful and failed simulations
        successful = [r for r in all_results if r['success']]
        failed = [r for r in all_results if not r['success']]
        
        if failed:
            print(f"\nFailed simulations: {len(failed)}")
            for f in failed[:5]:  # Show first 5 failures
                print(f"  - {f['profile']}: {f.get('error', 'Unknown error')}")
            if len(failed) > 5:
                print(f"  ... and {len(failed) - 5} more")
        
        # Compile and save results
        if successful:
            kpi_data = []
            for r in successful:
                kpi_record = {**r['kpis'], 'cost_R': r['cost_R'], 'profile': r['profile']}
                kpi_data.append(kpi_record)
            
            df_kpi = pd.DataFrame(kpi_data)
            output_path = 'results_kpis.csv'
            df_kpi.to_csv(output_path, index=False)
            print(f"Results saved to: {output_path}")
            
            # Print summary statistics
            print("\n" + "="*60)
            print("SIMULATION SUMMARY")
            print("="*60)
            print(f"Total households: {len(profiles)}")
            print(f"Successful simulations: {len(successful)}")
            print(f"Failed simulations: {len(failed)}")
            
            if len(successful) > 0:
                print(f"\nENERGY CONSUMPTION (kWh/year)")
                print(f"Grid - Mean: {df_kpi['annual_grid_kwh'].mean():.0f}, Range: {df_kpi['annual_grid_kwh'].min():.0f}-{df_kpi['annual_grid_kwh'].max():.0f}")
                print(f"Solar - Mean: {df_kpi['annual_solar_kwh'].mean():.0f}, Range: {df_kpi['annual_solar_kwh'].min():.0f}-{df_kpi['annual_solar_kwh'].max():.0f}")
                print(f"Demand - Mean: {df_kpi['annual_demand_kwh'].mean():.0f}, Range: {df_kpi['annual_demand_kwh'].min():.0f}-{df_kpi['annual_demand_kwh'].max():.0f}")
                
                print(f"\nPERFORMANCE METRICS")
                print(f"Solar fraction - Mean: {df_kpi['solar_fraction'].mean():.1%}, Range: {df_kpi['solar_fraction'].min():.1%}-{df_kpi['solar_fraction'].max():.1%}")
                print(f"Cold draws - Mean: {df_kpi['cold_draw_pct'].mean():.1f}%, Range: {df_kpi['cold_draw_pct'].min():.1f}%-{df_kpi['cold_draw_pct'].max():.1f}%")
                print(f"Avg temp - Mean: {df_kpi['avg_temp'].mean():.1f}°C, Range: {df_kpi['avg_temp'].min():.1f}°C-{df_kpi['avg_temp'].max():.1f}°C")
                
                print(f"\nCOST ANALYSIS (R/year)")
                print(f"Per household - Mean: R{df_kpi['cost_R'].mean():.0f}, Range: R{df_kpi['cost_R'].min():.0f}-R{df_kpi['cost_R'].max():.0f}")
                print(f"Total all households: R{df_kpi['cost_R'].sum():.0f}")
                
                # Estimate savings
                avg_solar_kwh = df_kpi['annual_solar_kwh'].mean()
                avg_rate = 1.5
                print(f"\nESTIMATED SAVINGS")
                print(f"Avg solar generation per household: {avg_solar_kwh:.0f} kWh/year")
                print(f"Avg savings per household: R{avg_solar_kwh * avg_rate:.0f}/year")
                print(f"Total savings all households: R{df_kpi['annual_solar_kwh'].sum() * avg_rate:.0f}/year")
            
            print("="*60)
            
        else:
            print("No successful simulations to summarize")
            
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    main()