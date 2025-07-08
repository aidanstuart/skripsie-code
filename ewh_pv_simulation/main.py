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
from config import TANK_PARAMS, SIM_PARAMS, SYSTEM_PARAMS, MODULE_NAME
from demand import DemandProfile
import glob
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def load_irradiance(path: str) -> pd.DataFrame:
    """Load and process irradiance data from Solcast CSV format."""
    try:
        print(f"Loading irradiance data from: {path}")
        
        # Read CSV with proper datetime parsing
        df = pd.read_csv(path, parse_dates=['period_end'])
        
        # Set index to period_end
        df.set_index('period_end', inplace=True)
        
        # Rename columns to match what the PV module expects
        column_mapping = {
            'gti': 'poa_global',
            'air_temp': 'temp_air',
            'wind_speed_10m': 'wind_speed',
            'dni': 'dni',
            'ghi': 'ghi',
            'dhi': 'dhi',
        }
        
        # Check if required columns exist
        required_original_cols = ['gti', 'air_temp', 'wind_speed_10m', 'dni', 'ghi', 'dhi']
        missing_cols = [col for col in required_original_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in irradiance data: {missing_cols}")
        
        # Apply column mapping
        df = df.rename(columns=column_mapping)
        
        # Select only the columns we need
        df = df[['poa_global', 'temp_air', 'wind_speed', 'dni', 'ghi', 'dhi']]
        
        # Remove timezone info and sort
        df.index = df.index.tz_localize(None)
        df.sort_index(inplace=True)
        
        print(f"Loaded {len(df)} irradiance records")
        return df
        
    except Exception as e:
        print(f"Error loading irradiance data: {e}")
        raise

def load_tariff(path: str) -> pd.DataFrame:
    """Load TOU tariff data."""
    try:
        print(f"Loading tariff data from: {path}")
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
        
        # Filter by season and day_type first
        filtered_tariff = tariff[
            (tariff['season'] == season) &
            (tariff['day_type'] == day_type)
        ]
        
        # Handle time periods that may wrap around midnight
        for _, row in filtered_tariff.iterrows():
            start_hour = int(row['start_hour'])
            end_hour = int(row['end_hour'])
            
            # Check if time period wraps around midnight (e.g., 23-7)
            if start_hour > end_hour:
                # Period wraps around midnight
                if hour >= start_hour or hour < end_hour:
                    return float(row['rate_R_per_kWh'])
            else:
                # Normal period within same day
                if start_hour <= hour < end_hour:
                    return float(row['rate_R_per_kWh'])
        
        # If no match found, print warning and use default
        print(f"Warning: No tariff found for {ts} (season={season}, day_type={day_type}, hour={hour})")
        print(f"Available tariff periods for {season} {day_type}:")
        for _, row in filtered_tariff.iterrows():
            print(f"  {row['tou_period']}: {row['start_hour']}-{row['end_hour']} = R{row['rate_R_per_kWh']}")
        return 1.5  # Default rate
        
    except Exception as e:
        print(f"Error getting hourly rate for {ts}: {e}")
        return 1.5  # Default rate

def simulate_household(profile_path: str, irr_df: pd.DataFrame, tariff: pd.DataFrame) -> dict:
    """Simulate a single household's energy usage and costs."""
    try:
        print(f"Simulating household: {profile_path}")
        
        # Load demand profile
        inlet_temp = SIM_PARAMS["cold_event_temperature"]
        dp = DemandProfile(profile_path, TANK_PARAMS['setpoint'], inlet_temp)
        
        # Get energy demand and align with irradiance data
        energy_demand = dp.get_draw_energy()
        
        # Align demand with irradiance timeline
        draw = energy_demand.reindex(irr_df.index, method='nearest').fillna(0)
        
        # Initialize PV system
        try:
            cec_mod = pvlib.pvsystem.retrieve_sam('CECMod')
            module_params = cec_mod[MODULE_NAME]
            pv_sys = PVModule(module_params, SYSTEM_PARAMS)
            pv_power = pv_sys.get_power(irr_df)
        except Exception as e:
            print(f"Warning: PV system initialization failed: {e}")
            print("Using zero PV power")
            pv_power = pd.Series(0.0, index=irr_df.index)
        
        # Initialize tank
        tank = StratifiedTank(**TANK_PARAMS)
        tank.initialize(TANK_PARAMS['setpoint'])
        
        # Simulation timestep
        dt_h = TANK_PARAMS['dt_s'] / 3600
        
        # Run simulation
        records = []
        for ts in irr_df.index:
            try:
                # Get PV power available
                p_pv = pv_power.loc[ts] if ts in pv_power.index else 0.0
                
                # Solar-medium strategy: use PV first, then grid
                p_pv_used = min(p_pv, tank.element_rating)
                p_grid = max(0, tank.element_rating - p_pv_used)
                
                # Get ambient temperature and demand
                t_amb = irr_df.loc[ts, 'temp_air']
                demand = draw.loc[ts] if ts in draw.index else 0.0
                
                # Update tank state
                top_temp, bot_temp = tank.step(p_grid + p_pv_used, demand, t_amb)
                
                # Record results
                records.append({
                    'ts': ts,
                    'top_T': top_temp,
                    'bot_T': bot_temp,
                    'grid_kwh': p_grid * dt_h,
                    'pv_kwh': p_pv_used * dt_h,
                    'demand_kwh': demand,
                    'pv_available': p_pv
                })
                
            except Exception as e:
                print(f"Error at timestamp {ts}: {e}")
                continue
        
        # Convert to DataFrame
        df = pd.DataFrame(records).set_index('ts')
        
        # Aggregate to hourly
        hourly = df.resample('H').agg({
            'grid_kwh': 'sum',
            'pv_kwh': 'sum',
            'demand_kwh': 'sum',
            'pv_available': 'mean',
            'top_T': 'mean',
            'bot_T': 'mean'
        })
        
        # Calculate solar fraction
        total_energy = hourly['grid_kwh'] + hourly['pv_kwh']
        hourly['solar_fraction'] = hourly['pv_kwh'] / total_energy.replace(0, np.nan)
        
        # Calculate costs
        hourly['rate_R'] = hourly.index.to_series().apply(lambda x: get_hourly_rate(x, tariff))
        hourly['cost_R'] = hourly['grid_kwh'] * hourly['rate_R']
        
        # Calculate KPIs
        kpis = {
            'annual_grid_kwh': hourly['grid_kwh'].sum(),
            'annual_solar_kwh': hourly['pv_kwh'].sum(),
            'annual_demand_kwh': hourly['demand_kwh'].sum(),
            'solar_fraction': hourly['pv_kwh'].sum() / (hourly['grid_kwh'].sum() + hourly['pv_kwh'].sum()),
            'cold_draw_pct': (hourly['top_T'] < TANK_PARAMS['min_usage_temperature']).mean() * 100,
            'avg_temp': hourly['top_T'].mean(),
        }
        
        total_cost = hourly['cost_R'].sum()
        
        print(f"Completed simulation for {profile_path}")
        print(f"  - Annual grid consumption: {kpis['annual_grid_kwh']:.1f} kWh")
        print(f"  - Annual solar consumption: {kpis['annual_solar_kwh']:.1f} kWh")
        print(f"  - Solar fraction: {kpis['solar_fraction']:.1%}")
        print(f"  - Total cost: R{total_cost:.2f}")
        
        return {
            'profile': profile_path,
            'hourly': hourly,
            'kpis': kpis,
            'cost_R': total_cost
        }
        
    except Exception as e:
        print(f"Error simulating household {profile_path}: {e}")
        return {
            'profile': profile_path,
            'hourly': pd.DataFrame(),
            'kpis': {},
            'cost_R': 0
        }

def main():
    """Main execution function."""
    try:
        # Load input data
        irr_path = 'ewh_pv_simulation/solar_data/Stellenbosch/irradiance_data/solcast_2024_whole_year.csv'
        tariff_path = 'ewh_pv_simulation/tou_tariff.csv'
        
        if not os.path.exists(irr_path):
            raise FileNotFoundError(f"Irradiance data not found: {irr_path}")
        if not os.path.exists(tariff_path):
            raise FileNotFoundError(f"Tariff data not found: {tariff_path}")
        
        irr = load_irradiance(irr_path)
        tariff = load_tariff(tariff_path)
        
        # Find profile files
        profile_pattern = 'ewh_pv_simulation/user_data/Source_Data/ewh_profile*.csv'
        profiles = sorted(glob.glob(profile_pattern))
        
        if not profiles:
            raise FileNotFoundError(f"No profile files found with pattern: {profile_pattern}")
        
        print(f"Found {len(profiles)} profile files")
        
        # Run simulations in parallel
        print("Starting parallel simulations...")
        results = Parallel(n_jobs=-1)(
            delayed(simulate_household)(f, irr, tariff) for f in profiles
        )
        
        # Compile results
        kpi_data = []
        for r in results:
            if r['kpis']:  # Only include successful simulations
                kpi_record = {**r['kpis'], 'cost_R': r['cost_R'], 'profile': r['profile']}
                kpi_data.append(kpi_record)
        
        if kpi_data:
            df_kpi = pd.DataFrame(kpi_data)
            output_path = 'results_kpis.csv'
            df_kpi.to_csv(output_path, index=False)
            print(f"Results saved to: {output_path}")
            
            # Print summary statistics
            print("\n=== SUMMARY STATISTICS ===")
            print(f"Successfully simulated {len(kpi_data)} households")
            print(f"Average annual grid consumption: {df_kpi['annual_grid_kwh'].mean():.1f} kWh")
            print(f"Average annual solar consumption: {df_kpi['annual_solar_kwh'].mean():.1f} kWh")
            print(f"Average solar fraction: {df_kpi['solar_fraction'].mean():.1%}")
            print(f"Average annual cost: R{df_kpi['cost_R'].mean():.2f}")
            print(f"Average cold draw percentage: {df_kpi['cold_draw_pct'].mean():.1f}%")
        else:
            print("No successful simulations to save")
            
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise

if __name__ == '__main__':
    main()