"""
main.py

Memory-efficient solar-medium strategy simulations,
organized by usage category, with summarized Excel output.
"""

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import pvlib
from pv_module import PVModule
from tank import StratifiedTank
from config import TANK_PARAMS, SIM_PARAMS, SYSTEM_PARAMS, MODULE_NAME
from demand import DemandProfile
from simulate import simulate_household_efficient
from utils import CHUNK_SIZE

import glob
import os
import warnings
import gc
from datetime import datetime

warnings.filterwarnings('ignore')

CHUNK_SIZE = 1000
MAX_WORKERS = 4

def load_irradiance_chunked(path: str, chunk_size: int = CHUNK_SIZE) -> pd.DataFrame:
    try:
        print(f"Loading irradiance data from: {path}")
        df_info = pd.read_csv(path, nrows=10, sep=None, engine='python')
        available_cols = list(df_info.columns)

        column_mappings = {
            'period_end': ['period_end', 'timestamp'],
            'gti': ['gti', 'poa_global'],
            'air_temp': ['air_temp', 'temperature'],
            'wind_speed_10m': ['wind_speed_10m', 'wind_speed'],
            'dni': ['dni'],
            'ghi': ['ghi'],
            'dhi': ['dhi']
        }

        actual_cols = {}
        for key, options in column_mappings.items():
            for name in options:
                if name in available_cols:
                    actual_cols[key] = name
                    break
            else:
                raise ValueError(f"Missing column for {key}")

        df = pd.read_csv(path, usecols=list(actual_cols.values()), sep=None, engine='python')
        df.rename(columns={v: k for k, v in actual_cols.items()}, inplace=True)
        df['period_end'] = pd.to_datetime(df['period_end'])
        df.set_index('period_end', inplace=True)
        df.index = df.index.tz_localize(None)

        rename_map = {'gti': 'poa_global', 'air_temp': 'temp_air', 'wind_speed_10m': 'wind_speed'}
        df.rename(columns=rename_map, inplace=True)

        for col in ['poa_global', 'temp_air', 'wind_speed', 'dni', 'ghi', 'dhi']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if col in ['poa_global', 'dni', 'ghi', 'dhi']:
                df[col].fillna(0, inplace=True)
            elif col == 'temp_air':
                df[col].fillna(20, inplace=True)
            elif col == 'wind_speed':
                df[col].fillna(2, inplace=True)

        df = df.iloc[::4]
        return df

    except Exception as e:
        print(f"Error loading irradiance: {e}")
        raise

def load_tariff(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def map_season(ts: pd.Timestamp) -> str:
    return 'high' if ts.month in [11, 12, 1, 2, 3, 4] else 'low'

def map_day_type(ts: pd.Timestamp) -> str:
    return 'weekend' if ts.weekday() >= 5 else 'weekday'

def get_hourly_rate(ts: pd.Timestamp, tariff: pd.DataFrame) -> float:
    try:
        season = map_season(ts)
        day_type = map_day_type(ts)
        hour = int(ts.hour)
        subset = tariff[(tariff['season'] == season) & (tariff['day_type'] == day_type)]

        for _, row in subset.iterrows():
            start = int(row['start_hour'])
            end = int(row['end_hour'])
            if start > end:
                if hour >= start or hour < end:
                    return float(row['rate_R_per_kWh'])
            else:
                if start <= hour < end:
                    return float(row['rate_R_per_kWh'])

        return 1.5
    except:
        return 1.5

def create_summary_statistics(results_by_category: dict) -> pd.DataFrame:
    """
    Create a summary statistics DataFrame for all categories.
    """
    summary_data = []
    
    for category, results in results_by_category.items():
        successful_results = [r for r in results if r['success']]
        
        if not successful_results:
            print(f"⚠️ No successful results for {category} category")
            continue
            
        # Extract KPIs from all households in this category
        kpis_list = [r['kpis'] for r in successful_results]
        costs_list = [r['cost_R'] for r in successful_results]
        
        if not kpis_list:
            continue
            
        # Calculate summary statistics
        summary_stats = {
            'Category': category,
            'Number_of_Households': len(successful_results),
            
            # Solar vs Grid heating percentages
            'Avg_Solar_Heating_Time_Pct': np.mean([kpi.get('solar_heating_time_percentage', 0) for kpi in kpis_list]),
            'Avg_Grid_Heating_Time_Pct': np.mean([kpi.get('grid_heating_time_percentage', 0) for kpi in kpis_list]),
            'Min_Solar_Heating_Time_Pct': np.min([kpi.get('solar_heating_time_percentage', 0) for kpi in kpis_list]),
            'Max_Solar_Heating_Time_Pct': np.max([kpi.get('solar_heating_time_percentage', 0) for kpi in kpis_list]),
            
            # Energy consumption
            'Avg_Annual_Grid_kWh': np.mean([kpi.get('annual_grid_kwh', 0) for kpi in kpis_list]),
            'Avg_Annual_Solar_kWh': np.mean([kpi.get('annual_solar_kwh', 0) for kpi in kpis_list]),
            'Avg_Annual_Demand_kWh': np.mean([kpi.get('annual_demand_kwh', 0) for kpi in kpis_list]),
            'Avg_Solar_Fraction': np.mean([kpi.get('solar_fraction', 0) for kpi in kpis_list]),
            
            # Cost savings
            'Avg_Annual_Cost_R': np.mean(costs_list),
            'Avg_Cost_Without_Solar_R': np.mean([kpi.get('cost_without_solar_R', 0) for kpi in kpis_list]),
            'Avg_Solar_Savings_R': np.mean([kpi.get('annual_solar_savings_R', 0) for kpi in kpis_list]),
            'Avg_Savings_Percentage': np.mean([kpi.get('savings_percentage', 0) for kpi in kpis_list]),
            'Min_Savings_Percentage': np.min([kpi.get('savings_percentage', 0) for kpi in kpis_list]),
            'Max_Savings_Percentage': np.max([kpi.get('savings_percentage', 0) for kpi in kpis_list]),
            
            # Total category savings
            'Total_Annual_Savings_R': np.sum([kpi.get('annual_solar_savings_R', 0) for kpi in kpis_list]),
            'Total_Grid_kWh': np.sum([kpi.get('annual_grid_kwh', 0) for kpi in kpis_list]),
            'Total_Solar_kWh': np.sum([kpi.get('annual_solar_kwh', 0) for kpi in kpis_list]),
            
            # Performance metrics
            'Avg_Cold_Draw_Pct': np.mean([kpi.get('cold_draw_pct', 0) for kpi in kpis_list]),
            'Avg_Tank_Temp_C': np.mean([kpi.get('avg_temp', 0) for kpi in kpis_list])
        }
        
        summary_data.append(summary_stats)
    
    return pd.DataFrame(summary_data)

def print_summary_report(summary_df: pd.DataFrame):
    """
    Print a formatted summary report to console.
    """
    print("\n" + "="*80)
    print("SOLAR WATER HEATING SIMULATION SUMMARY")
    print("="*80)
    
    for _, row in summary_df.iterrows():
        category = row['Category']
        print(f"\n📊 {category.upper()} USAGE CATEGORY:")
        print(f"   Number of households: {row['Number_of_Households']}")
        
        print(f"\n   🔥 HEATING ENERGY SOURCE:")
        print(f"   • Solar heating:  {row['Avg_Solar_Heating_Time_Pct']:.1f}% (range: {row['Min_Solar_Heating_Time_Pct']:.1f}% - {row['Max_Solar_Heating_Time_Pct']:.1f}%)")
        print(f"   • Grid heating:   {row['Avg_Grid_Heating_Time_Pct']:.1f}%")
        
        print(f"\n   💰 COST SAVINGS:")
        print(f"   • Average annual cost: R{row['Avg_Annual_Cost_R']:.2f}")
        print(f"   • Cost without solar:  R{row['Avg_Cost_Without_Solar_R']:.2f}")
        print(f"   • Average savings:     R{row['Avg_Solar_Savings_R']:.2f} ({row['Avg_Savings_Percentage']:.1f}%)")
        print(f"   • Savings range:       {row['Min_Savings_Percentage']:.1f}% - {row['Max_Savings_Percentage']:.1f}%")
        print(f"   • Total category savings: R{row['Total_Annual_Savings_R']:.2f}")
        
        print(f"\n   ⚡ ENERGY CONSUMPTION:")
        print(f"   • Average grid consumption:  {row['Avg_Annual_Grid_kWh']:.0f} kWh")
        print(f"   • Average solar consumption: {row['Avg_Annual_Solar_kWh']:.0f} kWh")
        print(f"   • Solar fraction:            {row['Avg_Solar_Fraction']:.1%}")
        
        print(f"\n   🌡️ PERFORMANCE:")
        print(f"   • Cold draws: {row['Avg_Cold_Draw_Pct']:.1f}%")
        print(f"   • Average tank temperature: {row['Avg_Tank_Temp_C']:.1f}°C")
        
        print("-" * 60)
    
    # Overall totals
    total_households = summary_df['Number_of_Households'].sum()
    total_savings = summary_df['Total_Annual_Savings_R'].sum()
    total_solar_kwh = summary_df['Total_Solar_kWh'].sum()
    total_grid_kwh = summary_df['Total_Grid_kWh'].sum()
    
    print(f"\n🎯 OVERALL RESULTS:")
    print(f"   • Total households simulated: {total_households}")
    print(f"   • Total annual savings: R{total_savings:.2f}")
    print(f"   • Total solar energy used: {total_solar_kwh:.0f} kWh")
    print(f"   • Total grid energy used: {total_grid_kwh:.0f} kWh")
    print(f"   • Overall solar fraction: {total_solar_kwh/(total_solar_kwh + total_grid_kwh):.1%}")
    print("="*80)

def main():
    print("🔄 Starting categorized solar simulations...")

    irr_path = 'ewh_pv_simulation/solar_data/Stellenbosch/irradiance_data/solcast_2024_whole_year.csv'
    tariff_path = 'ewh_pv_simulation/tou_tariff.csv'
    irr = load_irradiance_chunked(irr_path)
    tariff = load_tariff(tariff_path)
    gc.collect()

    base_dir = 'ewh_pv_simulation/user_data/Classified_Profiles'
    categories = ['Light', 'Medium', 'Heavy']
    results_by_category = {}

    for category in categories:
        profile_paths = sorted(glob.glob(os.path.join(base_dir, category, '*.csv')))
        print(f"\n📁 Processing category: {category} ({len(profile_paths)} files)")

        category_results = []
        for i in range(0, len(profile_paths), 10):
            batch = profile_paths[i:i+10]
            print(f"  ➤ Batch {i//10 + 1} of {(len(profile_paths)-1)//10 + 1}")
            batch_results = Parallel(n_jobs=MAX_WORKERS)(
                delayed(simulate_household_efficient)(f, irr, tariff) for f in batch
            )
            category_results.extend(batch_results)
            gc.collect()

        results_by_category[category] = category_results

    # Create summary statistics
    print("\n📈 Creating summary statistics...")
    summary_df = create_summary_statistics(results_by_category)
    
    # Print summary report
    print_summary_report(summary_df)

    # Save results to Excel
    all_dfs = []
    output_excel = 'simulation_summary_all_categories.xlsx'
    with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
        # Write summary sheet first
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Write individual category sheets
        for cat, res_list in results_by_category.items():
            successful = [r for r in res_list if r['success']]
            df_cat = pd.DataFrame([
                {**r['kpis'], 'cost_R': r['cost_R'], 'profile': r['profile'], 'category': cat}
                for r in successful
            ])
            print(f"📝 Writing {len(df_cat)} rows for {cat}...")
            if df_cat.empty:
                print(f"⚠️ No data for {cat} – possible simulation failure or data issue.")

            df_cat.to_excel(writer, sheet_name=cat, index=False)
            all_dfs.append(df_cat)

    print(f"\n✅ Results saved to: {output_excel}")
    print(f"✅ Summary statistics saved to 'Summary' sheet")

if __name__ == '__main__':
    main()