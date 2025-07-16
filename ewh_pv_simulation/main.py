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

    all_dfs = []
    output_excel = 'simulation_summary_all_categories.xlsx'
    with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
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

if __name__ == '__main__':
    main()

