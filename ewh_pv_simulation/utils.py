CHUNK_SIZE = 1000

def map_season(ts):
    return 'high' if ts.month in [11, 12, 1, 2, 3, 4] else 'low'

def map_day_type(ts):
    return 'weekend' if ts.weekday() >= 5 else 'weekday'

def get_hourly_rate(ts, tariff):
    try:
        season = map_season(ts)
        day_type = map_day_type(ts)
        hour = int(ts.hour)
        rows = tariff[(tariff['season'] == season) & (tariff['day_type'] == day_type)]

        for _, row in rows.iterrows():
            start, end = int(row['start_hour']), int(row['end_hour'])
            if start > end:
                if hour >= start or hour < end:
                    return float(row['rate_R_per_kWh'])
            else:
                if start <= hour < end:
                    return float(row['rate_R_per_kWh'])
        return 1.5
    except:
        return 1.5
