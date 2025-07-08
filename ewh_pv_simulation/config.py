# config.py

# Thermostat / tank parameters
TANK_PARAMS = {
    "setpoint": 60.0,           # °C, your desired tank temperature
    "deadband": 3.0,            # °C, how far below setpoint the heater kicks on
    "volume_l": 150,            # litres
    "c": 4184,                  # J/(kg·K)
    "rho": 1000,                # kg/m3
    "R_th": 0.4807,             # K/W
    "element_rating_kw": 3,     # kW
    "dt_s": 300,                # 5-min intervals
    "min_usage_temperature": 50, # °C
}

# Demand profile parameters
SIM_PARAMS = {
    "cold_event_temperature": 15.0,   # °C, incoming mains water temp
    "min_draw_l_per_event": 2.0,      # L, etc…
}

# System parameters
SYSTEM_PARAMS = {
    'tilt': 30,
    'azimuth': 180,
    'inverter': {'pdc0': 3000,         # inverter DC capacity in W
                 'eta_inv_nom': 0.96}, # nominal efficiency
    'latitude': -33.934,               # e.g. Stellenbosch latitude
    'longitude': 18.860,               # Stellenbosch longitude
    'timezone': 'Africa/Johannesburg',
    'racking_model': 'open_rack_glass_glass',
}

MODULE_NAME = 'Canadian_Solar_Inc__CS5P_220M'  # default CEC module