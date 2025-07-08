# config.py

# Thermostat / tank parameters
TANK_PARAMS = {
    "setpoint": 70.0,           # °C, your desired tank temperature
    "deadband": 3.0,            # °C, how far below setpoint the heater kicks on
}

# Demand–profile parameters
SIM_PARAMS = {
    "cold_event_temperature": 15.0,   # °C, incoming mains water temp
    "min_draw_l_per_event": 2.0,      # L, etc…
    # …any other DP settings…
}
