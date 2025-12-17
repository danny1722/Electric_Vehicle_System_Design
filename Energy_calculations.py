import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

acc = 0.5       # m/s²
dec = 0.5       # m/s²
mass = 83000    # kg
Cd = 2.1        # Aerodynamic drag coefficient
p = 1.225       # Air density
Af =  8.4       # m² Equivalent frontal area
Cr = 0.0015     # Rolling resistance
regen_eff = 0.6     # Efficiency of regenerative breaking
motor_eff = 0.85    # Efficiency of motor and drive train
max_speed = 120 / 3.6   # m/s

#https://www.molicel.com/wp-content/uploads/Product-Data-Sheet-of-INR-18650-P30B-80111-2.pdf
Li_ion_energy_density = 234     # Wh/kg
Li_ion_charge_rate = 0.833      # W/Wh
Li_ion_discharge_rate = 8.33    # W/Wh

#https://maxwell.com/wp-content/uploads/2025/05/3003345.3_160V-10F_EN.3_20250409.pdf
cap_energy_density = 5.1        # Wh/kg
cap_specific_power = 2.6        # kW/kg
cap_charge_rate = (2.6*1000) / 5.1  # W/Wh

# --------------------
# Load data
# --------------------
station_list = pd.read_excel("Route_Description/Apeldoorn_Zutphun.xlsx", sheet_name="station", index_col=0)
speed_limit_df = pd.read_excel("Route_Description/Apeldoorn_Zutphun.xlsx", sheet_name="speed_limit")
electrified = pd.read_excel("Route_Description/Apeldoorn_Zutphun.xlsx", sheet_name="electrified")

# --------------------
# Convert units
# --------------------
station_list["Total distance"] *= 1000  # km → m
speed_limit_df["Distance"] *= 1000           # km → m
speed_limit_df["Total distance"] *= 1000     # km → m
speed_limit_df["speed limit"] = speed_limit_df["speed limit"] * (1000/3600)  # km/h → m/s
electrified["Start electrification"] *= 1000     # km → m
electrified["Stop electrification"] *= 1000      # km → m
station_list["Electrified"] = station_list["Electrified"].str.strip().str.lower() == "yes"

# --------------------
# Create piecewise speed limit function
# --------------------
def speed_limit_at(x):
    for i in range(len(speed_limit_df)):
        if x <= speed_limit_df["Total distance"].iloc[i]:
            return speed_limit_df["speed limit"].iloc[i]
    return speed_limit_df["speed limit"].iloc[-1]

# --------------------
# Prepare simulation
# --------------------
route_length = station_list["Total distance"].iloc[-1]
dx = 1.0  # step size in meters

positions = np.arange(0, route_length + dx, dx)
speed = np.zeros_like(positions)    # m/s
dt = np.zeros_like(positions)       # s
dv = np.zeros_like(positions)       # m/s
power = np.zeros_like(positions)    #MW
regen_power = np.zeros_like(positions)  # MW
total_power = 0
total_power_regen = 0
total_time = 0

station_stops = station_list["Total distance"].values
speed_limit_positions = speed_limit_df["Total distance"].values
speed_limit_values = speed_limit_df["speed limit"].values

# --------------------
# Simulate speed along route
# --------------------
v = 0

for i, x in enumerate(positions):
    # current speed limit at position x
    limit = speed_limit_at(x)

    # --- find next station (target speed = 0) ---
    future_stations = station_stops[station_stops > x]
    next_station = future_stations[0] if len(future_stations) > 0 else None
    dist_to_station = (next_station - x) if next_station is not None else np.inf
    target_speed_station = 0.0

    future_limit_positions = speed_limit_positions[speed_limit_positions >= x]

    # Initialize limit change as "not relevant"
    dist_to_limit_change = np.inf
    next_limit_value = limit

    if future_limit_positions.size > 1:
        next_limit_pos = future_limit_positions[1]  # start of next segment
        next_limit_value = speed_limit_at(next_limit_pos)

        # Only consider if it's a decrease
        if next_limit_value < limit:
            dist_to_limit_change = future_limit_positions[0] - x
        else:
            next_limit_value = limit  # irrelevant

    # --- decide which event is the earliest relevant one ---
    # Compare distance to station and to limit-reduction
    if dist_to_station <= dist_to_limit_change:
        # station is the relevant event
        dist_to_event = dist_to_station
        v_target = target_speed_station
        event_type = "station"
    else:
        # speed reduction is the relevant event
        dist_to_event = dist_to_limit_change
        v_target = next_limit_value
        event_type = "limit_reduction"

    # --- compute braking distance needed to reduce from current v to v_target ---
    # braking distance formula: d = (v^2 - v_t^2) / (2 * dec) if v > v_t else 0
    if v > v_target:
        braking_distance_needed = (v ** 2 - v_target ** 2) / (2.0 * dec)
    else:
        braking_distance_needed = 0.0

        # if braking distance needed is greater or equal than distance to event -> must brake now
    must_brake = braking_distance_needed >= dist_to_event

    # --- integrate speed over spatial step dx ---
    # convert dx -> dt using current speed: dt = dx / v (avoid division by zero)
    # we use small cutoffs to prevent huge dt when v ~ 0
    if must_brake:
        # braking: reduce towards v_target, but ensure non-negative
        # Use dt approximation; when v is very small, we cap denominator to avoid huge value
        dt[i] = dx / max(v, 0.01)
        dv[i] = -dec * dt[i]
        v_new = v - dec * dt[i]
        # prevent going below the target speed (or below zero)
        v = max(v_new, v_target, 0.0)
    else:
        # accelerate but do not exceed current segment limit
        dt[i] = dx / max(v, 1.0)  # cap denominator for initial start to avoid enormous dt
        dv[i] = acc * dt[i] if v + acc * dt[i] < limit else 0
        v = min(v + acc * dt[i], limit)

    speed[i] = v

# --------------------
# Simulate consumed energy
# --------------------
F_roll = Cr * mass * 9.81
for i, x in enumerate(positions):
    v = speed[i]
    F_ad = 0.5 * p * Cd * Af * v**2

    F_tr = mass*dv[i]/max(dt[i], 0.01) + F_ad + F_roll

    if F_tr >= 0:
        power[i] = max(F_tr * v, 0) * 1/motor_eff / 1e6  # Power in MW
        regen_power[i] = (F_tr * v) * 1/motor_eff / 1e6  # Power with regenerative breaking in MW
    else:
        power[i] = max(F_tr * v, 0) / 1e6  # Power in MW
        regen_power[i] = (F_tr * v) * regen_eff / 1e6  # Power with regenerative breaking in MW

    # convert MW to kWh
    total_power += power[i] * 1000 * dt[i] / 3600
    total_power_regen += regen_power[i] * 1000 * dt[i] / 3600

    total_time = total_time + dt[i]


print(f"Total power used for a one way drive without regen: {total_power:.4f} kWh")
print(f"Total power used for a one way drive with regen: {total_power_regen:.4f} kWh")

max_power = np.max(power)
print(f"Max power: {max_power:.4f} MW")

required_battery_capacity = max_power * 1000 / Li_ion_discharge_rate  # kWh
print(f"Required battery capacity to provide required power: {required_battery_capacity:.4f} kwh")

energy_needed = 2 * total_power_regen  # kWh
charge_time = 15 / 60  # hours
required_battery_capacity_charging = energy_needed / Li_ion_charge_rate * (1/charge_time)  # kWh
required_battery_capacity_charging = float(required_battery_capacity_charging)
print(f"Required battery capacity to provide required energy with charging time of 15 min: {required_battery_capacity_charging:.4f} kWh")




# --------------------
# Calculate electrified driving time
# --------------------
electrified_driving_time = 0.0
for _, row in electrified.iterrows():
    start_pos = row['Start electrification']
    stop_pos = row['Stop electrification']

    # find all indices where the train is inside this electrified region
    mask = (positions >= start_pos) & (positions < stop_pos)

    electrified_driving_time += dt[mask].sum()

#print(electrified_driving_time)

# --------------------
# Plot speed profile
# --------------------
plt.figure(figsize=(12, 6))
plt.plot(positions / 1000, speed * 3.6, label="Train speed")

# Mark stations
for name, row in station_list.iterrows():
    plt.axvline(row["Total distance"] / 1000, color="red", linestyle="--", alpha=0.5)
    plt.text(row["Total distance"] / 1000, 2, name, rotation=90, verticalalignment="bottom")

plt.xlabel("Distance along route (km)")
plt.ylabel("Speed (km/h)")
plt.title("Train Speed Profile Along Route Apeldoorn → Zutphen")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# --------------------
# Plot power
# --------------------
plt.figure(figsize=(12, 6))
plt.plot(positions / 1000, power, label="Power")

# Mark stations
for name, row in station_list.iterrows():
    plt.axvline(row["Total distance"] / 1000, color="red", linestyle="--", alpha=0.5)
    plt.text(row["Total distance"] / 1000, 0.01, name, rotation=90, verticalalignment="bottom")

plt.xlabel("Distance along route (km)")
plt.ylabel("Power (MW)")
plt.title("Required Power Along Route Apeldoorn → Zutphen")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# --------------------
# Plot power with regenerative braking
# --------------------
plt.figure(figsize=(12, 6))
plt.plot(positions / 1000, regen_power, label="Power")

# Mark stations
for name, row in station_list.iterrows():
    plt.axvline(row["Total distance"] / 1000, color="red", linestyle="--", alpha=0.5)
    plt.text(row["Total distance"] / 1000, 0.01, name, rotation=90, verticalalignment="bottom")

plt.xlabel("Distance along route (km)")
plt.ylabel("Power (MW)")
plt.title("Required Power with regenerative braking Along Route Apeldoorn → Zutphen")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()