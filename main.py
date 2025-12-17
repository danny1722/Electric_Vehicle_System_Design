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
safety_factor = 1.2

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
speed_limit_df["speed limit"] *= (1000/3600)  # km/h → m/s
electrified["Start electrification"] *= 1000     # km → m
electrified["Stop electrification"] *= 1000      # km → m
station_list["Electrified"] = station_list["Electrified"].str.strip().str.lower() == "yes"

# --------------------
# Power required to maintain max speed
# --------------------
F_ad = 0.5 * p * Cd * Af * max_speed**2
F_roll = Cr * mass * 9.81
F_tr = F_ad + F_roll
P_wheel = F_tr * max_speed  # W
P_max_speed = P_wheel / motor_eff  # kW
print(f"Power required to maintain max speed: {P_max_speed/1e6:.3f} MW")

# --------------------
# Energy required to reach max_speed and required supercapacitor capacity
# --------------------
dt_acc = 0.1        # time step [s]
v = 0.0
x = 0.0

energy_to_vmax = 0.0    # kWh
cap_capacity = 0.0  # kWh
time_to_vmax = 0.0

F_roll = Cr * mass * 9.81

while v < max_speed:
    F_ad = 0.5 * p * Cd * Af * v**2
    F_tr = mass * acc + F_ad + F_roll

    # Power at wheels
    P_wheel = F_tr * v  # W

    # Electrical power (motor losses)
    P_elec = P_wheel / motor_eff

    # Integrate energy
    energy_to_vmax += P_elec * dt_acc / 3600 / 1000  # kWh
    if P_elec > P_max_speed:
        cap_capacity += (P_elec - P_max_speed) * dt_acc / 3600 / 1000  # kWh
    
    time_to_vmax += dt_acc

    # Integrate speed
    v += acc * dt_acc
    x += v * dt_acc

print(f"Time to reach max speed: {time_to_vmax:.1f} s")
print(f"Distance to reach max speed: {x:.1f} m")
print(f"Energy to reach max speed: {energy_to_vmax:.3f} kWh")
print(f"Super capicitor capacity: {cap_capacity:.3f} kWh")

# --------------------
# Create piecewise speed limit function
# --------------------
def speed_limit_at(x, speed_limit_dist, speed_limit_val):
    for i in range(len(speed_limit_dist)):
        if x <= speed_limit_dist[i]:
            return speed_limit_val[i]
    return speed_limit_val[-1]

def next_speed_limit_drop(x, speed_limit_dist, speed_limit_val):
    for i in range(len(speed_limit_dist) - 1):
        x_change = speed_limit_dist[i]
        v_now = speed_limit_val[i]
        v_next = speed_limit_val[i + 1]
        if x < x_change and v_next < v_now:
            return x_change, v_next

    return None, None

def simulate_speed_profile(
    route_length,
    station_stops,
    speed_limit_dist,
    speed_limit_val,
    stop_time,
    acc,
    dec,
    max_speed,
    dt=1.0
):
    # --------------------
    # Simulation arrays
    # --------------------
    t = [0.0]
    x = [0.0]
    v = [0.0]

    state = "DRIVE"
    station_idx = 0
    dwell_timer = 0.0
    braking = False

    # --------------------
    # Simulation loop
    # --------------------
    while True:

        x_curr = x[-1]
        v_curr = v[-1]
        t_curr = t[-1]

        # --------------------
        # Handle station stop
        # --------------------
        if station_idx < len(station_stops):
            dist_to_station = station_stops[station_idx] - x_curr
        else:
            dist_to_station = np.inf

        braking_dist = v_curr**2 / (2 * dec)

        if (state == "DRIVE" and dist_to_station <= braking_dist + 1.0) or braking:
            # Start braking for station
            a = -dec
            if v_curr <= 0.1:
                v_next = 0.0
                state = "DWELL"
                braking = False
                dwell_timer = stop_time[station_idx]
            else:
                v_next = max(0.0, v_curr + a * dt)
                braking = True

        elif state == "DWELL":
            dwell_timer -= dt
            v_next = 0.0
            if station_idx == len(station_stops) - 1:
                return np.array(t), np.array(x), np.array(v)
            if dwell_timer <= 0:
                state = "DRIVE"
                station_idx += 1

        else:
            # --------------------
            # Normal driving
            # --------------------
            v_limit = min(speed_limit_at(x_curr, speed_limit_dist, speed_limit_val), max_speed)

            # Look ahead for speed limit reduction
            x_drop, v_limit_next = next_speed_limit_drop(x_curr, speed_limit_dist, speed_limit_val)

            must_brake_for_speed = False
            if x_drop is not None and v_limit_next < v_limit:
                dist_to_drop = x_drop - x_curr
                brake_dist_speed = (v_curr**2 - v_limit_next**2) / (2 * dec)

                if brake_dist_speed >= dist_to_drop:
                    must_brake_for_speed = True

            v_min = 0.0
            # Decide acceleration
            if must_brake_for_speed:
                a = -dec
                v_min = v_limit_next
            elif v_curr < v_limit:
                a = acc
            elif v_curr > v_limit:
                a = -dec
                v_min = v_limit
            else:
                a = 0.0

            v_next = np.clip(v_curr + a * dt, v_min, v_limit)

        # --------------------
        # Integrate motion
        # --------------------
        x_next = x_curr + v_next * dt

        t.append(t_curr + dt)
        v.append(v_next)
        x.append(x_next)

# --------------------
# Simulate consumed energy
# --------------------
def simulate_energy(
    v,
    mass,
    Cd,
    Af,
    p,
    Cr,
    motor_eff,
    regen_eff,
    dt = 1.0
):
    a = np.diff(v) / np.maximum(dt, 1e-3)

    power = np.zeros_like(v)
    regen_power = np.zeros_like(v)

    F_roll = Cr * mass * 9.81

    total_energy = 0.0
    total_energy_regen = 0.0

    # Start from index 1 (because of diff)
    for i in range(1, len(v)):
        vi = v[i]
        ai = a[i - 1]

        F_ad = 0.5 * p * Cd * Af * vi**2
        F_tr = mass * ai + F_ad + F_roll

        # --------------------
        # Power calculation
        # --------------------
        if F_tr >= 0:
            Pi = F_tr * vi / motor_eff
            Preg = Pi
        else:
            Pi = F_tr * vi
            Preg = F_tr * vi * regen_eff

        # Convert to MW
        power[i] = Pi / 1e6
        regen_power[i] = Preg / 1e6

        # --------------------
        # Energy integration
        # --------------------
        total_energy += max(power[i], 0) * 1000 * dt / 3600
        total_energy_regen += regen_power[i] * 1000 * dt / 3600

    return power, regen_power, total_energy, total_energy_regen

def main():
    # --------------------
    # Prepare simulation
    # --------------------
    route_length = station_list["Total distance"].iloc[-1]
    dt = 1.0  # step size in seconds

    station_stops = station_list["Total distance"].values
    stop_time = station_list["Stop time (min)"].values * 60  # convert to seconds
    speed_limit_dist = speed_limit_df["Total distance"].values
    speed_limit_val = speed_limit_df["speed limit"].values

    t, x, v = simulate_speed_profile(
        route_length,
        station_stops,
        speed_limit_dist,
        speed_limit_val,
        stop_time,
        acc,
        dec,
        max_speed,
        dt
    )

    # --------------------
    # Reverse the order of stops to simulate return trip
    # --------------------
    reverse_station_stops = np.zeros_like(station_stops)
    for i in range(len(station_stops)):
        reverse_station_stops[i] = route_length - station_stops[-1 - i]

    reverse_speed_limit_dist = np.zeros_like(speed_limit_dist)
    for i in range(len(speed_limit_dist)):
        if i == len(speed_limit_dist) - 1:
            reverse_speed_limit_dist[i] = route_length
        else:
            reverse_speed_limit_dist[i] = route_length - speed_limit_dist[-2 - i]

    reverse_stop_time = stop_time[::-1]
    reverse_speed_limit_val = speed_limit_val[::-1]

    rev_t, rev_x, rev_v = simulate_speed_profile(
        route_length,
        reverse_station_stops,
        reverse_speed_limit_dist,
        reverse_speed_limit_val,
        reverse_stop_time,
        acc,
        dec,
        max_speed,
        dt
    )

    t_total = np.concatenate((t, rev_t + t[-1]))
    x_total = np.concatenate((x, rev_x + x[-1]))
    v_total = np.concatenate((v, rev_v))

    power, regen_power, total_energy, total_energy_regen = simulate_energy(
        v_total,
        mass,
        Cd,
        Af,
        p,
        Cr,
        motor_eff,
        regen_eff,
        dt
    )

    print(f"Total energy consumed for round trip: {total_energy_regen:.3f} kWh")

    cap_charge_time = np.min(stop_time)/3600 #hours
    cap_charge_power = cap_capacity / cap_charge_time  # kW
    print(f"Supercapacitor charge power: {cap_charge_power/1000:.3f} MW")

    total_stops = (len(station_stops) - 1) * 2
    needed_capacity = total_energy - cap_capacity * total_stops

    print(f"Required battery capacity for a round trip: {needed_capacity:.3f} kWh")

    

    plt.figure()
    plt.plot(x_total / 1000, v_total * 3.6)
    plt.xlabel("Distance [km]")
    plt.ylabel("Speed [km/h]")
    plt.title("Speed profile along route (with station stops)")
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(t_total, v_total * 3.6)
    plt.xlabel("time [s]")
    plt.ylabel("Speed [km/h]")
    plt.title("Speed profile along route (with station stops)")
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(x_total / 1000, regen_power)
    plt.xlabel("distance [km]")
    plt.ylabel("Power [MW]")
    plt.title("Power profile along route (with station stops)")
    plt.grid()
    plt.show()

if __name__=="__main__":
    main()