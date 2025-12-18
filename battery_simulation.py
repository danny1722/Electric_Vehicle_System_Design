import numpy as np

def optimize_storage(
    v, 
    x, 
    power, 
    stop_time,
    station_stops,
    electrified_driving,
    electrified_stations,
    battery_charge_rate,
    battery_discharge_rate,
    supercap_charge_rate,
    battery_energy_density,
    supercap_energy_density,
    energy_consumption,
    round_trips,
    safety_factor,
    dt=1.0
):
    best = None
    best_battery_charge = np.zeros_like(x, dtype=float)
    best_supercap_charge = np.zeros_like(x, dtype=float)

    for battery_capacity in np.linspace(energy_consumption, energy_consumption * 4, 20):     # kWh
        for supercap_capacity in np.linspace(0, 5, 10):  # kWh

            battery_charge, supercap_charge = simulate_energy_storage(
                v, 
                x, 
                power, 
                stop_time, 
                station_stops, 
                electrified_driving,
                electrified_stations,
                battery_capacity,
                battery_capacity * 0.5,
                battery_charge_rate,
                battery_discharge_rate,
                supercap_capacity,
                supercap_capacity * 0.5,
                supercap_charge_rate,
                dt
            )

            safety_capacity = battery_capacity * safety_factor
            charge_difference = battery_charge[0] - battery_charge[-1]
            min_charge = np.max(battery_charge) - np.min(battery_charge)
            remaining_capacity = min_charge - round_trips * charge_difference - safety_capacity

            if remaining_capacity < 0:
                continue  # reject if not enough capacity for round trips

            mass = storage_mass(
                battery_capacity,
                supercap_capacity,
                battery_energy_density,
                supercap_energy_density
            )

            if best is None or mass < best["mass"]:
                best = {
                    "battery_capacity": battery_capacity,
                    "supercap_capacity": supercap_capacity,
                    "mass": mass
                }
                best_battery_charge = battery_charge
                best_supercap_charge = supercap_charge

    return best, best_battery_charge, best_supercap_charge

def storage_mass(
    battery_capacity,      # kWh
    supercap_capacity,     # kWh
    battery_energy_density,  # Wh/kg
    supercap_energy_density  # Wh/kg
):
    mb = battery_capacity * 1000 / battery_energy_density
    ms = supercap_capacity * 1000 / supercap_energy_density
    return mb + ms

def simulate_energy_storage(
    v,
    x,
    power,
    stop_time,
    station_stops,
    electrified_driving,
    electrified_stations,
    battery_capacity,
    battery_starting_charge,
    battery_charge_rate,
    battery_discharge_rate,
    supercap_capacity,
    supercap_starting_charge,
    supercap_charge_rate,
    dt=1.0
):
    current_battery_charge = battery_starting_charge # kWh
    current_supercap_charge = supercap_starting_charge # kWh
    battery_charge = np.zeros_like(x, dtype=float)
    supercap_charge = np.zeros_like(x, dtype=float)
    max_battery_charge = battery_capacity * battery_charge_rate # kW
    max_battery_discharge = battery_capacity * battery_discharge_rate # kW
    max_supercap_charge = supercap_capacity * supercap_charge_rate # kW
    station_idx = 0
    stopped = False

    for i in range(len(x)):
        Pi = power[i] * 1000  # MW → kW

        if electrified_driving[i] and not stopped:
            current_battery_charge += max_battery_charge * dt / 3600  # kWh
        elif Pi >= 0: # Driving power
            demand = Pi
            # First use supercap
            if current_supercap_charge > 0:
                discharge_power = min(Pi, max_supercap_charge)
                current_supercap_charge -= discharge_power * dt / 3600
                demand -= discharge_power
            # Then use battery
            if demand > 0:
                current_battery_charge -= demand * dt / 3600
        elif Pi < 0: #regenerative braking
            # First use battery 
            charge_power = min(-Pi, max_battery_charge)
            current_battery_charge += charge_power * dt / 3600
            
            remaining_power = -Pi - charge_power
            if remaining_power > 0 and supercap_capacity > 0:
                # Then use supercap
                charge_power = min(remaining_power, max_supercap_charge)
                current_supercap_charge += charge_power * dt / 3600

        # --- Station handling ---
        if station_idx < len(station_stops):
            dist_to_station = station_stops[station_idx] - x[i]

            if abs(dist_to_station) <= 20.0 and v[i] < 0.1:
                stopped = True

                if electrified_stations[station_idx]:
                    # Charge battery
                    required_energy = battery_capacity - current_battery_charge  # kWh
                    station_power = required_energy / stop_time[station_idx] * 3600
                    charge_power = min(max_battery_charge, station_power)

                    current_battery_charge += charge_power * dt / 3600
                    current_battery_charge = min(current_battery_charge, battery_capacity)

                    # Charge supercap
                    required_energy_supercap = supercap_capacity - current_supercap_charge  # kWh
                    station_power = required_energy_supercap / stop_time[station_idx] * 3600
                    charge_power = min(max_supercap_charge, station_power)

                    current_supercap_charge += charge_power * dt / 3600
                    current_supercap_charge = min(current_supercap_charge, supercap_capacity)
                    #print(f"Station {station_idx+1}: Charging with {charge_power:.1f} kW for {stop_time[station_idx]:.1f} s")

            # Detect departure
            if stopped and v[i] > 0.1:
                stopped = False
                station_idx += 1

        # Clamp capacity
        current_battery_charge = np.clip(current_battery_charge, 0.0, battery_capacity)
        current_supercap_charge = np.clip(current_supercap_charge, 0.0, supercap_capacity)
        battery_charge[i] = current_battery_charge
        supercap_charge[i] = current_supercap_charge

    return battery_charge, supercap_charge