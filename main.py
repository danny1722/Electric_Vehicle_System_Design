import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from initialize_data import load_route_data
from simulation import TrainSimulation

def main():
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
    safety_factor = 0.2     # Safety factor for how much power remains at the end of the day
    round_trips = 19    # Number of round trips to be made in a single day

    #https://www.molicel.com/wp-content/uploads/Product-Data-Sheet-of-INR-18650-P30B-80111-2.pdf
    Li_ion_energy_density = 234     # Wh/kg
    Li_ion_charge_rate = 0.9      # W/Wh
    Li_ion_discharge_rate = 9    # W/Wh

    #https://maxwell.com/wp-content/uploads/2025/05/3003345.3_160V-10F_EN.3_20250409.pdf
    cap_energy_density = 5.1        # Wh/kg
    cap_specific_power = 2.6        # kW/kg
    cap_charge_rate = (cap_specific_power*1000) / cap_energy_density  # W/Wh

    route_data = load_route_data("Apeldoorn_Zutphun")

    print(f"Station Stops: {route_data['station_stops']}")
    print(f"Speed Limit Distances: {route_data['speed_limit_dist']}")
    print(f"Speed Limit Values: {route_data['speed_limit_val']}")
    print(f"Electrified Stations: {route_data['electrified_stations']}")
    print(f"Electrified Start Positions: {route_data['electrified_start']}")
    print(f"Electrified Stop Positions: {route_data['electrified_stop']}")
    print(f"Stop Times: {route_data['stop_time']}")
    print(f"Route Length: {route_data['route_length']}")

    sim = TrainSimulation(
        route_length=route_data['route_length'],
        station_stops=route_data['station_stops'],
        stop_time=route_data['stop_time'],
        speed_limit_dist=route_data['speed_limit_dist'],
        speed_limit_val=route_data['speed_limit_val'],
        electrified_start=route_data['electrified_start'],
        electrified_stop=route_data['electrified_stop'],
        electrified_stations=route_data['electrified_stations'],
        acc=acc,
        dec=dec,    
        max_speed=max_speed,
        mass=mass,
        Cd=Cd,
        Af=Af,
        p=p,
        Cr=Cr,
        motor_eff=motor_eff,
        regen_eff=regen_eff,
        charge_rate=Li_ion_charge_rate,
        discharge_rate=Li_ion_discharge_rate,
        dt=1.0
    )

    #sim.run_simulation()
 
    battery_capacity = sim.optimize_battery_capacity(
        target_final_charge=safety_factor,
        tol=0.01,
        max_iter=20,
        step_size=0.05,
        round_trips=round_trips
    )

    t = sim.t
    v = sim.v
    x = sim.x

    power = sim.regen_power
    battery_charge = sim.battery_charge

    plt.figure()
    plt.plot(t, v * 3.6)
    plt.xlabel("Distance [km]")
    plt.ylabel("Speed [km/h]")
    plt.title("Speed profile along route")
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(t, power)
    plt.xlabel("distance [km]")
    plt.ylabel("Power [MW]")
    plt.title("Power profile along route")
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(x / 1000, battery_charge)
    plt.xlabel("distance [km]")
    plt.ylabel("Battery charge [kWh]")
    plt.title("Battery charge profile along route")
    plt.grid()
    plt.show()

if __name__=="__main__":
    main()