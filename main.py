import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from initialize_data import load_route_data
from simulation import TrainSimulation

def main():
    # Train parameters
    mass = 83000    # kg
    Cd = 2.1        # Aerodynamic drag coefficient
    p = 1.225       # Air density
    Af =  8.4       # m² Equivalent frontal area
    Cr = 0.0015     # Rolling resistance
    regen_eff = 0.6     # Efficiency of regenerative breaking
    motor_eff = 0.85    # Efficiency of motor and drive train
    max_speed = 120 / 3.6   # m/s
    utilities_power = 35    # Kw Power used by utilities like air conditioning, lighting, etc.
    safety_factor = 0.2     # Safety factor for how much power remains at the end of the day
    round_trips = 19    # Number of round trips to be made in a single day
    dt = 0.1             # Time step for simulation

    # Acceleration and deceleration profile parameters
    initial_dec = 0.6  # m/s²
    dec_drop_off_speed = max_speed * 0.3  # m/s
    final_dec = 0.1  # m/s²

    initial_acc = 0.6  # m/s²
    acc_drop_off_speed = max_speed * 0.3  # m/s
    final_acc = 0.1  # m/s²

    # Pantograph parameters
    pantograph_weight = 158 # kg
    pantograph_count = 2
    

    #https://www.molicel.com/wp-content/uploads/Product-Data-Sheet-of-INR-18650-P30B-80111-2.pdf
    Li_ion_energy_density = 234     # Wh/kg
    Li_ion_charge_rate = 0.9      # W/Wh
    Li_ion_discharge_rate = 9    # W/Wh

    route_data = load_route_data("Apeldoorn_Zutphun")

    sim = TrainSimulation(
        route_data=route_data,
        initial_acc=initial_acc,
        acc_drop_off_speed=acc_drop_off_speed,
        final_acc=final_acc,
        initial_dec=initial_dec,
        dec_drop_off_speed=dec_drop_off_speed,
        final_dec=final_dec,    
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
        dt=dt,
        utilities_power=utilities_power
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

    max_power = np.max(sim.regen_power)
    print(f"Max Power: {max_power:.2f} MW")

    print(f"Regenerative Energy over route: {sim.regenerated_power:.2f} kWh")
    print(f"Total energy regenrated over {round_trips} round trips: {sim.regenerated_power * round_trips:.2f} kWh")

    print(f"Pantograph Energy drawn over route: {np.sum(sim.pantograph_power):.2f} kWh")
    print(f"Total pantograph energy drawn over {round_trips} round trips: {np.sum(sim.pantograph_power) * round_trips:.2f} kWh")
    print(f"Max Pantograph Power drawn over route: {np.max(sim.pantograph_power) * (1/dt) * 3600:.2f} kW")

    sim2 = TrainSimulation(
        route_data=route_data,
        initial_acc=initial_acc,
        acc_drop_off_speed=acc_drop_off_speed,
        final_acc=final_acc,
        initial_dec=initial_dec,
        dec_drop_off_speed=dec_drop_off_speed,
        final_dec=final_dec,    
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
        dt=dt,
        using_pantograph=False,
        utilities_power=utilities_power
    )

    battery_capacity_no_pantograph = sim2.optimize_battery_capacity(
        target_final_charge=safety_factor,
        tol=0.01,
        max_iter=20,
        step_size=0.05,
        round_trips=round_trips
    )

    battery_capacity_difference = battery_capacity_no_pantograph - battery_capacity
    print(f"Battery size reduction by using pantograph: {battery_capacity_difference:.2f} kWh")

    weight_saving = (battery_capacity_difference / Li_ion_energy_density) * 1000  # kg
    print(f"Weight saving by using pantograph: {weight_saving - pantograph_weight * pantograph_count:.2f} kg")

    print(f"power used {sim.total_energy_regen / (sim.t[-1] / 3600):.2f} kW")
    print(f"Power used by utilities: {utilities_power:.2f} kW")

    battery_mass = battery_capacity / Li_ion_energy_density * 1000  # kg
    print(f"Required battery mass: {battery_mass:.2f} kg")

    energy_per_km = (sim.total_energy_regen / (sim.t[-1] / 3600)) / (x[-1] / 1000) + utilities_power / (x[-1] / 1000)  # kWh/km
    print(f"Energy consumption: {energy_per_km:.2f} kWh/km")


    plt.figure()
    plt.plot(t, v * 3.6)
    plt.xlabel("Time [s]")
    plt.ylabel("Speed [km/h]")
    plt.title("Speed profile along route")
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(t, power)
    plt.xlabel("Time [s]")
    plt.ylabel("Power [MW]")
    plt.title("Power profile along route")
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(t, battery_charge)
    plt.xlabel("Time [s]")
    plt.ylabel("Battery charge [kWh]")
    plt.title("Battery charge profile along route")
    plt.grid()
    plt.show()

if __name__=="__main__":
    main()