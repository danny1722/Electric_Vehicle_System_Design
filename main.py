import pandas as pd
import numpy as np
import globals
import matplotlib.pyplot as plt
from initialize_data import load_route_data
from simulation import TrainSimulation

def main(debug=True):
    route_data = load_route_data("Apeldoorn_Zutphun")

    sim = TrainSimulation(
        route_data=route_data,
        initial_acc=globals.initial_acc,
        acc_drop_off_speed=globals.acc_drop_off_speed,
        final_acc=globals.final_acc,
        initial_dec=globals.initial_dec,
        dec_drop_off_speed=globals.dec_drop_off_speed,
        final_dec=globals.final_dec,    
        max_speed=globals.max_speed,
        mass=globals.mass,
        Cd=globals.Cd,
        Af=globals.Af,
        p=globals.p,
        Cr=globals.Cr,
        motor_eff=globals.motor_eff,
        regen_eff=globals.regen_eff,
        charge_rate=globals.Li_ion_charge_rate,
        discharge_rate=globals.Li_ion_discharge_rate,
        dt=globals.dt,
        using_pantograph = globals.using_pantograph
    )

    #sim.run_simulation()
 
    globals.battery_capacity = sim.optimize_battery_capacity(
        target_final_charge=globals.safety_factor,
        tol=0.01,
        max_iter=20,
        step_size=0.05,
        round_trips=globals.round_trips,
        debug = debug
    )

    t = sim.t
    v = sim.v
    x = sim.x

    power = sim.regen_power
    battery_charge = sim.battery_charge

    plt.figure()
    plt.plot(t, v * 3.6)
    plt.xlabel("Time [s]")
    plt.ylabel("Speed [km/h]")
    plt.title("Speed profile along route")
    plt.grid()
    # plt.show()
    plt.savefig('static/speedProfile.png')

    plt.figure()
    plt.plot(t, power)
    plt.xlabel("Time [s]")
    plt.ylabel("Power [MW]")
    plt.title("Power profile along route")
    plt.grid()
    # plt.show()
    plt.savefig('static/powerProfile.png')

    plt.figure()
    plt.plot(t, battery_charge)
    plt.xlabel("Time [s]")
    plt.ylabel("Battery charge [kWh]")
    plt.title("Battery charge profile along route")
    plt.grid()
    # plt.show()
    plt.savefig('static/batteryProfile.png')

if __name__=="__main__":
    main()