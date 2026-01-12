# Train parameters
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
dt = 0.1             # Time step for simulation

# Acceleration and deceleration profile parameters
initial_dec = 0.6  # m/s²
dec_drop_off_speed = max_speed * 0.3  # m/s
final_dec = 0.1  # m/s²

initial_acc = 0.6  # m/s²
acc_drop_off_speed = max_speed * 0.3  # m/s
final_acc = 0.1  # m/s²

#https://www.molicel.com/wp-content/uploads/Product-Data-Sheet-of-INR-18650-P30B-80111-2.pdf
Li_ion_energy_density = 234     # Wh/kg
Li_ion_charge_rate = 0.9      # W/Wh
Li_ion_discharge_rate = 9    # W/Wh