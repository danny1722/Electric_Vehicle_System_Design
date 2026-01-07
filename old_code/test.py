import numpy as np
import matplotlib.pyplot as plt
import time

max_speed = 120 / 3.6   # m/s

initial_dec = 0.9  # m/s²
dec_drop_off_speed = max_speed * 0.6  # m/s
final_dec = 0.1  # m/s²

initial_acc = 0.9  # m/s²
acc_drop_off_speed = max_speed * 0.6  # m/s
final_acc = 0.1  # m/s²

def deceleration_profile(speed):
    if speed < dec_drop_off_speed:
        return initial_dec
    else:
        dec = initial_dec - (initial_dec - final_dec) * (speed - dec_drop_off_speed) / (max_speed - dec_drop_off_speed)
        return dec
    
def acceleration_profile(speed):
    if speed < acc_drop_off_speed:
        return initial_acc
    else:
        acc = initial_acc - (initial_acc - final_acc) * (speed - acc_drop_off_speed) / (max_speed - acc_drop_off_speed)
        return acc

start = time.time()

# Own implementation
def calculate_stopping_distance(num_points=1000):
    # Speeds to evaluate
    speed_table = np.linspace(0, max_speed, num_points)

    # Storage
    stop_dist_table = np.zeros_like(speed_table)

    deceleration_profiles = np.vectorize(deceleration_profile)
    dec_table = deceleration_profiles(speed_table)

    previous_speed = 0.0
    distance = 0.0
    for i in range(len(speed_table)):
        v = speed_table[i]
        dec = dec_table[i]
        
        current_speed = v

        speed_dif = current_speed - previous_speed
        time = speed_dif / dec if dec != 0 else 0
        distance += current_speed * time - 0.5 * dec * time**2

        stop_dist_table[i] = distance
        previous_speed = current_speed

    return speed_table, stop_dist_table

speed_table, stop_dist_table = calculate_stopping_distance()
print(stop_dist_table[-1])

end = time.time()
print(f"Computation Time: {end - start:.4f} seconds")

# ChatGPT suggested code below
start = time.time()

dt = 0.05
max_speed = 120/3.6

def build_stopping_distance_table(num_points=500):
    speed_table = np.linspace(0, max_speed, num_points)
    stop_dist_table = np.zeros_like(speed_table)

    for i, v0 in enumerate(speed_table):
        v = v0
        d = 0.0
        while v > 0:
            dec = deceleration_profile(v)
            v = max(v - dec * dt, 0)
            d += v * dt
        stop_dist_table[i] = d

    return speed_table, stop_dist_table

speed_table, stop_dist_table = build_stopping_distance_table()
print(stop_dist_table[-1])

end = time.time()
print(f"Computation Time: {end - start:.4f} seconds")

s1, d1 = calculate_stopping_distance()
s2, d2 = build_stopping_distance_table()
print(len(s1), len(s2))

plt.plot(s1, d1, label="incremental method")
plt.plot(s2, d2, label="correct")
plt.legend()
plt.grid()
plt.show()

num_points = 1000
speed_table = np.linspace(0, max_speed, num_points)

deceleration_profiles = np.vectorize(deceleration_profile)
dec_table = deceleration_profiles(speed_table)

plt.plot(speed_table, dec_table)
plt.xlabel("Speed (m/s)")
plt.ylabel("Deceleration (m/s²)")
plt.title("Deceleration Profile")
plt.grid()
plt.show()

acceleration_profiles = np.vectorize(acceleration_profile)
acc_table = acceleration_profiles(speed_table)

plt.plot(speed_table, acc_table)
plt.xlabel("Speed (m/s)")
plt.ylabel("Acceleration (m/s²)")
plt.title("Acceleration Profile")
plt.grid()
plt.show()