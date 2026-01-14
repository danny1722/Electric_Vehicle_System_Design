import numpy as np

class TrainSimulation:
    def __init__(
    self,
    route_data,
    initial_acc,
    acc_drop_off_speed,
    final_acc,
    initial_dec,
    dec_drop_off_speed,
    final_dec,
    max_speed,
    mass,
    Cd,
    Af,
    p,
    Cr,
    motor_eff,
    regen_eff,
    charge_rate,
    discharge_rate,
    dt=1.0,
    utilities_percentage=0.0,
    using_pantograph=True
    ):
        self.route_length = route_data['route_length']
        self.station_stops = route_data['station_stops']
        self.speed_limit_dist = route_data['speed_limit_dist']
        self.speed_limit_val = route_data['speed_limit_val']
        self.stop_time = route_data['stop_time']
        self.electrified_start = route_data['electrified_start']
        self.electrified_stop = route_data['electrified_stop']
        self.electrified_stations = route_data['electrified_stations']
        self.initial_acc = initial_acc
        self.acc_drop_off_speed = acc_drop_off_speed
        self.final_acc = final_acc
        self.initial_dec = initial_dec
        self.dec_drop_off_speed = dec_drop_off_speed
        self.final_dec = final_dec
        self.max_speed = max_speed
        self.mass = mass
        self.Cd = Cd
        self.Af = Af
        self.p = p
        self.Cr = Cr
        self.motor_eff = motor_eff
        self.regen_eff = regen_eff
        self.charge_rate = charge_rate
        self.discharge_rate = discharge_rate
        self.dt = dt
        self.utilities_percentage = utilities_percentage
        self.using_pantograph = using_pantograph
        
        self.initilized = False
        self.t = None
        self.x = None
        self.v = None

        self.regenerated_power = 0.0

    def initialize_variables(self):
        if not self.initilized and self.t is not None:
            self.power = np.zeros_like(self.v)
            self.regen_power = np.zeros_like(self.v)
            self.total_energy = 0.0
            self.total_energy_regen = 0.0
            self.capacity = 0.0
            self.utilities_power = 0.0

            self.electrified_driving = np.zeros_like(self.x, dtype=bool)
            self.battery_charge = np.zeros_like(self.x, dtype=float)

            self.pantograph_power = np.zeros_like(self.x, dtype=float)

            self.initilized = True
        elif self.t is None:
            raise ValueError("Speed profile must be simulated before initializing variables.")
    
    def deceleration_profile(self, speed):
        if speed < self.dec_drop_off_speed:
            return self.initial_dec
        else:
            dec = self.initial_dec - (self.initial_dec - self.final_dec) * (speed - self.dec_drop_off_speed) / (self.max_speed - self.dec_drop_off_speed)
            return dec

    def acceleration_profile(self, speed):
        if speed < self.acc_drop_off_speed:
            return self.initial_acc
        else:
            acc = self.initial_acc - (self.initial_acc - self.final_acc) * (speed - self.acc_drop_off_speed) / (self.max_speed - self.acc_drop_off_speed)
            return acc
        
    def build_stopping_distance_table(self, num_points=5000):
        # Speeds to evaluate
        self.speed_table = np.linspace(0, self.max_speed, num_points)

        # Storage
        self.stop_dist_table = np.zeros_like(self.speed_table)

        deceleration_profiles = np.vectorize(self.deceleration_profile)
        dec_table = deceleration_profiles(self.speed_table)

        previous_speed = 0.0
        distance = 0.0
        for i in range(len(self.speed_table)):
            v = self.speed_table[i]
            dec = dec_table[i]
            
            current_speed = v

            speed_dif = current_speed - previous_speed
            time = speed_dif / dec if dec != 0 else 0
            distance += current_speed * time - 0.5 * dec * time**2

            self.stop_dist_table[i] = distance
            previous_speed = current_speed

    def stopping_distance(self, speed, speed_next = 0.0):
        if speed_next == 0.0:
            return np.interp(speed, self.speed_table, self.stop_dist_table)
        else:
            return np.interp(speed, self.speed_table, self.stop_dist_table) - np.interp(speed_next, self.speed_table, self.stop_dist_table)

    def speed_limit_at(self, x):
        for i in range(len(self.speed_limit_dist)):
            if x <= self.speed_limit_dist[i]:
                return self.speed_limit_val[i]
        return self.speed_limit_val[-1]

    def next_speed_limit_drop(self, x):
        for i in range(len(self.speed_limit_dist) - 1):
            x_change = self.speed_limit_dist[i]
            v_now = self.speed_limit_val[i]
            v_next = self.speed_limit_val[i + 1]
            if x < x_change and v_next < v_now:
                return x_change, v_next

        return None, None

    def simulate_speed_profile(self):
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

        self.build_stopping_distance_table()

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
            if station_idx < len(self.station_stops):
                dist_to_station = self.station_stops[station_idx] - x_curr
            else:
                dist_to_station = np.inf

            braking_dist = self.stopping_distance(v_curr)

            if (state == "DRIVE" and dist_to_station <= braking_dist + 1.0) or braking:
                # Start braking for station
                a = -self.deceleration_profile(v_curr)
                if v_curr <= 0.1:
                    v_next = 0.0
                    state = "DWELL"
                    braking = False
                    dwell_timer = self.stop_time[station_idx]
                else:
                    v_next = max(0.0, v_curr + a * self.dt)
                    braking = True

            elif state == "DWELL":
                dwell_timer -= self.dt
                v_next = 0.0
                if station_idx == len(self.station_stops) - 1:
                    return np.array(t), np.array(x), np.array(v)
                if dwell_timer <= 0:
                    state = "DRIVE"
                    station_idx += 1

            else:
                # --------------------
                # Normal driving
                # --------------------
                v_limit = min(self.speed_limit_at(x_curr), self.max_speed)

                # Look ahead for speed limit reduction
                x_drop, v_limit_next = self.next_speed_limit_drop(x_curr)

                must_brake_for_speed = False
                if x_drop is not None and v_limit_next < v_limit:
                    dist_to_drop = x_drop - x_curr
                    brake_dist_speed = self.stopping_distance(v_curr, v_limit_next)

                    if brake_dist_speed >= dist_to_drop:
                        must_brake_for_speed = True

                v_min = 0.0
                # Decide acceleration
                if must_brake_for_speed:
                    a = -self.deceleration_profile(v_curr)
                    v_min = v_limit_next
                elif v_curr < v_limit:
                    a = self.acceleration_profile(v_curr)
                elif v_curr > v_limit:
                    a = -self.deceleration_profile(v_curr)
                    v_min = v_limit
                else:
                    a = 0.0
                #print(f"x: {x_curr:.1f} m, v: {v_curr*3.6:.1f} km/h, v_limit: {v_limit*3.6:.1f} km/h, a: {a:.2f} m/s²")
                v_next = np.clip(v_curr + a * self.dt, v_min, v_limit)

            # --------------------
            # Integrate motion
            # --------------------
            x_next = x_curr + v_next * self.dt

            t.append(t_curr + self.dt)
            v.append(v_next)
            x.append(x_next)

    # --------------------
    # Simulate consumed energy
    # --------------------
    def simulate_energy(self):
        a = np.diff(self.v) / np.maximum(self.dt, 1e-3)

        F_roll = self.Cr * self.mass * 9.81

        # Start from index 1 (because of diff)
        for i in range(1, len(self.v)):
            vi = self.v[i]
            ai = a[i - 1]

            F_ad = 0.5 * self.p * self.Cd * self.Af * vi**2
            F_tr = self.mass * ai + F_ad + F_roll

            # --------------------
            # Power calculation
            # --------------------
            if F_tr >= 0:
                Pi = F_tr * vi / self.motor_eff
                Preg = Pi
            else:
                Pi = F_tr * vi
                Preg = F_tr * vi * self.regen_eff
            # Convert to MW
            self.power[i] = Pi / 1e6
            self.regen_power[i] = Preg / 1e6

            # -------------------
            # Energy integration
            # --------------------
            self.total_energy += max(self.power[i], 0) * 1000 * self.dt / 3600
            self.total_energy_regen += self.regen_power[i] * 1000 * self.dt / 3600

    def electrified_time(self):
        station_electrified_time = 0.0
        for i in range(len(self.electrified_stations) - 1):
            if self.electrified_stations[i]:
                station_electrified_time += self.stop_time[i]

        for start, stop in zip(self.electrified_start, self.electrified_stop):
            self.electrified_driving |= (
                (self.x >= start) &
                (self.x <  stop) &
                (self.v > 0.1)
        )
        electrified_driving_time = self.electrified_driving.sum() * self.dt
        self.total_time_electrified = station_electrified_time + electrified_driving_time

    def simulate_battery_capacity(self, capacity, starting_charge, power):
        current_capacity = starting_charge # kWh
        max_charge = capacity * self.charge_rate # kW
        max_discharge = capacity * self.discharge_rate # kW
        self.battery_charge = np.zeros_like(self.x, dtype=float)
        station_idx = 0
        stopped = False
        self.regenerated_power = 0.0
        self.pantograph_power_total = 0.0

        for i in range(len(self.x)):
            Pi = power[i] * 1000  # MW → kW

            if self.electrified_driving[i] and not stopped and self.using_pantograph: # Drawing power from overhead lines
                current_capacity += max_charge * self.dt / 3600  # kWh
                self.pantograph_power[i] = (max_charge * self.dt / 3600) + (Pi * self.dt / 3600) + (self.utilities_power * self.dt / 3600)
            elif Pi >= 0 and not stopped: # Power consumption
                current_capacity -= Pi * self.dt / 3600 + (self.utilities_power * self.dt / 3600)
            elif Pi < 0 and not stopped: # Regenerative braking
                charge_power = min(-Pi, max_charge)
                self.regenerated_power += charge_power * self.dt / 3600 - (self.utilities_power * self.dt / 3600)
                current_capacity += charge_power * self.dt / 3600 - (self.utilities_power * self.dt / 3600)

            # Clamp capacity
            current_capacity = max(0.0, min(current_capacity, capacity))
            self.battery_charge[i] = current_capacity

            # --- Station handling ---
            if station_idx < len(self.station_stops):
                dist_to_station = self.station_stops[station_idx] - self.x[i]

                if abs(dist_to_station) <= 50.0 and self.v[i] < 0.1:
                    stopped = True
                    # Debug info
                    #if i % 100 == 0:
                    #    print(f"At x={self.x[i]:.1f} m, dist to station {station_idx+1}: {dist_to_station:.1f} m, speed: {self.v[i]:.2f} m/s, current capacity: {current_capacity:.2f} kWh")

                    if self.electrified_stations[station_idx]:
                        required_energy = capacity - current_capacity  # kWh
                        max_station_power = required_energy / self.stop_time[station_idx] * 3600
                        charge_power = min(max_charge, max_station_power)
                        #print(f"Station {station_idx+1}: Charging with {charge_power:.1f} kW for {self.stop_time[station_idx]:.1f} s")

                        current_capacity += charge_power * self.dt / 3600
                        current_capacity = min(current_capacity, capacity)
                    else:
                        # No charging at this station
                        current_capacity -= self.utilities_power * self.dt / 3600
                        current_capacity = max(0.0, current_capacity)

                # Detect departure
                if stopped and self.v[i] > 0.1:
                    stopped = False
                    station_idx += 1


    def run_simulation(self):
        # 1. Run the kinematic simulation
        self.t, self.x, self.v = self.simulate_speed_profile()

        # 2. Allocate dependent arrays
        self.initialize_variables()

        # 3. Energy model
        self.simulate_energy()

        # 4. Electrified time tracking
        self.electrified_time()

        # 5. Battery sizing logic
        charging_time = self.total_time_electrified / 3600  # hours
        self.capacity = self.total_energy_regen / charging_time / self.charge_rate  # kWh
        print(f"Charging time (h): {charging_time:.3f}")
        print(f"Estimated required battery capacity: {self.capacity:.3f} kWh")

        # 6. Battery charge simulation
        self.simulate_battery_capacity(
            capacity=self.capacity,
            starting_charge=self.capacity * 0.5,
            power=self.regen_power
        )

        self.utilities_power = (self.total_energy_regen / (1 - self.utilities_percentage) / (self.t[-1] / 3600)) - (self.total_energy_regen / (self.t[-1] / 3600))
    
    def optimize_battery_capacity(self, target_final_charge=0.2, tol=0.01, max_iter=20, step_size=0.05, round_trips=10, debug=True):
        self.run_simulation()

        test_capacity = self.capacity * 1.7
        prev_capacity = 0.0

        for iteration in range(max_iter):
            self.simulate_battery_capacity(
                capacity=test_capacity,
                starting_charge=test_capacity * 0.5,
                power=self.regen_power
            )

            start_charge = self.battery_charge[0]
            end_charge = self.battery_charge[-1]
            #print(f"Start charge: {start_charge:.2f} kWh, End charge: {end_charge:.2f} kWh")
            charge_diff = start_charge - end_charge
            total_charge_needed = round_trips * charge_diff
            #print(f"Total charge needed for {round_trips} round trips: {total_charge_needed:.2f} kWh")

            max_charge = np.max(self.battery_charge)
            min_charge = np.min(self.battery_charge)
            charge_range = max_charge - min_charge
            #print(f"Charge range during simulation: {charge_range:.2f} kWh")

            final_charge = (test_capacity - total_charge_needed - charge_range) / test_capacity
            if debug:
                print(f"Iteration {iteration+1}: Test capacity = {test_capacity:.2f} kWh, Final charge = {final_charge:.4f}")

            if abs(final_charge - target_final_charge) < tol:
                if debug:
                    print(f"Optimal battery capacity found: {test_capacity:.2f} kWh")
                return test_capacity

            if final_charge > target_final_charge:
                prev_capacity = test_capacity
                test_capacity -=  test_capacity * (final_charge - target_final_charge) * step_size
            else:
                return prev_capacity
        if debug:
            print(f"Max iterations reached. Estimated optimal battery capacity: {test_capacity:.2f} kWh")
        return test_capacity