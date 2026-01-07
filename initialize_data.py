import numpy as np
import pandas as pd


def load_route_data(filename):
    """
    Load route description Excel data and return the processed arrays.
    """

    # ---------- Load ----------
    station_list = pd.read_excel(f"Route_Description/{filename}.xlsx", sheet_name="station", index_col=0)
    speed_limit_df = pd.read_excel(f"Route_Description/{filename}.xlsx", sheet_name="speed_limit")
    electrified = pd.read_excel(f"Route_Description/{filename}.xlsx", sheet_name="electrified")

    # ---------- Unit conversion ----------
    station_list["Total distance"] *= 1000          # km → m
    station_list["Stop time"] *= 60                 # min → s
    speed_limit_df["Distance"] *= 1000              # km → m
    speed_limit_df["Total distance"] *= 1000        # km → m
    speed_limit_df["speed limit"] *= (1000 / 3600)  # km/h → m/s
    electrified["Start electrification"] *= 1000
    electrified["Stop electrification"] *= 1000

    # Normalize electrified flag
    station_list["Electrified"] = (
        station_list["Electrified"].str.strip().str.lower() == "yes"
    )

    # ---------- Extract forward route ----------
    route_length = station_list["Total distance"].iloc[-1]

    station_stops = station_list["Total distance"].values
    electrified_stations = station_list["Electrified"].values
    stop_time = station_list["Stop time"].values

    speed_limit_dist = speed_limit_df["Total distance"].values
    speed_limit_val = speed_limit_df["speed limit"].values

    start_pos = electrified["Start electrification"].values
    stop_pos = electrified["Stop electrification"].values

    # ---------- Build reverse route ----------

    # Stations
    reverse_station_stops = route_length - station_stops[:-1][::-1]
    station_stops = np.concatenate((station_stops, reverse_station_stops + route_length))

    # Speed limit distance
    reverse_speed_limit_dist = route_length - speed_limit_dist[:-1][::-1]
    speed_limit_dist = np.concatenate((speed_limit_dist, reverse_speed_limit_dist + route_length))
    speed_limit_dist = np.append(speed_limit_dist, route_length * 2)

    # Stop times
    stop_time = np.concatenate((stop_time, stop_time[:-1][::-1]))

    # Speed limit values
    speed_limit_val = np.concatenate((speed_limit_val, speed_limit_val[::-1]))

    # Electrification segments
    reverse_start_pos = route_length - stop_pos[::-1]
    reverse_stop_pos = route_length - start_pos[::-1]

    start_pos = np.concatenate((start_pos, reverse_start_pos + route_length))

    stop_pos = np.concatenate((stop_pos, reverse_stop_pos + route_length))

    # Station electrification
    reverse_electrified_station = electrified_stations[::-1]
    total_electrified_stations = np.concatenate(
        (electrified_stations, reverse_electrified_station[1:])
    )

    # ---------- Return ----------
    return dict(
        route_length=route_length,
        station_stops=station_stops,
        stop_time=stop_time,
        speed_limit_dist=speed_limit_dist,
        speed_limit_val=speed_limit_val,
        electrified_start=start_pos,
        electrified_stop=stop_pos,
        electrified_stations=total_electrified_stations,
        station_table=station_list,
        speed_table=speed_limit_df,
        electrified_table=electrified
    )