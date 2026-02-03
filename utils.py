from functools import reduce
import os 
from glob import glob
import pandas as pd
import random
import numpy as np
import torch
from pyproj import Transformer
 

def parse_header(file_path):
    """Extract network, station, lat, lon from the first line of a dataset file."""
    with open(file_path, "r") as f:
        header_line = f.readline().strip()
    
    parts = header_line.split()
    network = parts[1]        # second field = network
    station = parts[2]        # third field = station
    lat = float(parts[3])     # latitude
    lon = float(parts[4])     # longitude
    return network, station, lat, lon

# Function to read a .stm file and return a dataframe with datetime + value
def read_stm(file_path):
    df = pd.read_csv(file_path, delim_whitespace=True, header=None, skiprows=1,
                     names=['date','time','value','sensor_code','flag'])
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y/%m/%d %H:%M')
    df = df[['datetime','value']]
    return df


# Function to create a dataset of the daily soil moisture stations records, including soil moisture, soil temerature, air temerature and precipitation

def create_dataset(station_folder):
        # Define variables and depth filtering
    variables = {
        'sm': '0.05',  # soil moisture 5cm
        'ts': '0.05',  # soil temperature 5cm
        'ta': None,        # air temperature
        'p': None    # precipitation
    }

    all_stations = []
    
    # Loop through each station folder
    for station_name in os.listdir(station_folder):
        station_path = os.path.join(station_folder, station_name)
        if not os.path.isdir(station_path):
            continue
    
        dfs_per_var = {}
    
        for var, depth in variables.items():
            if depth:
                files = glob(os.path.join(station_path, f"*_{var}_{depth}*.stm"))
            else:
                files = glob(os.path.join(station_path, f"*_{var}*.stm"))  # any depth
            if not files:
                continue
            
            
    
            # Read and merge replicate sensors
            dfs = [read_stm(f).rename(columns={'value': os.path.basename(f)}) for f in files]
            merged = reduce(lambda left,right: pd.merge(left, right, on='datetime', how='outer'), dfs)
            # Average replicate sensors
            merged[f'{var}_5cm' if var in ['sm','ts'] else var] = merged.iloc[:,1:].mean(axis=1)
            dfs_per_var[var] = merged[['datetime', f'{var}_5cm' if var in ['sm','ts'] else var]]
    
        if dfs_per_var:
            # Merge all variables for this station
            station_df = reduce(lambda left,right: pd.merge(left, right, on='datetime', how='outer'), dfs_per_var.values())
            station_df['station'] = station_name  # add station column
            all_stations.append(station_df)
    
    # Combine all stations
    final_df = pd.concat(all_stations, ignore_index=True)
    final_df = final_df.sort_values(['station','datetime']).reset_index(drop=True)
    final_df = final_df.dropna().reset_index(drop=True)
    final_df["datetime"] = final_df["datetime"].dt.date
    final_df["datetime"] = pd.to_datetime(final_df['datetime'])


    daily_df = (
        final_df.groupby(["datetime", "station"]).agg({
        "sm_5cm": "mean",
        "ts_5cm": "mean",
        "ta": "mean",
        "p": "sum"   # <-- use sum for precipitation
        }).reset_index()
    )
    
    return daily_df


def set_seed(seed: int = 123467):
    """
    Set random seeds for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def filter_stations(df):
    time_bounds = (
        df.groupby('station')['datetime']
        .agg(start_date='min', end_date='max')
        .reset_index()
    )
    # Count how many stations share each start-end pair
    pair_counts = (
        time_bounds.groupby(['start_date', 'end_date'])
        .size()
        .reset_index(name='count')
        .sort_values('count', ascending=False)
    )
    
    # Get the most common combination
    common_start, common_end = pair_counts.iloc[0][['start_date', 'end_date']]


    # Keep only stations with that same start & end
    valid_stations = time_bounds[
        (time_bounds['start_date'] == common_start) &
        (time_bounds['end_date'] == common_end)
    ]['station']

    
    df_filtered = df[df['station'].isin(valid_stations)].copy()
    return df_filtered

def latlon_to_meters2(lon, lat, epsg_out=3857):
    # convert lat/lon (EPSG:4326) to a metric CRS (default WebMercator EPSG:3857).
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_out}", always_xy=True)
    xs, ys = transformer.transform(lon, lat)
    return [xs, ys]

def compute_scores(df):
    """
    lower_q_cal, upper_q_cal, y_true : 1D numpy arrays (same length)
    returns: nonconformity scores: max(lower_q - y, y - upper_q), clipped >=0
    """
    lower_res = df["lower_quantile"].values - df["y_true"].values
    upper_res = df["y_true"].values - df["upper_quantile"].values
    scores = np.maximum(lower_res, upper_res)
    return scores


def apply_lcp(df_cal, df_test, alpha,
              bandwidth_space,
              bandwidth_time=None,
              time_window_days=None):
    """
    df_cal: calibration DataFrame with x,y,date,q05,q95,y_true
    df_test: test DataFrame with x,y,date,q05,q95
    """

    scores_cal = compute_scores(df_cal)  # shape (N_cal,)

    q_locals = []
    PI_low = []
    PI_high = []

    for idx, row in df_test.iterrows():
        test_x = row["x"]
        test_y = row["y"]
        test_date = row["time"]

        # ---- compute  weights ----
        mask = df_cal["time"] < test_date
        if time_window_days is not None:
            mask &= (test_date - df_qr_cal["time"]).dt.days <= time_window_days
            
        df_sub = df_cal.loc[mask]
        coords_cal = df_sub[["x", "y"]].values
        test_coord = np.array([test_x, test_y])
        d_space = np.linalg.norm(coords_cal - test_coord, axis=1)

        # spatio_tempo kernel
        if bandwidth_space is not None and bandwidth_time is not None:
            w_space = np.exp(-0.5 * (d_space / bandwidth_space) ** 2)
            dt = (test_date - df_sub["time"]).dt.days.values
            w_time = np.exp(-0.5 * (dt / bandwidth_time) ** 2)       
            w = w_space * w_time

        # spatial kernel
        elif bandwidth_space is not None:
            w_space = np.exp(-0.5 * (d_space / bandwidth_space) ** 2)
            w = w_space

        # temporal kernel
        elif bandwidth_time is not None:
            dt = (test_date - df_sub["time"]).dt.days.values
            w_time = np.exp(-0.5 * (dt / bandwidth_time) ** 2)
            w = w_time
        
        # here - above me
        if len(w) == 0:
            # no calibration points available
            q_adj = 0
        else:
            inds = df_sub.index.values
            V_sel = scores_cal[inds]
            
            df_sub["w"] = w
            df_sub["V"] = V_sel
            
            df_sub.reset_index(drop=True, inplace=True)
    
    
            trim_eps = 1e-12
            top_M = None #30
            
            keep = w > trim_eps
            if top_M is not None:
                # keep top_M by weight
                top_idx = np.argsort(w)[-top_M:]
                keep_idx = np.zeros_like(w, dtype=bool)
                keep_idx[top_idx] = True
                keep = keep & keep_idx

            inds = np.where(keep)[0]
            if len(inds) == 0:
                # fallback: keep the maximum weight
                inds = [int(np.argmax(w))]
            w_sel = w[inds].astype(float)
            V_sel = df_sub.loc[inds]["V"].values
            
            p = w_sel / w_sel.sum()
            # compute jump points of the weighted CDF centered at test:
            sorter = np.argsort(V_sel)
            p_sorted = p[sorter]
            cum = np.cumsum(p_sorted)
            
            # find smallest jump >= alpha
            jumps = cum
            mask = jumps >= alpha
            mask
            
            if mask.any():
                alpha_tilde = jumps[mask][0]
            
            sorter = np.argsort(V_sel)
            vals = V_sel[sorter]
            w = p[sorter]
            cumw = np.cumsum(w)
            
            q=1 - alpha_tilde
            cutoff = q * cumw[-1]
            idx = np.searchsorted(cumw, cutoff, side='right')
            #q_hat = vals[min(idx, len(vals)-1)]
            q_adj = vals[idx]



        # adjust intervals
        PI_low.append(row["lower_quantile"] - q_adj)
        PI_high.append(row["upper_quantile"] + q_adj)

        q_locals.append(q_adj)

    df_out = df_test.copy()
    df_out["q_local"] = q_locals
    df_out["PI_lower"] = PI_low
    df_out["PI_upper"] = PI_high

    
    df_out = df_out[df_out['q_local']!= 0]
    df_out['PI_lower'] = df_out['PI_lower'].apply(lambda x: 0 if x < 0 else x)


    return df_out
