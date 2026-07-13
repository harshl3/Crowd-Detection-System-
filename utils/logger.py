import os
import csv
import pandas as pd
import time
from datetime import datetime

# Global dictionary to throttle logging per session/runner context
_last_log_times = {}

def log_crowd_data(count, threshold, density_level, overcrowded, grid_data=None, log_dir="logs", file_name="crowd_log.csv", throttle_seconds=2):
    """
    Logs current crowd stats into a CSV file. Throttled to prevent file bloat on frame-by-frame updates.
    """
    global _last_log_times
    current_time = time.time()
    
    # Check if we should skip writing based on throttle threshold
    key = f"{log_dir}/{file_name}"
    if key in _last_log_times:
        if current_time - _last_log_times[key] < throttle_seconds:
            return False  # Throttled, did not write
            
    _last_log_times[key] = current_time
    
    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, file_name)
    file_exists = os.path.exists(file_path)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Formulate grid representation if present
    grid_str = ""
    if grid_data:
        # Convert dictionary to string like "cell1:3,cell2:1..."
        grid_str = ",".join([f"Zone {k}:{v}" for k, v in grid_data.items()])
        
    row = {
        "Timestamp": timestamp,
        "People Count": count,
        "Limit Threshold": threshold,
        "Density Level": density_level,
        "Overcrowded": 1 if overcrowded else 0,
        "Zone Distribution": grid_str
    }
    
    fieldnames = ["Timestamp", "People Count", "Limit Threshold", "Density Level", "Overcrowded", "Zone Distribution"]
    
    try:
        with open(file_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        return True
    except Exception as e:
        # Silently fail if file is locked or unavailable
        return False

def get_historical_logs(log_dir="logs", file_name="crowd_log.csv"):
    """
    Reads the logged CSV data and returns it as a pandas DataFrame.
    """
    file_path = os.path.join(log_dir, file_name)
    if not os.path.exists(file_path):
        return pd.DataFrame(columns=["Timestamp", "People Count", "Limit Threshold", "Density Level", "Overcrowded", "Zone Distribution"])
    
    try:
        df = pd.read_csv(file_path)
        # Parse timestamp
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        return df
    except Exception as e:
        return pd.DataFrame(columns=["Timestamp", "People Count", "Limit Threshold", "Density Level", "Overcrowded", "Zone Distribution"])

def compute_log_metrics(df):
    """
    Computes key dashboard metrics from historical logs.
    """
    if df.empty:
        return {
            "peak_count": 0,
            "average_count": 0.0,
            "total_alerts": 0,
            "high_density_pct": 0.0
        }
        
    peak = df["People Count"].max()
    avg = df["People Count"].mean()
    alerts = df["Overcrowded"].sum()
    
    # Calculate percentage of time spent in High density
    high_density_count = df[df["Density Level"] == "High"].shape[0]
    high_density_pct = (high_density_count / len(df)) * 100.0 if len(df) > 0 else 0.0
    
    return {
        "peak_count": int(peak) if not pd.isna(peak) else 0,
        "average_count": round(avg, 1) if not pd.isna(avg) else 0.0,
        "total_alerts": int(alerts) if not pd.isna(alerts) else 0,
        "high_density_pct": round(high_density_pct, 1)
    }

def clear_logs(log_dir="logs", file_name="crowd_log.csv"):
    """
    Deletes the log file.
    """
    file_path = os.path.join(log_dir, file_name)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            # Remove from throttle cache to allow immediate rewrite
            key = f"{log_dir}/{file_name}"
            if key in _last_log_times:
                del _last_log_times[key]
            return True
        except Exception as e:
            return False
    return False
