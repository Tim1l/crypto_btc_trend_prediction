import requests
import pandas as pd
import time
import os
from pathlib import Path
import datetime as dt

# Configuration
symbol = "BTC/USDT"
interval = "1"  # 1-minute interval
output_dir = "./data"
output_file = f"{output_dir}/BTCUSDT_1min.csv"
min_weeks = 85  # Minimum required weeks of data
min_minutes = min_weeks * 7 * 24 * 60  # ~787,680 minutes
limit = 1000  # Max candles per API request
api_url = "https://api.bybit.com/v5/market/kline"
save_every = 10000  # Save after collecting N candles

# Ensure output directory exists
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Load existing data
existing_df = None
if os.path.exists(output_file):
    try:
        existing_df = pd.read_csv(output_file, parse_dates=['DATETIME'])
        existing_df.set_index('DATETIME', inplace=True)
        existing_df = existing_df[~existing_df.index.duplicated(keep='last')]
        print(f"Existing data loaded: {existing_df.shape}")
        print(f"Data range: {existing_df.index.min()} to {existing_df.index.max()}")
    except Exception as e:
        print(f"Error loading existing data: {e}. Starting fresh.")
        existing_df = None
else:
    print("No existing data found. Starting fresh.")

# Determine start time
current_time = dt.datetime.now().replace(tzinfo=None)  # Naive datetime
min_start_time = current_time - dt.timedelta(weeks=min_weeks)
if existing_df is not None and not existing_df.empty:
    last_timestamp = existing_df.index.max()
    minutes_available = (last_timestamp - existing_df.index.min()).total_seconds() / 60
    print(f"Available data: {minutes_available/60/24:.2f} days ({minutes_available/60/24/7:.2f} weeks)")
    if minutes_available >= min_minutes and (current_time - last_timestamp).total_seconds() / 60 < 2:
        print("Sufficient data available and up-to-date. No fetch needed.")
        existing_df = existing_df.astype({'OPEN': float, 'HIGH': float, 'LOW': float, 'CLOSE': float})
        existing_df.to_csv(output_file)
        print(f"Data saved to {output_file}")
        print(f"Final dataset size: {existing_df.shape}")
        print(existing_df.tail())
        exit(0)
    else:
        print("Fetching new data from last timestamp.")
        start_time = int(last_timestamp.timestamp() * 1000) + 60_000  # Next minute
else:
    print(f"Fetching {min_weeks} weeks of data from {min_start_time}")
    start_time = int(min_start_time.timestamp() * 1000)

# Fetch new data
candles = []
total_new = 0
while True:
    params = {
        "category": "linear",
        "symbol": symbol.replace("/", ""),
        "interval": interval,
        "limit": limit
    }
    if start_time:
        params["start"] = start_time

    try:
        resp = requests.get(api_url, params=params)
        data = resp.json()
        if data["retCode"] != 0:
            print(f"API error: {data['retMsg']}")
            break
    except Exception as e:
        print(f"Request failed: {e}. Retrying in 5 seconds.")
        time.sleep(5)
        continue

    klines = data["result"]["list"]
    if not klines:
        print("No more new data available.")
        break

    # Sort and filter new candles
    klines = sorted(klines, key=lambda x: int(x[0]))
    klines = [k for k in klines if int(k[0]) >= start_time]
    if not klines:
        break

    candles.extend(klines)
    total_new += len(klines)
    print(f"Downloaded {total_new} new candles (buffer: {len(candles)})")

    # Save periodically or at end
    if len(candles) >= save_every or len(klines) < limit:
        df_new = pd.DataFrame(candles, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])
        df_new['DATETIME'] = pd.to_datetime(df_new['timestamp'].astype('int64'), unit='ms')
        df_new = df_new[['DATETIME', 'open', 'high', 'low', 'close']]
        df_new = df_new.rename(columns={
            'open': 'OPEN', 'high': 'HIGH', 'low': 'LOW', 'close': 'CLOSE'
        })
        df_new = df_new.astype({
            'OPEN': float, 'HIGH': float, 'LOW': float, 'CLOSE': float
        })
        df_new = df_new.set_index('DATETIME')
        df_new = df_new[~df_new.index.duplicated(keep='last')]

        # Merge with existing data
        if existing_df is not None and not existing_df.empty:
            combined_df = pd.concat([existing_df, df_new])
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            combined_df = combined_df.sort_index()
        else:
            combined_df = df_new

        # Trim to at least 78 weeks
        min_start = combined_df.index.max() - dt.timedelta(weeks=min_weeks)
        combined_df = combined_df[combined_df.index >= min_start]

        # Fill gaps (forward fill)
        combined_df = combined_df.asfreq('1min').ffill()

        # Save to CSV
        combined_df.to_csv(output_file)
        print(f"Saved {output_file}. Total rows: {len(combined_df)}")
        candles = []  # Clear buffer
        existing_df = combined_df  # Update existing_df for next iteration

    # Update start_time for next request
    start_time = int(klines[-1][0]) + 60_000
    time.sleep(1.1)  # Rate limit delay

# Final save if any candles remain
if candles:
    df_new = pd.DataFrame(candles, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "turnover"
    ])
    df_new['DATETIME'] = pd.to_datetime(df_new['timestamp'].astype('int64'), unit='ms')
    df_new = df_new[['DATETIME', 'open', 'high', 'low', 'close']]
    df_new = df_new.rename(columns={
        'open': 'OPEN', 'high': 'HIGH', 'low': 'LOW', 'close': 'CLOSE'
    })
    df_new = df_new.astype({
        'OPEN': float, 'HIGH': float, 'LOW': float, 'CLOSE': float
    })
    df_new = df_new.set_index('DATETIME')
    df_new = df_new[~df_new.index.duplicated(keep='last')]

    # Merge with existing data
    if existing_df is not None and not existing_df.empty:
        combined_df = pd.concat([existing_df, df_new])
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
        combined_df = combined_df.sort_index()
    else:
        combined_df = df_new

    # Trim to at least 78 weeks
    min_start = combined_df.index.max() - dt.timedelta(weeks=min_weeks)
    combined_df = combined_df[combined_df.index >= min_start]

    # Fill gaps (forward fill)
    combined_df = combined_df.asfreq('1min').ffill()

    # Check for large gaps
    gaps = combined_df.index.to_series().diff().dt.total_seconds() / 60
    large_gaps = gaps[gaps > 5]
    if not large_gaps.empty:
        print(f"Warning: Found {len(large_gaps)} gaps > 5 minutes. Largest gap: {large_gaps.max()} minutes")

    # Save to CSV
    combined_df.to_csv(output_file)
    print(f"Final save to {output_file}. Total rows: {len(combined_df)}")
    print(f"Data range: {combined_df.index.min()} to {combined_df.index.max()}")
    print(combined_df.tail())

print("Data fetching completed.")