import time
import pandas as pd
import datetime as dt
import os
from pathlib import Path
import ccxt

# Configuration
symbol = 'BTC/USDT:USDT'
interval = '1m'  # 1-minute interval
output_dir = './data'
old_file = f'{output_dir}/BTCUSDT_1min.csv'
output_file = f'{output_dir}/BTCUSDT_1min.csv'
min_weeks = 128

# Ensure output directory exists
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Initialize Bybit exchange via ccxt
exchange = ccxt.bybit({
    'enableRateLimit': True,  # Respect API rate limits
})

# Load old data if exists
old_df = None
if os.path.exists(old_file):
    try:
        old_df = pd.read_csv(old_file, parse_dates=['DATETIME'])
        old_df.set_index('DATETIME', inplace=True)
        duplicates_old = old_df.index.duplicated().sum()
        if duplicates_old > 0:
            print(f"Found {duplicates_old} duplicate timestamps in old_df, keeping first occurrence")
            old_df = old_df[~old_df.index.duplicated(keep='first')]
        print(f"Old data loaded: {old_df.shape}")
        print(f"Data range: {old_df.index.min()} to {old_df.index.max()}")
    except Exception as e:
        print(f"Error loading old data: {e}. Starting fresh.")
        old_df = None
else:
    print("No old data found. Starting fresh.")

# Determine cutoff and filter old_df
if old_df is not None and not old_df.empty:
    cutoff_date = old_df.index.max()
    start_date = cutoff_date - pd.Timedelta(weeks=min_weeks)
    old_df = old_df[(old_df.index >= start_date) & (old_df.index <= cutoff_date)]
    old_df = old_df[['OPEN', 'HIGH', 'LOW', 'CLOSE']]
    old_df = old_df.astype({'OPEN': float, 'HIGH': float, 'LOW': float, 'CLOSE': float})
    start_timestamp = int((cutoff_date + pd.Timedelta(minutes=1)).timestamp() * 1000)
else:
    end_datetime = dt.datetime.now(dt.timezone.utc)
    start_datetime = end_datetime - pd.Timedelta(weeks=min_weeks)
    start_timestamp = int(start_datetime.timestamp() * 1000)
    cutoff_date = None  # Will be set after fetching

end_timestamp = int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000)

# Collect new data
df_list = []
since = start_timestamp
while since < end_timestamp:
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, interval, since, limit=1000)
        if not ohlcv:
            print("No data available for this time range")
            break

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df_list.append(df)

        since = int(df['timestamp'].iloc[-1].timestamp() * 1000) + 1
        print(f"Collecting data starting from {dt.datetime.fromtimestamp(since / 1000, tz=dt.timezone.utc)}")

        time.sleep(exchange.rateLimit / 1000)  # Pause to respect rate limits
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(5)  # Pause on error
        continue

# Combine new data
new_df = pd.DataFrame()
if df_list:
    new_df = pd.concat(df_list, ignore_index=True)
    new_df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
    new_df.sort_values('timestamp', inplace=True)
    new_df = new_df.rename(columns={
        'open': 'OPEN',
        'high': 'HIGH',
        'low': 'LOW',
        'close': 'CLOSE',
        'timestamp': 'DATETIME'
    })
    new_df = new_df.drop(columns=['volume'])
    new_df['DATETIME'] = pd.to_datetime(new_df['DATETIME']).dt.tz_localize(None)
    new_df = new_df.set_index('DATETIME')
    new_df = new_df.astype({'OPEN': float, 'HIGH': float, 'LOW': float, 'CLOSE': float})
    print(f"New data collected: {new_df.shape}")
    print(new_df.head())

# Combine old and new
if old_df is not None and not old_df.empty:
    combined_df = pd.concat([old_df, new_df], axis=0)
else:
    combined_df = new_df

# Ensure no duplicate timestamps
duplicates_combined = combined_df.index.duplicated().sum()
if duplicates_combined > 0:
    print(f"Found {duplicates_combined} duplicate timestamps in combined_df, keeping first occurrence")
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]

# Fill any gaps in minute-level data by forward fill
combined_df = combined_df.asfreq('1min').ffill()

# Check for large gaps
gaps = combined_df.index.to_series().diff().dt.total_seconds() / 60
large_gaps = gaps[gaps > 5]
if not large_gaps.empty:
    print(f"Warning: Found {len(large_gaps)} gaps > 5 minutes. Largest gap: {large_gaps.max()} minutes")

# Save combined data
combined_df.to_csv(output_file)
print(f"Combined data saved to {output_file}")
print(f"Dataset size: {combined_df.shape}")
print(combined_df.head(3))
print(combined_df.tail(3))

print("Data fetching completed.")