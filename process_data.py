import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, MACD, SMAIndicator, IchimokuIndicator, CCIIndicator, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator, UltimateOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from pathlib import Path
import datetime as dt

# Configuration
input_file = './data/BTCUSDT_1min.csv'
output_dir = './data'
min_weeks = 78  # Minimum weeks of data
epsilon = 1e-6  # Prevent division by zero
scale_factor = 10.0  # Normalization scaling

timeframes = {
    '4H': {'name': '4H', 'atr_window': 42, 'threshold': 0.000},
    'D': {'name': 'D', 'atr_window': 30, 'threshold': 0.00},
    'W': {'name': 'W', 'atr_window': 26, 'threshold': 0.0}
}

timeframe_intervals = {
    '4H': pd.Timedelta(hours=4),
    'D': pd.Timedelta(days=1),
    'W': pd.Timedelta(weeks=1)
}

# Ensure output directory exists
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Load minute-level data
train_data = pd.read_csv(input_file, parse_dates=['DATETIME'])
train_data.set_index('DATETIME', inplace=True)
train_data = train_data[~train_data.index.duplicated(keep='last')]
train_data = train_data.sort_index()

# Check minute-level data for NaN and gaps
print(f"Checking minute-level data ({train_data.shape[0]} rows, {train_data.shape[1]} columns):")
nan_counts = train_data.isna().sum()
if nan_counts.sum() == 0:
    print("  No NaN found")
else:
    print("  NaN found in the following columns:")
    for col, count in nan_counts[nan_counts > 0].items():
        print(f"    {col}: {count} NaN")

expected_interval = pd.Timedelta(minutes=1)
time_diffs = train_data.index.to_series().diff().dropna()
gaps = time_diffs[time_diffs > expected_interval]
if len(gaps) == 0:
    print("  No gaps in time series")
else:
    print(f"  Found {len(gaps)} gaps in time series:")
    for idx, diff in gaps[gaps > pd.Timedelta(minutes=5)].items():
        print(f"    Gap at {idx}: interval {diff} (expected {expected_interval})")

if nan_counts.sum() == 0 and len(gaps) == 0:
    print("  All good, no NaN or gaps")
else:
    print("  Issues detected, see details above")

# Forward-fill gaps
train_data = train_data.asfreq('1min').ffill()

# Define custom functions
def calculate_fractals(df, window=5):
    highs = df['HIGH'].rolling(window=window, center=True).apply(lambda x: 1 if x[2] == max(x) else 0, raw=True)
    lows = df['LOW'].rolling(window=window, center=True).apply(lambda x: 1 if x[2] == min(x) else 0, raw=True)
    return highs, lows

def calculate_heikin_ashi(df):
    ha = pd.DataFrame(index=df.index)
    ha['HA_CLOSE'] = (df['OPEN'] + df['HIGH'] + df['LOW'] + df['CLOSE']) / 4
    ha['HA_OPEN'] = (df['OPEN'].shift(1) + df['CLOSE'].shift(1)) / 2
    ha.loc[ha.index[0], 'HA_OPEN'] = df['OPEN'].iloc[0]
    ha['HA_HIGH'] = pd.concat([df['HIGH'], ha['HA_OPEN'], ha['HA_CLOSE']], axis=1).max(axis=1)
    ha['HA_LOW'] = pd.concat([df['LOW'], ha['HA_OPEN'], ha['HA_CLOSE']], axis=1).min(axis=1)
    return ha

# Process timeframes
dfs = {}
current_time = pd.Timestamp.now(tz=None)  # Current time, naive datetime
for tf, config in timeframes.items():
    tf_name = config['name']
    atr_window = config['atr_window']
    threshold = config['threshold']
    interval = timeframe_intervals[tf_name]

    # Resample data
    if tf_name == 'W':
        # Weekly: align to Monday start (00:00:00), label at start of interval
        df = train_data.resample('W-MON', label='left', closed='left').agg({
            'OPEN': 'first',
            'HIGH': 'max',
            'LOW': 'min',
            'CLOSE': 'last'
        }).dropna()
    else:
        # 4H and D: use standard resampling with start-of-interval timestamps
        df = train_data.resample(tf, label='left', closed='left').agg({
            'OPEN': 'first',
            'HIGH': 'max',
            'LOW': 'min',
            'CLOSE': 'last'
        }).dropna()

    # Determine the last closed candle
    last_data_time = train_data.index.max()  # Последняя минутная свеча
    if tf_name == 'W':
        last_closed = last_data_time - pd.Timedelta(days=last_data_time.weekday()) - pd.Timedelta(weeks=1)
    else:
        last_closed = last_data_time.floor(interval)
    df = df[df.index < last_closed]

    # # Trim to 78 weeks
    # min_start = df.index.max() - pd.Timedelta(weeks=85)
    # df = df[df.index >= min_start]

    # Calculate indicators
    df['RSI_7'] = RSIIndicator(df['CLOSE'], window=7).rsi()
    df['RSI_14'] = RSIIndicator(df['CLOSE'], window=14).rsi()
    df['RSI_21'] = RSIIndicator(df['CLOSE'], window=21).rsi()
    df['MACD'] = MACD(df['CLOSE']).macd()
    df['MACD_signal'] = MACD(df['CLOSE']).macd_signal()
    df['BB_high_10'] = BollingerBands(df['CLOSE'], window=10).bollinger_hband()
    df['BB_low_10'] = BollingerBands(df['CLOSE'], window=10).bollinger_lband()
    df['BB_high_20'] = BollingerBands(df['CLOSE'], window=20).bollinger_hband()
    df['BB_low_20'] = np.clip(BollingerBands(df['CLOSE'], window=20).bollinger_lband(), 0, None)
    df['BB_high_30'] = BollingerBands(df['CLOSE'], window=30).bollinger_hband()
    df['BB_low_30'] = np.clip(BollingerBands(df['CLOSE'], window=30).bollinger_lband(), 0, None)
    df['ATR_7'] = AverageTrueRange(df['HIGH'], df['LOW'], df['CLOSE'], window=7).average_true_range()
    df['ATR_14'] = AverageTrueRange(df['HIGH'], df['LOW'], df['CLOSE'], window=14).average_true_range()
    df['ATR_21'] = AverageTrueRange(df['HIGH'], df['LOW'], df['CLOSE'], window=21).average_true_range()
    df[f'ATR_{atr_window}'] = AverageTrueRange(df['HIGH'], df['LOW'], df['CLOSE'], window=atr_window).average_true_range()
    for period in [5, 10, 20, 50]:
        df[f'EMA_{period}'] = EMAIndicator(df['CLOSE'], window=period).ema_indicator()
    for period in [10, 20, 50]:
        df[f'SMA_{period}'] = SMAIndicator(df['CLOSE'], window=period).sma_indicator()
    df['Stoch_K'] = StochasticOscillator(df['HIGH'], df['LOW'], df['CLOSE']).stoch()
    df['Stoch_D'] = StochasticOscillator(df['HIGH'], df['LOW'], df['CLOSE']).stoch_signal()
    df['CCI_14'] = CCIIndicator(df['HIGH'], df['LOW'], df['CLOSE'], window=14).cci()
    df['CCI_20'] = CCIIndicator(df['HIGH'], df['LOW'], df['CLOSE'], window=20).cci()
    df['ADX_14'] = ADXIndicator(df['HIGH'], df['LOW'], df['CLOSE'], window=14).adx()
    df['ADX_20'] = ADXIndicator(df['HIGH'], df['LOW'], df['CLOSE'], window=20).adx()
    df['WilliamsR_14'] = WilliamsRIndicator(df['HIGH'], df['LOW'], df['CLOSE'], lbp=14).williams_r()
    df['ROC_10'] = ROCIndicator(df['CLOSE'], window=10).roc()
    df['ROC_20'] = ROCIndicator(df['CLOSE'], window=20).roc()
    df['UO'] = UltimateOscillator(df['HIGH'], df['LOW'], df['CLOSE']).ultimate_oscillator()
    ichimoku = IchimokuIndicator(df['HIGH'], df['LOW'])
    df['Ichimoku_Tenkan'] = ichimoku.ichimoku_conversion_line()
    df['Ichimoku_Kijun'] = ichimoku.ichimoku_base_line()
    df['Ichimoku_Senkou_A'] = ichimoku.ichimoku_a()
    df['Ichimoku_Senkou_B'] = ichimoku.ichimoku_b()
    df['Fractal_High'], df['Fractal_Low'] = calculate_fractals(df)
    ha = calculate_heikin_ashi(df)
    df[['HA_OPEN', 'HA_HIGH', 'HA_LOW', 'HA_CLOSE']] = ha[['HA_OPEN', 'HA_HIGH', 'HA_LOW', 'HA_CLOSE']]

    # Calculate Trend
    price_change = (df['CLOSE'] / df['OPEN'] - 1)
    df['Trend'] = np.where(price_change > threshold, 1, 0)

    # Check class distribution
    class_dist = df['Trend'].value_counts(normalize=True)
    print(f"{tf_name} class distribution: Down = {class_dist.get(0, 0):.2%}, Up = {class_dist.get(1, 0):.2%}")

    # Remove initial rows for NaNs
    max_period = max(atr_window, 50)
    df = df.iloc[max_period:]

    # Fill remaining NaNs with mean
    df = df.fillna(df.mean())

    dfs[tf_name] = df

# Define columns for normalization
price_columns = [
    'OPEN', 'HIGH', 'LOW', 'CLOSE',
    'EMA_5', 'EMA_10', 'EMA_20', 'EMA_50',
    'SMA_10', 'SMA_20', 'SMA_50',
    'BB_high_10', 'BB_low_10', 'BB_high_20', 'BB_low_20', 'BB_high_30', 'BB_low_30',
    'Ichimoku_Tenkan', 'Ichimoku_Kijun', 'Ichimoku_Senkou_A', 'Ichimoku_Senkou_B',
    'HA_OPEN', 'HA_HIGH', 'HA_LOW', 'HA_CLOSE'
]
range_0_100_columns = ['RSI_7', 'RSI_14', 'RSI_21', 'Stoch_K', 'Stoch_D', 'UO']
unbounded_columns = ['MACD', 'MACD_signal', 'CCI_14', 'CCI_20', 'ROC_10', 'ROC_20']
positive_columns = ['ATR_7', 'ATR_14', 'ATR_21', 'ADX_14', 'ADX_20']
binary_columns = ['Fractal_High', 'Fractal_Low']
atr_columns = {'4H': 'ATR_42', 'D': 'ATR_30', 'W': 'ATR_26'}
scales = {
    'MACD': 1500.0, 'MACD_signal': 1500.0, 'CCI_14': 500.0, 'CCI_20': 700.0,
    'ROC_10': 30.0, 'ROC_20': 35.0
}

# Normalize data
normalized_dfs = {}
for tf_name, df in dfs.items():
    df_normalized = df.copy()
    atr_col = atr_columns[tf_name]

    # Normalize price columns
    for col in price_columns:
        ratio = df_normalized[col] / np.maximum(df_normalized[atr_col], epsilon)
        df_normalized[f'{col}_norm'] = np.log1p(ratio) / scale_factor

    # Normalize 0-100 range indicators
    for col in range_0_100_columns:
        df_normalized[f'{col}_norm'] = np.clip(df_normalized[col], 0, 100) / 100.0

    # Normalize WilliamsR_14
    anomalies = df[df['WilliamsR_14'] < -100].shape[0]
    print(f"Anomalies in {tf_name} (WilliamsR_14 < -100): {anomalies} rows")
    df_normalized['WilliamsR_14_norm'] = np.clip(df_normalized['WilliamsR_14'], -100, 0)
    df_normalized['WilliamsR_14_norm'] = (df_normalized['WilliamsR_14_norm'] + 100) / 100.0

    # Normalize unbounded indicators
    for col in unbounded_columns:
        df_normalized[f'{col}_norm'] = np.tanh(df_normalized[col] / scales[col])

    # Normalize positive indicators
    for col in positive_columns + [atr_col]:
        df_normalized[f'{col}_norm'] = np.log1p(df_normalized[col]) / scale_factor

    # Copy binary columns
    for col in binary_columns:
        df_normalized[f'{col}_norm'] = df_normalized[col]

    # Normalize Trend
    df_normalized['Trend_norm'] = df_normalized['Trend']

    # Save normalized dataframe
    file_path = f'{output_dir}/BTCUSDT_{tf_name}.csv'
    df_normalized.to_csv(file_path)
    print(f"Saved {tf_name}: {df_normalized.shape} to {file_path}")
    normalized_dfs[tf_name] = df_normalized

# Check for NaNs and gaps in normalized dataframes
for tf, df in normalized_dfs.items():
    tf_name = timeframes[tf]['name']
    print(f"\nChecking {tf_name} ({df.shape[0]} rows, {df.shape[1]} columns):")
    nan_counts = df.isna().sum()
    if nan_counts.sum() == 0:
        print("  No NaN found")
    else:
        print("  NaN found in the following columns:")
        for col, count in nan_counts[nan_counts > 0].items():
            print(f"    {col}: {count} NaN")
    
    expected_interval = timeframe_intervals[tf_name]
    time_diffs = df.index.to_series().diff().dropna()
    gaps = time_diffs[time_diffs != expected_interval]
    if len(gaps) == 0:
        print("  No gaps in time series")
    else:
        print(f"  Found {len(gaps)} gaps in time series:")
        for idx, diff in gaps.items():
            print(f"    Gap at {idx}: interval {diff} (expected {expected_interval})")
    
    if nan_counts.sum() == 0 and len(gaps) == 0:
        print("  All good, no NaN or gaps")
    else:
        print("  Issues detected, see details above")

# Print summary
for tf_name, df in normalized_dfs.items():
    print(f"{tf_name}: {df.shape}, columns: {len(df.columns)}")
    if not df.empty:
        print(f"  Data range: {df.index.min()} to {df.index.max()}")