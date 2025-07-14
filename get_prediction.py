import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall
from tcn import TCN
import datetime as dt
from pathlib import Path
import requests

#add here your telegram token and channel or switch off this part if you don't need
TELEGRAM_TOKEN = ""
TELEGRAM_CHANNEL = ""
ASSET = 'BTCUSDT'
added_rows = []

# Define F1 Score metric
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = Precision(thresholds=0.5)
        self.recall = Recall(thresholds=0.5)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

# Configuration
model_path = './tcn_lstm_all_tf_best.keras'
data_dir = './data'
output_file = f'predictions.csv'
timeframes = {
    '1week': {'interval': pd.Timedelta(weeks=1), 'file': 'BTCUSDT_W.csv', 'threshold': 0.0},
    '1day': {'interval': pd.Timedelta(days=1), 'file': 'BTCUSDT_D.csv', 'threshold': 0.00},
    '4hour': {'interval': pd.Timedelta(hours=4), 'file': 'BTCUSDT_4H.csv', 'threshold': 0.000}
}
xLen = 28  # Window size for predictions
trend_labels = {0: '⬇️', 1: '⬆️'}

# Ensure output directory exists
Path(data_dir).mkdir(parents=True, exist_ok=True)

# Load model
model = tf.keras.models.load_model(model_path, custom_objects={'F1Score': F1Score}, compile=False)

def load_last_predictions():
    """Load last predicted candle timestamp and forecast for each timeframe from predictions.csv."""
    last_predicted = {tf: {'timestamp': None, 'forecast': None} for tf in timeframes}
    if not Path(output_file).exists():
        return last_predicted
    
    # Read only the last row for each timeframe
    pred_df = pd.read_csv(output_file, parse_dates=['start_date'])
    if pred_df.empty:
        return last_predicted
    
    for tf in timeframes:
        tf_predictions = pred_df[pred_df['timeframe'] == tf]
        if not tf_predictions.empty:
            last_row = tf_predictions.iloc[-1]
            last_predicted[tf] = {
                'timestamp': last_row['start_date'],
                'forecast': last_row['forecast']
            }
    
    return last_predicted

def calculate_range(row):
    """Calculate candle range as percentage."""
    return ((row['HIGH'] - row['LOW']) / row['OPEN'] * 100).round(2)

def format_prediction(tf_name, timestamp, forecast, result, ohlc, is_correct, is_future=False):
    """Format a prediction as a dictionary for CSV."""
    end_date = timestamp + timeframes[tf_name]['interval'] - pd.Timedelta(seconds=1)
    parent_date = None
    if tf_name == '4hour':
        parent_date = timestamp.floor('D')
    elif tf_name == '1day':
        parent_date = timestamp - pd.Timedelta(days=timestamp.weekday())
    
    if is_future:
        return {
            'timeframe': tf_name,
            'start_date': timestamp,
            'end_date': end_date,
            'parent_date': parent_date,
            'forecast': forecast,
            'result': '⏳',
            'is_correct': '',
            'open': np.nan,
            'high': np.nan,
            'low': np.nan,
            'close': np.nan,
            'range': np.nan
        }
    
    return {
        'timeframe': tf_name,
        'start_date': timestamp,
        'end_date': end_date,
        'parent_date': parent_date,
        'forecast': forecast,
        'result': result if result else '⏳',
        'is_correct': '✅' if is_correct else '❌' if is_correct is False else '',
        'open': ohlc['OPEN'],
        'high': ohlc['HIGH'],
        'low': ohlc['LOW'],
        'close': ohlc['CLOSE'],
        'range': calculate_range(ohlc)
    }

def make_prediction(df, timestamp, tf_name, threshold, is_future=False):
    """Make a prediction for a specific candle timestamp."""
    feature_columns = [col for col in df.columns if col.endswith('_norm') and col != 'Trend_norm']
    X = df[feature_columns].values
    
    try:
        idx = df.index.get_loc(timestamp)
    except KeyError:
        print(f"No data for {tf_name} at {timestamp}")
        return None
    
    if idx < xLen - 1:
        print(f"Skipping {tf_name} at {timestamp}: not enough prior data (idx={idx}, need {xLen-1})")
        return None
    
    window = X[idx-xLen+1:idx+1]  # Window of xLen candles including timestamp
    X_input = window[np.newaxis, ...]  # Shape: (1, xLen, n_features)
    
    pred = model.predict(X_input, verbose=0)
    forecast = trend_labels[int(pred[0][0] > 0.5)]
    
    # Get OHLC for the predicted candle
    next_candle_time = timestamp + timeframes[tf_name]['interval']
    if is_future:
        # For future prediction, no OHLC or result
        result = None
        is_correct = None
        ohlc = None
    else:
        next_candle = df[df.index == next_candle_time]
        if next_candle.empty:
            print(f"No data for {tf_name} at {next_candle_time} (next candle)")
            return None
        ohlc = next_candle.iloc[0][['OPEN', 'HIGH', 'LOW', 'CLOSE']]
        # Calculate actual trend
        result = None
        is_correct = None
        if 'Trend' in df.columns and next_candle_time in df.index:
            trend = df.loc[next_candle_time, 'Trend']
            result = trend_labels[trend]
            is_correct = (forecast == result)
    
    return format_prediction(tf_name, next_candle_time, forecast, result, ohlc, is_correct, is_future)

def update_predictions(df, last_predicted, tf_name, threshold):
    """Update predictions with results when candles close."""
    last_timestamp = last_predicted[tf_name]['timestamp']
    last_forecast = last_predicted[tf_name]['forecast']
    updated_predictions = []
    
    if last_timestamp is None or last_forecast is None or not Path(output_file).exists():
        return updated_predictions
    
    # Read predictions and filter pending results
    pred_df = pd.read_csv(output_file, parse_dates=['start_date'])
    pending = pred_df[(pred_df['timeframe'] == tf_name) & (pred_df['result'] == '⏳') & (pred_df['start_date'] == last_timestamp)]
    
    if not pending.empty and last_timestamp in df.index:
        ohlc = df.loc[last_timestamp][['OPEN', 'HIGH', 'LOW', 'CLOSE']]
        trend = df.loc[last_timestamp, 'Trend']
        result = trend_labels[trend]
        is_correct = (last_forecast == result)
        updated_row = format_prediction(tf_name, last_timestamp, last_forecast, result, ohlc, is_correct, is_future=False)
        updated_predictions.append(updated_row)
        added_rows.append(updated_row)
    
    return updated_predictions

# Main processing
print(f"Processing predictions at {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load last predicted timestamps
last_predicted = load_last_predictions()

# Load dataframes for each timeframe
dfs = {}
for tf_name, config in timeframes.items():
    df = pd.read_csv(f"{data_dir}/{config['file']}", parse_dates=['DATETIME'])
    df.set_index('DATETIME', inplace=True)
    dfs[tf_name] = df

# Read existing predictions if file exists
predictions = pd.DataFrame(columns=['timeframe', 'start_date', 'end_date', 'parent_date', 'forecast', 'result', 'is_correct', 'open', 'high', 'low', 'close', 'range'])
if Path(output_file).exists():
    predictions = pd.read_csv(output_file, parse_dates=['start_date', 'parent_date'])

# Update pending predictions for all timeframes
new_predictions = []
for tf_name, config in timeframes.items():
    new_preds = update_predictions(dfs[tf_name], last_predicted, tf_name, config['threshold'])
    new_predictions.extend(new_preds)
    if new_preds:
        predictions = predictions[~((predictions['timeframe'] == tf_name) & (predictions['start_date'] == last_predicted[tf_name]['timestamp']) & (predictions['result'] == '⏳'))]
        predictions = pd.concat([predictions, pd.DataFrame(new_preds)], ignore_index=True)

# Determine the last closed candle for each timeframe
for tf_name, config in timeframes.items():
    last_data_time = dfs[tf_name].index.max()
    if tf_name == '1week':
        last_closed = last_data_time - pd.Timedelta(days=last_data_time.weekday())
        hist_start = last_closed - pd.Timedelta(weeks=1)  # One week before for historical
    else:
        last_closed = last_data_time.floor(config['interval'])
        hist_start = last_closed - config['interval']  # One candle before for historical
    
    # Historical prediction for the last closed candle
    if len(dfs[tf_name]) >= xLen and last_closed in dfs[tf_name].index and not any((predictions['timeframe'] == tf_name) & (predictions['start_date'] == last_closed)):
        pred_line = make_prediction(dfs[tf_name], hist_start, tf_name, config['threshold'], is_future=False)
        if pred_line and predictions[(predictions['timeframe'] == tf_name) & (predictions['start_date'] == last_closed)].empty:
            predictions = pd.concat([predictions, pd.DataFrame([pred_line])], ignore_index=True)
            added_rows.append(pred_line)
    
    # Real-time prediction for the next candle
    next_candle_time = last_closed + config['interval']
    if len(dfs[tf_name]) >= xLen and last_closed in dfs[tf_name].index and not any((predictions['timeframe'] == tf_name) & (predictions['start_date'] == next_candle_time)):
        pred_line = make_prediction(dfs[tf_name], last_closed, tf_name, config['threshold'], is_future=True)
        if pred_line and predictions[(predictions['timeframe'] == tf_name) & (predictions['start_date'] == next_candle_time)].empty:
            predictions = pd.concat([predictions, pd.DataFrame([pred_line])], ignore_index=True)
            added_rows.append(pred_line)

# Save predictions
if not predictions.empty:
    timeframe_order = {'1week': 1, '1day': 2, '4hour': 3}
    predictions['tf_order'] = predictions['timeframe'].map(timeframe_order)
    predictions = predictions.sort_values(['start_date', 'tf_order']).drop(columns=['tf_order'])
    predictions.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Saved {len(predictions)} predictions to {output_file}")
else:
    print("No new predictions to save")


def send_telegram_update(added_rows):
    if not added_rows:
        return
    
    # Группировка по TF
    from collections import defaultdict  # Добавь импорт в начало, если нет
    grouped = defaultdict(lambda: {'prev': [], 'next': []})
    for row in added_rows:
        tf = row['timeframe']
        date_str = row['start_date'].strftime('%Y-%m-%d') if tf in ['1week', '1day'] else row['start_date'].strftime('%Y-%m-%d %H:%M')
        forecast = row['forecast']
        if row['result'] == '⏳':
            grouped[tf]['next'].append(f"{date_str} - Forecast {forecast} ⏳")
        else:
            result = row['result']
            correct = row['is_correct']
            range_val = f"{row['range']}%" if not pd.isna(row['range']) else ''
            grouped[tf]['prev'].append(f"{date_str} - Forecast {forecast}, Result {result} {correct}, Range {range_val}")
    
    # Строим сообщение
    message = f"📈 {ASSET} Update 🚀\n\n"
    grouped_by_date = defaultdict(list)
    for row in added_rows:
        grouped_by_date[row['start_date']].append(row)
    message = f"📈 {ASSET} Update 🚀\n\n"
    timeframe_order = {'1week': 1, '1day': 2, '4hour': 3}
    for start_date in sorted(grouped_by_date.keys()):
        rows = grouped_by_date[start_date]
        rows_sorted = sorted(rows, key=lambda x: timeframe_order[x['timeframe']])
        for row in rows_sorted:
            tf = row['timeframe']
            date_str = row['start_date'].strftime('%Y-%m-%d') if tf in ['1week', '1day'] else row['start_date'].strftime('%Y-%m-%d %H:%M')
            forecast = row['forecast']
            if row['result'] == '⏳':
                message += f"**{tf}:**\nNext:\n{date_str} - Forecast {forecast} ⏳\n\n"
            else:
                result = row['result']
                correct = row['is_correct']
                range_val = f"{row['range']}%" if not pd.isna(row['range']) else ''
                message += f"**{tf}:**\nPrevious:\n{date_str} - Forecast {forecast}, Result {result} {correct}, Range {range_val}\n\n"
    
    # Отправка
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': TELEGRAM_CHANNEL, 'text': message, 'parse_mode': 'Markdown'}
    try:
        response = requests.post(url, data=payload)
        if response.status_code != 200:
            print(f"Telegram send error: {response.text}")
    except Exception as e:
        print(f"Telegram exception: {e}")

# Вызов перед сохранением
send_telegram_update(added_rows)