import farcaster as fc
from dotenv import load_dotenv
import os
import pandas as pd
from datetime import datetime, timedelta
from openai import OpenAI
import requests
from PIL import Image

# Load variables from .xenv
load_dotenv('.xenv')
proxies = os.getenv("proxies")
os.environ['http_proxy'] = proxies
os.environ['https_proxy'] = proxies

# Get keys
dalle_api_key = os.getenv("dalle")
farc_mnemonic = os.getenv("farc_mnemonic")
imgur_client_id = os.getenv("imgur_client_id")  # Add your Imgur Client-ID to .xenv

# OpenAI client
openai_client = OpenAI(api_key=dalle_api_key)

# Paths
predictions_path = 'predictions.csv'  # Adjust if needed
state_path = 'state_farc.csv'  # In current folder

# Hardcoded probabilities
prob_map = {0: 66, 1: 72, 2: 78, 3: 81, 4: 100, 5: 100}  # After n fails

# Prompts for images
bull_prompts = {
    66: "Small uncertain cartoon bull in green colors (no text on the picture)",
    72: "Teenage bull looking more confident, cartoon style in green (no text on the picture)",
    78: "Adult confident bull standing strong, cartoon in green (no text on the picture)",
    81: "Very confident bull with determined expression, cartoon green (no text on the picture)",
    100: "Mega powerful muscular charging bull, epic cartoon in green (no text on the picture)"
}
bear_prompts = {
    66: "Small uncertain cartoon bear in red colors (no text on the picture)",
    72: "Teenage bear looking more confident, cartoon style in red (no text on the picture)",
    78: "Adult confident bear standing strong, cartoon in red (no text on the picture)",
    81: "Very confident bear with determined expression, cartoon red (no text on the picture)",
    100: "Mega powerful muscular charging bear, epic cartoon in red (no text on the picture)"
}

# Load predictions.csv (assuming it has columns: timeframe, start_date, forecast, result, etc.)
predictions_df = pd.read_csv(predictions_path, parse_dates=['start_date'])

# Filter for 1day timeframe
daily_df = predictions_df[predictions_df['timeframe'] == '1day']

# Get current forecast (last row with result == '⏳')
current_row = daily_df[daily_df['result'] == '⏳'].iloc[-1]
current_date = current_row['start_date']
current_forecast = current_row['forecast']

# Load or initialize state.csv
if os.path.exists(state_path):
    state_df = pd.read_csv(state_path, parse_dates=['post_date'])
else:
    state_df = pd.DataFrame(columns=['post_date', 'forecast', 'probability', 'post_id', 'actual_result'])
    state_df.to_csv(state_path, index=False)
    
state_df['actual_result'] = state_df['actual_result'].astype('object')

# Update actual_result for previous posts if needed
for idx, row in state_df.iterrows():
    if pd.isna(row['actual_result']):
        # Find matching row in predictions_df
        match = daily_df[daily_df['start_date'] == row['post_date']]
        if not match.empty:
            actual = match.iloc[0]['result']
            if actual != '⏳':
                is_correct = '✅' if actual == row['forecast'] else '❌'
                state_df.at[idx, 'actual_result'] = is_correct
state_df.to_csv(state_path, index=False)

# Calculate fail_streak from daily_df before current ⏳
fail_streak = 0
prev_rows = daily_df[daily_df['result'] != '⏳'].tail(5)  # Last 5 completed for max streak
for i in range(len(prev_rows) - 1, -1, -1):
    if prev_rows.iloc[i]['is_correct'] == '❌':
        fail_streak += 1
    else:
        break

# Get probability
prob = prob_map.get(fail_streak, 66)  # Default to 66 if >5
print(f"Fail streak: {fail_streak}")
print(f"Probability: {prob}%")

# Check if should post (daily, but check if already posted)
should_post = True
if current_date in state_df['post_date'].values:
    print("Already posted for this date")
    should_post = False

print(f"Should post: {should_post}")
if should_post:
    # Get last post info if exists
    last_str = ""
    if not state_df.empty:
        last_row = state_df.iloc[-1]
        last_date = last_row['post_date'].strftime('%Y-%m-%d')
        last_forecast = last_row['forecast']
        last_result = last_row['actual_result']
        last_str = f"Last: {last_date} Forecast {last_forecast} {last_result} | "

    # Format tweet
    current_date_str = current_date.strftime('%Y-%m-%d')
    tweet = f"{last_str} Forecast for Today: {current_date_str} daily candle for BTCUSDT (UTC time): {current_forecast} (Probability {prob}%) | Not financial advice, for entertainment purposes only"

    # Check for existing image from X, else generate
    fc_client = fc.Warpcast(mnemonic=farc_mnemonic)
    response = fc_client.post_cast(text=tweet)
    post_id = response.cast.hash

    # Add to state
    new_row = pd.DataFrame({
        'post_date': [current_date],
        'forecast': [current_forecast],
        'probability': [prob],
        'post_id': [post_id],
        'actual_result': [pd.NA]
    })
    state_df = pd.concat([state_df, new_row], ignore_index=True)
    state_df.to_csv(state_path, index=False)

    print(f"Cast posted: {tweet}")
else:
    print("No post today")