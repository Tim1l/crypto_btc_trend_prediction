# 📊 Crypto BTC Trend Prediction

![Python](https://img.shields.io/badge/python-3.10.9-blue.svg) ![Status](https://img.shields.io/badge/status-active-brightgreen.svg)

This project predicts the trend of Bitcoin (BTCUSDT) prices across multiple timeframes (1 week, 1 day, 4 hours) using a LSTM and TCN models with additional Attention layer. It generates predictions, saves them to predictions.csv, and sends updates to a Telegram channel. The project includes scripts to fetch market data, preprocess it, and make predictions.

## 📋 Prerequisites

- Python 3.10.9: Required for compatibility with dependencies.
- Git: To clone the repository.
- Git LFS: For handling large files (model and data CSVs).
- Windows/Linux/Mac: Instructions are Windows-focused but adaptable.
- Telegram Bot: Optional, for receiving prediction updates in your channel.

## ⚙️ Installation

Follow these steps exactly to set up and run the project:

### Step 1: Install Python 3.10.9

- Download and install Python 3.10.9 from [python.org](https://python.org).
- During installation, check "Add Python 3.10 to PATH".
- Verify installation:

```bash
py -3.10 --version
```

Expected output: `Python 3.10.9`.

### Step 2: Clone the Repository

- Open a terminal (Command Prompt or PowerShell on Windows).
- Clone the repository:

```bash
git clone https://github.com/Tim1l/crypto_btc_trend_prediction.git
cd crypto_btc_trend_prediction
```

- If large files (e.g., `tcn_lstm_all_tf_best.keras`, `data/*.csv`) are used, install Git LFS:

```bash
git lfs install
```

- Verify files:

```bash
dir
dir data
```

Expected files:

- get_prediction.py
- get_quotations.py
- process_data.py
- requirements.txt
- tcn_lstm_all_tf_best.keras
- predictions.csv
- data\BTCUSDT_1min.csv
- data\BTCUSDT_4H.csv
- data\BTCUSDT_D.csv
- data\BTCUSDT_W.csv

### Step 3: Set Up Virtual Environment

- Create a virtual environment with Python 3.10:

```bash
py -3.10 -m venv venv
.\venv\Scripts\activate
```

- You should see `(venv)` in the terminal.
- Verify Python version:

```bash
python --version
```

Expected output: `Python 3.10.9`.

- Update pip:

```bash
python -m pip install --upgrade pip
```

### Step 4: Install Dependencies

- Install all required packages:

```bash
pip install -r requirements.txt
```

### Step 5: Configure Telegram (Optional)

To receive prediction updates in a Telegram channel:

- Open `get_prediction.py` in a text editor (e.g., VS Code).
- Replace the following lines with your own Telegram bot token and channel ID:

```python
TELEGRAM_TOKEN = "your_bot_token"
TELEGRAM_CHANNEL = "your_channel_id"
```

- If you don't want Telegram notifications, comment out the `send_telegram_update` call at the end of `get_prediction.py`:

```python
#send_telegram_update(added_rows)
```

## 🚀 Usage

Run the scripts in this order to update data and generate predictions, or use `pipeline.py` to execute all steps in one go.

### Step 1: Update Market Data

- Run `get_quotations.py` to fetch the latest BTCUSDT market data:

```bash
python get_quotations.py
```

- This updates `data/BTCUSDT_*.csv` files with the latest prices.

### Step 2: Preprocess Data

- Run `process_data.py` to preprocess the data for the model:

```bash
python process_data.py
```

- This prepares the data in `data/` for predictions.

### Step 3: Generate Predictions

- Run `get_prediction.py` to generate trend predictions and save them to `predictions.csv`:

```bash
python get_prediction.py
```

- If Telegram is configured, updates will be sent to your channel.
- Check `predictions.csv` for results.

### Alternatively: Run Pipeline

- Run `pipeline.py` to execute all steps (update data, preprocess, generate predictions) in one go, with logs saved to `pipeline_log.txt`:

```bash
python pipeline.py
```

- Check `pipeline_log.txt` for execution details and `predictions.csv` for results.

## 📈 Example Prediction (Telegram Output)

> 📈 BTCUSDT Update 🚀

- **1day**:
  - Previous: 2025-07-13 - Forecast ⬆️, Result ⬆️ ✅, Range 1.96%
  - Next: 2025-07-14 - Forecast ⬇️ ⏳
- **1week**:
  - Previous: 2025-07-07 - Forecast ⬆️, Result ⬆️ ✅, Range 11.09%
  - Next: 2025-07-14 - Forecast ⬆️ ⏳
- **4hour**:
  - Previous: 2025-07-13 08:00 - Forecast ⬇️, Result ⬆️ ❌, Range 0.34%
  - 2025-07-13 20:00 - Forecast ⬆️, Result ⬆️ ✅, Range 0.99%
  - Next: 2025-07-14 00:00 - Forecast ⬇️ ⏳

## 🛠️ Troubleshooting

- **Python version error**: Ensure `python --version` shows `Python 3.10.9` in the virtual environment.
- **ModuleNotFoundError: No module named 'tcn'**: Verify `keras-tcn==3.5.6` is installed (`pip list`) and the import in `get_prediction.py` is `from tcn import TCN`.
- **Installation fails**: Clear pip cache (`pip cache purge`) and retry `pip install -r requirements.txt`.
- **Logs**: Save errors to a file:

```bash
python get_prediction.py > log.txt 2>&1
```

## ℹ️ Notes

- **🔥 Performance**: Our neural network (LSTM + TCN with Attention layer) crushes it, predicting Bitcoin price trends with a stellar ~66.5% average accuracy across multiple timeframes! 🚀 Check out the results:

| Timeframe | Accuracy  |
|-----------|-----------|
| Weekly    | ~60.4%    |
| Daily     | ~65.7%    |
| 4-Hour    | ~66.7%    |

Based on historical data. Use at your own risk; results are not guaranteed.

- **Disclaimer**: This tool is for research purposes only. It is not financial advice. Trading involves significant risk; consult a financial advisor before making decisions.