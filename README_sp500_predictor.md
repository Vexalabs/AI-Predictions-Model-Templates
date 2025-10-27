# S&P 500 Daily OHLC Predictor

This project provides two implementations for predicting S&P 500 daily OHLC (Open, High, Low, Close) values using Facebook Prophet:

1. **Interactive Brokers Version** (`sp500_daily_predictor.py`) - Uses IB API
2. **Standalone Version** (`sp500_daily_predictor_standalone.py`) - Uses yfinance

## Features

- **Daily OHLC Predictions**: Predicts Open, High, Low, Close, and Volume for the next 7 days
- **Technical Indicators**: Uses SMA, EMA, MACD, RSI, Bollinger Bands, and Volume ratios
- **Prophet Integration**: Leverages Facebook Prophet for time series forecasting
- **JSON Output**: Saves predictions in structured JSON format
- **Comprehensive Logging**: Detailed logging for monitoring and debugging
- **Caching**: Intelligent caching to avoid redundant API calls

## Prerequisites

### For Interactive Brokers Version
- Interactive Brokers TWS or IB Gateway running
- Python 3.8+
- Required packages (see `requirements_daily_predictor.txt`)

### For Standalone Version
- Python 3.8+
- Internet connection for data fetching
- Required packages (see `requirements_standalone.txt`)

## Installation

### Option 1: Interactive Brokers Version
```bash
pip install -r requirements_daily_predictor.txt
```

### Option 2: Standalone Version (Recommended)
```bash
pip install -r requirements_standalone.txt
```

## Usage

### Standalone Version (Recommended)

The standalone version is easier to use and doesn't require Interactive Brokers:

```bash
python sp500_daily_predictor_standalone.py
```

This will:
1. Fetch the last 252 trading days of S&P 500 data using yfinance
2. Train Prophet models for each OHLC metric
3. Generate predictions for the next 7 trading days
4. Save results to `sp500_predictions/sp500_daily_predictions_YYYYMMDD_HHMMSS.json`

### Interactive Brokers Version

If you have Interactive Brokers TWS/Gateway running:

```bash
python sp500_daily_predictor.py
```

## Output Format

The predictions are saved in JSON format matching the AI Predictions API structure:

```json
{
  "prediction_timestamp": "2025-01-27T10:30:45.123456Z",
  "predictions": [
    {
      "name": "S&P 500 Daily Prophet Model",
      "description": "Daily OHLC prediction using Facebook Prophet with technical indicators",
      "horizon": 7,
      "frequency": 1,
      "stock_name": "S&P 500",
      "forecasts": [
        {
          "timestamp": "2025-01-28T00:00:00",
          "forecast_index": 1,
          "open": 4850.25,
          "high": 4875.50,
          "low": 4840.75,
          "close": 4865.30,
          "volume": 45000000,
          "pct_change": 0.15,
          "direction": "UP"
        }
      ]
    }
  ],
  "metadata": {
    "model_type": "Prophet",
    "training_days": 252,
    "last_training_date": "2025-01-24",
    "prediction_horizon": 7,
    "symbol": "SPY",
    "data_source": "yfinance"
  }
}
```

## Model Configuration

### Prophet Settings

The model is configured specifically for daily financial data:

- **Daily Seasonality**: Disabled (not relevant for daily data)
- **Weekly Seasonality**: Enabled (weekday effects)
- **Yearly Seasonality**: Enabled (annual patterns)
- **Seasonality Mode**: Multiplicative (better for financial data)
- **Changepoint Prior Scale**: 0.05 (allows trend changes)

### Technical Indicators Used

- **Moving Averages**: SMA 20, SMA 50, SMA 200
- **Exponential Moving Averages**: EMA 12, EMA 26
- **MACD**: MACD line, Signal line, Histogram
- **RSI**: 14-period RSI
- **Bollinger Bands**: 20-period with 2 standard deviations
- **Volume Indicators**: Volume SMA, Volume ratio
- **Volatility**: 20-period rolling standard deviation

### Prediction Bounds

Predictions are bounded based on historical volatility:
- Maximum change per day: 10% or 3x historical volatility
- Bounds increase slightly for longer prediction horizons
- Prevents unrealistic predictions

## File Structure

```
├── sp500_daily_predictor.py              # IB API version
├── sp500_daily_predictor_standalone.py  # Standalone version (recommended)
├── requirements_daily_predictor.txt     # Dependencies for IB version
├── requirements_standalone.txt          # Dependencies for standalone version
├── README.md                            # This file
└── sp500_predictions/                   # Output directory (created automatically)
    └── sp500_daily_predictions_*.json   # Prediction files
```

## Customization

### Changing Prediction Horizon

Modify the `prediction_horizon` parameter in the class initialization:

```python
predictor = SP500DailyPredictorStandalone()
predictor.prediction_horizon = 14  # Predict next 14 days
```

### Changing Training Period

Modify the `training_days` parameter:

```python
predictor.training_days = 500  # Use 2 years of data
```

### Adding Custom Regressors

You can add additional regressors to the Prophet model by modifying the `prepare_prophet_data` method:

```python
prophet_df['custom_indicator'] = df['custom_indicator']
```

And adding it to the model:

```python
model.add_regressor('custom_indicator')
```

## Troubleshooting

### Common Issues

1. **No data received**: Check internet connection (standalone) or IB connection (IB version)
2. **Insufficient data**: Ensure you have at least 50 trading days of data
3. **Prophet errors**: Check that all regressors have valid values (no NaN)

### Logging

Both versions create detailed logs in `sp500_daily_predictions.log`. Check this file for debugging information.

### Performance

- **Training time**: ~30-60 seconds for 252 days of data
- **Memory usage**: ~100-200MB during training
- **Cache duration**: 1 hour (configurable)

## Integration with AI Predictions Platform

This script is designed to be compatible with the AI Predictions Model Templates structure. To integrate:

1. Place the script in the `src/` directory of your AI Predictions project
2. Modify `main.py` to call the prediction function
3. Ensure the JSON output format matches the expected API response

## License

This project follows the same license as the AI Predictions Model Templates project.

## Support

For issues or questions:
- Check the logs in `sp500_daily_predictions.log`
- Review the troubleshooting section above
- Contact: devteam@aipredictions.ai
