# S&P 500 Stock Prediction API

Welcome to the S&P 500 Stock Prediction project! This project provides AI-powered stock price predictions using Facebook Prophet with technical indicators.

## Prerequisites

Before getting started, make sure your system meets these requirements:

### Required Software
- **Python**: Version 3.10 or higher - [Download Python](https://www.python.org/downloads/)
- **Visual Studio Code**: Latest version - [Download VS Code](https://code.visualstudio.com/)
- **Docker Desktop**: See requirements below

### Docker Desktop Requirements

**Windows:**
- Windows 10/11 64-bit: Pro, Enterprise, or Education (Build 16299 or later)
- WSL 2 feature enabled
- 4GB system RAM minimum
- BIOS-level hardware virtualization enabled
- [Download Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)

**macOS:**
- macOS version 11 or newer (Intel or Apple Silicon)
- At least 4GB of RAM
- [Download Docker Desktop for Mac](https://www.docker.com/products/docker-desktop/)

**Linux:**
- 64-bit kernel and CPU support for virtualization
- systemd init system
- At least 4GB of RAM
- [Download Docker Desktop for Linux](https://www.docker.com/products/docker-desktop/)

## Fire Up Your Model with Docker!

Time to unleash your model! In your terminal, run these commands:

1. **Change directory into the stocks template:**
   ```bash
   cd Stocks
   ```

2. **Build the magic image:**
   ```bash
   docker build -t stocks-predictor .
   ```

3. **Run your new image:**
   ```bash
   docker run -p 8000:8000 stocks-predictor
   ```

Keep this terminal running! It's busy making predictions!

## Is It Working? Let's Find Out!

Open your favorite web browser and navigate to: http://localhost:8000/health

If all is well, you'll see a happy message like this:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "last_trained_at": "2025-01-25 09:33:58"
}
```

You can also see the docs at: http://localhost:8000/docs

## Now, let's make some predictions!

In Visual Studio Code, go to the top right of your terminal, click the + dropdown, and select Git Bash. This will open a second terminal.

In this new terminal, paste and run the following command:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
         "name": "AAPL",
         "date": "2025-04-10",
         "current_price": 202.38
       }'
```

Get ready for your first prediction! How exciting is that?!

The response should be formatted as follows:

```json
{
  "prediction_timestamp": "2025-08-12T07:41:42.236533Z",
  "predictions": [
    {
      "name": "S&P 500 Prophet Model",
      "description": "Stock price forecast with 7-day horizon using Facebook Prophet",
      "horizon": 7,
      "frequency": 1,
      "stock_name": "AAPL",
      "forecasts": [
        {
          "timestamp": "2025-08-11T00:00:00",
          "forecast_index": 1,
          "price": 637.01,
          "pct_change": -0.03,
          "direction": "DOWN"
        },
        {
          "timestamp": "2025-08-12T00:00:00",
          "forecast_index": 2,
          "price": 628.82,
          "pct_change": -1.31,
          "direction": "DOWN"
        },
        {
          "timestamp": "2025-08-13T00:00:00",
          "forecast_index": 3,
          "price": 640.50,
          "pct_change": 0.52,
          "direction": "UP"
        },
        {
          "timestamp": "2025-08-14T00:00:00",
          "forecast_index": 4,
          "price": 638.11,
          "pct_change": 0.15,
          "direction": "UP"
        },
        {
          "timestamp": "2025-08-15T00:00:00",
          "forecast_index": 5,
          "price": 640.84,
          "pct_change": 0.57,
          "direction": "UP"
        },
        {
          "timestamp": "2025-08-18T00:00:00",
          "forecast_index": 6,
          "price": 639.42,
          "pct_change": 0.35,
          "direction": "UP"
        },
        {
          "timestamp": "2025-08-19T00:00:00",
          "forecast_index": 7,
          "price": 642.48,
          "pct_change": 0.83,
          "direction": "UP"
        }
      ]
    }
  ]
}
```

## Craft Your Own Prediction Masterpiece!

This is where the real fun begins! You get to customize how your model thinks.

Open `src/models.py`. This is your canvas! Update the logic in the `predict` method to reflect your brilliant prediction strategy.

Made changes? Awesome! Here's how to see them in action:

1. Go back to your first terminal (where the model is running) and press `Ctrl + c` to stop it.
2. Simply repeat the "Fire Up" and "Is It Working" steps above. Easy peasy!

Congratulations, you are now a bona fide model builder!

## Peek Under the Hood: Project Files Explained

Curious about what makes the Stocks model tick? Here's a quick tour:

- **main.py**: This is the heart of your project, the starting point for your API and where your model gets called.
- **src/api.py**: Defines how your model talks to the world (the API structure).
- **src/models.py**: This is where YOUR custom prediction logic lives! Get creative here.
- **src/schemas.py**: Lays out the data structures the API expects and provides.

And the rest:

- **.dockerignore**: Tells Docker what files to skip. You probably won't need to touch this often.
- **.gitignore**: Tells Git what files to ignore. No need to edit unless you add new files you don't want tracked.
- **LICENSE**: The project's license.
- **README.md**: What you're reading right now!
- **requirements.txt**: Lets Docker know which Python libraries your project needs. Important! If you add new Python libraries to your custom logic, remember to add them here!

## Features

- **S&P 500 Stock Predictions**: Predict prices for any S&P 500 stock
- **Facebook Prophet**: Advanced time series forecasting
- **Technical Indicators**: SMA, EMA, MACD, RSI, Bollinger Bands, Volume analysis
- **7-Day Horizon**: Predict up to 7 days ahead
- **Real-time Data**: Uses yfinance for live market data
- **RESTful API**: Clean, documented API endpoints
- **Docker Ready**: Easy deployment and scaling
- **Health Monitoring**: Built-in health checks

## API Endpoints

- `GET /health` - Health check and model status
- `POST /predict` - Generate stock price predictions
- `GET /symbols` - Get available S&P 500 symbols
- `GET /docs` - Interactive API documentation

## Model Details

- **Algorithm**: Facebook Prophet with technical indicators
- **Training Data**: 1 year of daily OHLCV data
- **Prediction Horizon**: 7 days
- **Data Source**: yfinance API
- **Technical Indicators**: SMA, EMA, MACD, RSI, Bollinger Bands, Volume ratios, Volatility

## Development

To run locally without Docker:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API
python main.py
```

The API will be available at http://localhost:8000

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
