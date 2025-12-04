# Stock Price Acceleration Analysis Dashboard

A comprehensive stock analysis dashboard that analyzes the "Physics of Price" by calculating the Second Derivative of price action, inspired by Druckenmiller's approach to momentum analysis.

## Features

- **Real-time Stock Data**: Fetch daily stock data using yfinance
- **Velocity Analysis**: Calculate Rate of Change (ROC) as the first derivative
- **Acceleration Analysis**: Calculate the second derivative (momentum of momentum)
- **Interactive Visualizations**: Three stacked charts showing Price, Velocity, and Acceleration
- **Inflection Point Detection**: Automatically identify potential tops and bottoms
- **Customizable Parameters**: Adjust ROC period, EMA smoothing, and SMA overlay

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The dashboard will open in your default web browser.

## How It Works

1. **Position**: The closing price of the stock
2. **Velocity (1st Derivative)**: Rate of Change (ROC) calculated over n periods
3. **Smoothing**: Exponential Moving Average (EMA) applied to Velocity to reduce noise
4. **Acceleration (2nd Derivative)**: The change in Smoothed Velocity, representing momentum of momentum

## Dashboard Components

- **Top Chart**: Candlestick chart with SMA overlay and inflection point markers
- **Middle Chart**: Velocity (ROC) line chart with zero reference line
- **Bottom Chart**: Acceleration histogram (green for positive, red for negative)

## Parameters

- **ROC Period**: Number of periods for Rate of Change calculation (default: 12)
- **EMA Smoothing**: Period for smoothing the Velocity (default: 3)
- **SMA Period**: Period for Simple Moving Average overlay (default: 20)

## Requirements

- Python 3.8+
- streamlit
- yfinance
- pandas
- numpy
- plotly
