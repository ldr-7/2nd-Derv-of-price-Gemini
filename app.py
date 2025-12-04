import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Stock Price Acceleration Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark mode compatibility
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stSelectbox label, .stSlider label, .stDateInput label {
        color: #fafafa;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 Stock Price Acceleration Analysis Dashboard")
st.markdown("### Analyzing the 'Physics of Price' - Second Derivative Analysis")

# Sidebar for user inputs
st.sidebar.header("📊 Configuration")

# Ticker input
ticker = st.sidebar.text_input("Stock Ticker", value="NVDA", help="Enter the stock ticker symbol")

# Date range selection
default_start = datetime.now() - timedelta(days=365*2)  # Default to 2 years ago
start_date = st.sidebar.date_input(
    "Start Date",
    value=default_start,
    min_value=datetime(2000, 1, 1),
    max_value=datetime.now()
)

# Calculation parameters
st.sidebar.subheader("Calculation Parameters")

# ROC period (Velocity calculation)
roc_period = st.sidebar.slider(
    "Rate of Change Period (n)",
    min_value=1,
    max_value=50,
    value=12,
    help="Period for calculating the Rate of Change (Velocity)"
)

# EMA smoothing period
ema_period = st.sidebar.slider(
    "EMA Smoothing Period",
    min_value=1,
    max_value=20,
    value=3,
    help="Exponential Moving Average period for smoothing Velocity"
)

# SMA period for price overlay
sma_period = st.sidebar.slider(
    "SMA Period for Price Overlay",
    min_value=5,
    max_value=100,
    value=20,
    help="Simple Moving Average period for price chart overlay"
)

# Fetch data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_stock_data(ticker_symbol, start):
    """Fetch stock data using yfinance"""
    try:
        ticker_obj = yf.Ticker(ticker_symbol)
        df = ticker_obj.history(start=start, end=datetime.now())
        if df.empty:
            return None, f"No data found for ticker {ticker_symbol}"
        return df, None
    except Exception as e:
        return None, f"Error fetching data: {str(e)}"

# Main content area
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()

# Fetch and process data
with st.spinner(f"Fetching data for {ticker}..."):
    df, error = fetch_stock_data(ticker, start_date)

if error:
    st.error(error)
    st.stop()

if df is None or df.empty:
    st.error("No data available. Please check the ticker symbol and date range.")
    st.stop()

# Prepare data
df = df.copy()
df.reset_index(inplace=True)
df['Date'] = pd.to_datetime(df['Date'])

# Ensure we have the required columns
required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
if not all(col in df.columns for col in required_cols):
    st.error("Missing required columns in the data.")
    st.stop()

# Calculations
st.sidebar.subheader("📐 Calculations")

# 1. Position: Closing Price (already have it as 'Close')
df['Position'] = df['Close']

# 2. Velocity (1st Derivative): Rate of Change
df['Velocity'] = df['Close'].pct_change(periods=roc_period) * 100  # As percentage

# 3. Smoothing: EMA on Velocity
df['Smoothed_Velocity'] = df['Velocity'].ewm(span=ema_period, adjust=False).mean()

# 4. Acceleration (2nd Derivative): Change in Smoothed Velocity
df['Acceleration'] = df['Smoothed_Velocity'].diff()

# 5. SMA for price overlay
df['SMA'] = df['Close'].rolling(window=sma_period).mean()

# Identify inflection points (where Acceleration changes sign)
df['Acceleration_Sign'] = np.sign(df['Acceleration'])
df['Sign_Change'] = df['Acceleration_Sign'].diff()
inflection_points = df[df['Sign_Change'] != 0].copy()

# Filter out NaN values for cleaner visualization
df_clean = df.dropna(subset=['Velocity', 'Smoothed_Velocity', 'Acceleration', 'SMA']).copy()

if df_clean.empty:
    st.warning("Not enough data points for calculations. Please select a longer date range or smaller periods.")
    st.stop()

# Create visualization with subplots
fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    subplot_titles=(
        f"{ticker} Price Action with SMA({sma_period})",
        f"Velocity (ROC {roc_period}%) - Smoothed with EMA({ema_period})",
        "Acceleration (2nd Derivative)"
    ),
    row_heights=[0.5, 0.25, 0.25]
)

# Chart 1: Candlestick with SMA
fig.add_trace(
    go.Candlestick(
        x=df_clean['Date'],
        open=df_clean['Open'],
        high=df_clean['High'],
        low=df_clean['Low'],
        close=df_clean['Close'],
        name="Price",
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ),
    row=1, col=1
)

# Add SMA overlay
fig.add_trace(
    go.Scatter(
        x=df_clean['Date'],
        y=df_clean['SMA'],
        mode='lines',
        name=f'SMA({sma_period})',
        line=dict(color='#ffa726', width=2)
    ),
    row=1, col=1
)

# Add inflection point markers
if not inflection_points.empty:
    # Filter inflection points to only those in df_clean date range
    inflection_filtered = inflection_points[inflection_points['Date'].isin(df_clean['Date'])]
    
    if not inflection_filtered.empty:
        # Potential tops (positive to negative)
        tops = inflection_filtered[inflection_filtered['Sign_Change'] < 0]
        # Potential bottoms (negative to positive)
        bottoms = inflection_filtered[inflection_filtered['Sign_Change'] > 0]
        
        if not tops.empty:
            fig.add_trace(
                go.Scatter(
                    x=tops['Date'],
                    y=tops['Close'],
                    mode='markers',
                    name='Potential Top',
                    marker=dict(
                        symbol='triangle-down',
                        size=12,
                        color='red',
                        line=dict(width=2, color='darkred')
                    ),
                    hovertemplate='Potential Top<br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        if not bottoms.empty:
            fig.add_trace(
                go.Scatter(
                    x=bottoms['Date'],
                    y=bottoms['Close'],
                    mode='markers',
                    name='Potential Bottom',
                    marker=dict(
                        symbol='triangle-up',
                        size=12,
                        color='green',
                        line=dict(width=2, color='darkgreen')
                    ),
                    hovertemplate='Potential Bottom<br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )

# Chart 2: Velocity (ROC)
fig.add_trace(
    go.Scatter(
        x=df_clean['Date'],
        y=df_clean['Smoothed_Velocity'],
        mode='lines',
        name='Smoothed Velocity',
        line=dict(color='#42a5f5', width=2),
        fill='tozeroy',
        fillcolor='rgba(66, 165, 245, 0.2)'
    ),
    row=2, col=1
)

# Add zero line
fig.add_hline(
    y=0,
    line_dash="dash",
    line_color="gray",
    opacity=0.5,
    row=2, col=1
)

# Chart 3: Acceleration (Histogram)
# Create positive and negative bars separately for coloring
positive_accel = df_clean[df_clean['Acceleration'] >= 0]
negative_accel = df_clean[df_clean['Acceleration'] < 0]

if not positive_accel.empty:
    fig.add_trace(
        go.Bar(
            x=positive_accel['Date'],
            y=positive_accel['Acceleration'],
            name='Positive Acceleration',
            marker_color='green',
            marker_line_color='darkgreen',
            marker_line_width=0.5
        ),
        row=3, col=1
    )

if not negative_accel.empty:
    fig.add_trace(
        go.Bar(
            x=negative_accel['Date'],
            y=negative_accel['Acceleration'],
            name='Negative Acceleration',
            marker_color='red',
            marker_line_color='darkred',
            marker_line_width=0.5
        ),
        row=3, col=1
    )

# Update layout for dark mode
fig.update_layout(
    height=900,
    template='plotly_dark',
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    hovermode='x unified',
    xaxis_rangeslider_visible=False
)

# Update axes labels
fig.update_xaxes(title_text="Date", row=3, col=1)
fig.update_yaxes(title_text="Price ($)", row=1, col=1)
fig.update_yaxes(title_text="Velocity (%)", row=2, col=1)
fig.update_yaxes(title_text="Acceleration", row=3, col=1)

# Display the chart
st.plotly_chart(fig, use_container_width=True)

# Display summary statistics
st.subheader("📊 Summary Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Current Price", f"${df_clean['Close'].iloc[-1]:.2f}")
    st.metric("Current Velocity", f"{df_clean['Smoothed_Velocity'].iloc[-1]:.2f}%")

with col2:
    price_change = df_clean['Close'].iloc[-1] - df_clean['Close'].iloc[0]
    st.metric("Price Change", f"${price_change:.2f}", f"{(price_change/df_clean['Close'].iloc[0]*100):.2f}%")
    st.metric("Current Acceleration", f"{df_clean['Acceleration'].iloc[-1]:.4f}")

with col3:
    st.metric("Max Velocity", f"{df_clean['Smoothed_Velocity'].max():.2f}%")
    st.metric("Min Velocity", f"{df_clean['Smoothed_Velocity'].min():.2f}%")

with col4:
    st.metric("Max Acceleration", f"{df_clean['Acceleration'].max():.4f}")
    st.metric("Min Acceleration", f"{df_clean['Acceleration'].min():.4f}")

# Display recent data table
with st.expander("📋 View Recent Data"):
    display_cols = ['Date', 'Close', 'Velocity', 'Smoothed_Velocity', 'Acceleration']
    st.dataframe(
        df_clean[display_cols].tail(20).style.format({
            'Close': '${:.2f}',
            'Velocity': '{:.2f}%',
            'Smoothed_Velocity': '{:.2f}%',
            'Acceleration': '{:.4f}'
        }),
        use_container_width=True
    )

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Stock Price Acceleration Analysis Dashboard | Built with Streamlit & Plotly</p>
        <p><small>The 'Physics of Price' - Analyzing momentum through second derivative calculations</small></p>
    </div>
    """, unsafe_allow_html=True)
