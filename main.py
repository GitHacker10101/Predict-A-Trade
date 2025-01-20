import streamlit as st
from datetime import date
import requests
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# Constants
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
API_KEY = "N392E740G7X4PI9G"
API_URL = "https://www.alphavantage.co/query"

# Streamlit App Title and UI Enhancements
st.set_page_config(page_title="PredictaTrade", page_icon="📈", layout="wide")

# Custom CSS to make the background white and text black, including sidebar and all components
st.markdown("""
    <style>
        body {
            background-color: white;
            color: black;
        }
        .css-18e3th9, .css-1v3fvcr, .css-1fgkdo6, .css-1s2v0on {
            background-color: white;
            color: black;
        }
        .css-1v3fvcr div, .css-1fgkdo6 div, .css-1s2v0on div {
            color: black;
        }
        .css-1v3fvcr {
            background-color: white !important;
            color: black !important;
        }
        .stSidebar {
            background-color: white !important;
            color: black !important;
        }
        .stTextInput>div>input {
            color: black !important;
            background-color: #f0f0f0 !important;
        }
        .stSelectbox>div>div>input {
            color: black !important;
            background-color: #f0f0f0 !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title('PredictaTrade')
stocks = ('GOOG', 'AAPL', 'MSFT', 'GME', 'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 
          'HDFCBANK.NS', 'BAJFINANCE.NS', 'IDFC.NS', 'OLAELEC.NS', 'MRF.NS', 'LICI.NS', 'NVDA')

selected_stock = st.selectbox('Select Stock Dataset for Prediction:', stocks)

n_years = st.slider('Years of Prediction:', 1, 5)
period = n_years * 365

# Function to Load Data
@st.cache_resource
def load_data(ticker):
    """
    Fetch stock data using Alpha Vantage API.
    """
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": ticker,
        "outputsize": "full",
        "apikey": API_KEY
    }
    response = requests.get(API_URL, params=params)
    
    if response.status_code == 200:
        data = response.json()
        daily_data = data.get("Time Series (Daily)")
        if not daily_data:
            raise ValueError(f"No data found for symbol: {ticker}")
        
        df = pd.DataFrame.from_dict(daily_data, orient="index")
        df.reset_index(inplace=True)
        df.rename(columns={"index": "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        df = df.rename(columns={
            "1. open": "open",
            "4. close": "close",
        })
        df["open"] = pd.to_numeric(df["open"])
        df["close"] = pd.to_numeric(df["close"])
        return df
    else:
        st.write("Error: Unable to fetch data from Alpha Vantage.")
        raise Exception(f"API Error: {response.status_code} {response.text}")

# Load Data
data_load_state = st.text('Loading data...')
try:
    data = load_data(selected_stock)
    data_load_state.text('Loading data... done!')
    
    # Show Raw Data Table
    st.subheader('Raw Stock Data')
    st.write(data.tail())

    # Plot Raw Data (Open vs Close)
    def plot_raw_data():
        fig = go.Figure()

        # Plot actual Open and Close values with thinner lines
        fig.add_trace(go.Scatter(x=data['date'], y=data['open'], name="Stock Open", line=dict(color='blue', width=1)))
        fig.add_trace(go.Scatter(x=data['date'], y=data['close'], name="Stock Close", line=dict(color='green', width=1)))
        
        # Improve layout with title and axis labels
        fig.layout.update(
            title_text=f'{selected_stock} Stock Price Time Series',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            xaxis_rangeslider_visible=True,
            plot_bgcolor='white',  # Set background color to white
            paper_bgcolor='white',  # Set paper background to white
            font=dict(color='black')  # Black font color
        )
        st.plotly_chart(fig)

    plot_raw_data()

    # Prepare data for forecasting
    df_train = data[['date', 'close']].rename(columns={"date": "ds", "close": "y"})

    # Forecast with Prophet
    m = Prophet(daily_seasonality=True)
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    # Show Forecast Data
    st.subheader('Forecast Data')
    st.write(forecast.tail())

    # Determine the color of the forecast line (green if going up on average, red if going down on average)
    forecast['color'] = ['green' if forecast['yhat'][i] < forecast['yhat'][i + 1] else 'red' for i in range(len(forecast) - 1)] + ['green']  # Last point assumed green

    # Plot Forecast (Actual vs Forecast)
    st.write(f'Forecast plot for the next {n_years} years')

    # Create a new figure for the forecast
    fig1 = go.Figure()

    # Plot Actual Close
    fig1.add_trace(go.Scatter(x=data['date'], y=data['close'], name="Actual Close", line=dict(color='blue', width=1)))
    
    # Plot Forecasted Close with dynamically changing color
    fig1.add_trace(go.Scatter(
        x=forecast['ds'], 
        y=forecast['yhat'], 
        name="Forecasted Close", 
        line=dict(color='green' if forecast['yhat'][0] < forecast['yhat'].iloc[-1] else 'red', width=2)  # Overall trend
    ))


    fig1.update_layout(
        title=f'{selected_stock} Stock Price Forecast',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        xaxis_rangeslider_visible=True,  
        xaxis=dict(type="date"),  
        plot_bgcolor='white',  
        paper_bgcolor='white',  
        font=dict(color='black') 
    )

    st.plotly_chart(fig1)

    # Price and Percentage Change
    current_price = data['close'].iloc[-1]
    prev_price = data['close'].iloc[-2]
    price_change = current_price - prev_price
    percentage_change = (price_change / prev_price) * 100

    st.subheader(f"Current Price and Percentage Change")
    st.write(f"**Current Price:** ${current_price:.2f}")
    st.write(f"**Price Change:** ${price_change:.2f}")
    st.write(f"**Percentage Change:** {percentage_change:.2f}%")

    # Plot Forecast Components
    st.write("Trends(Weekly, Yearly Seasonality)") #Fixed Title Bug 
    fig2 = m.plot_components(forecast)
    st.write(fig2)
    
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.write(f"Details: {e}")

st.write('Made By Krishnav Jain, Aryan Kapoor, and Vedant Jain')
