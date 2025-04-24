import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

try:
   from alpha_vantage.fundamentaldata import FundamentalData
except ImportError:
   st.error("The 'alpha_vantage' package is not installed. Please install it using: pip install alpha_vantage")
   st.stop()
try:
   from transformers import pipeline
except ImportError:
   st.error("The 'transformers' package is not installed. Please install it using: pip install transformers")
   st.stop()
except RuntimeError as e:
   st.error(f"Failed to initialize transformers due to: {str(e)}. Please install tf-keras using: pip install tf-keras")
   st.stop()
import requests
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import statsmodels.api as sm
from scipy import stats
import holidays
from lifelines import KaplanMeierFitter
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import xgboost as xgb
import os
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed
from functools import lru_cache
import logging
import tensorflow as tf

try:
   import torch

   TORCH_AVAILABLE = True
except ImportError:
   TORCH_AVAILABLE = False
   torch = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY')
if not ALPHA_VANTAGE_KEY:
   st.error("Alpha Vantage API key not found. Please set it in a .env file with ALPHA_VANTAGE_KEY=your_key.")
   st.stop()

# Set page config
st.set_page_config(page_title="Finance Analytics Suite", page_icon="📊", layout="wide")

# Initialize APIs
fd = FundamentalData(key=ALPHA_VANTAGE_KEY)
try:
   # Try TensorFlow model first
   logger.info("Attempting to initialize sentiment pipeline with TensorFlow")
   sentiment_pipeline = pipeline(
      "sentiment-analysis",
      model="distilbert-base-uncased-finetuned-sst-2-english",
      framework="tf"
   )
   logger.info("Successfully initialized sentiment pipeline with TensorFlow")
except Exception as e:
   logger.error(f"Failed to initialize TensorFlow sentiment pipeline: {str(e)}")
   if TORCH_AVAILABLE:
      try:
         logger.info("Falling back to PyTorch for sentiment pipeline")
         sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            framework="pt"
         )
         logger.info("Successfully initialized sentiment pipeline with PyTorch")
      except Exception as e2:
         logger.error(f"Failed to initialize PyTorch sentiment pipeline: {str(e2)}")
         st.warning("Sentiment analysis is disabled due to initialization failure. Using fallback neutral sentiment.")
         sentiment_pipeline = None
   else:
      logger.error("PyTorch not available and TensorFlow failed. Disabling sentiment analysis.")
      st.warning(
         "Sentiment analysis is disabled due to missing PyTorch and TensorFlow failure. Using fallback neutral sentiment.")
      sentiment_pipeline = None


# ======================
# HELPER FUNCTIONS
# ======================

@retry(stop=stop_after_attempt(3), wait=wait_fixed(60))
@lru_cache(maxsize=128)
def get_company_info(symbol):
   """Fetch company information from Yahoo Finance."""
   try:
      stock = yf.Ticker(symbol)
      info = stock.info
      return {
         'name': info.get('longName', symbol),
         'sector': info.get('sector', 'N/A'),
         'industry': info.get('industry', 'N/A'),
         'country': info.get('country', 'N/A'),
         'employees': info.get('fullTimeEmployees', 'N/A'),
         'summary': info.get('longBusinessSummary', 'No description available'),
         'market_cap': info.get('marketCap', 'N/A')
      }
   except Exception as e:
      logger.error(f"Error fetching company info for {symbol}: {str(e)}")
      st.error(f"Error fetching company info for {symbol}: {str(e)}")
      return None


@retry(stop=stop_after_attempt(3), wait=wait_fixed(60))
@lru_cache(maxsize=128)
def get_stock_data(symbol, period='1y'):
   """Fetch stock data from Yahoo Finance."""
   try:
      data = yf.download(symbol, period=period)
      if data.empty:
         st.error(f"No stock data available for {symbol}.")
         return pd.DataFrame()
      return data
   except Exception as e:
      logger.error(f"Error fetching stock data for {symbol}: {str(e)}")
      st.error(f"Error fetching stock data for {symbol}: {str(e)}")
      return pd.DataFrame()


@retry(stop=stop_after_attempt(3), wait=wait_fixed(60))
@lru_cache(maxsize=128)
def get_financial_statements(symbol, statement_type):
   """Fetch financial statements from Alpha Vantage."""
   try:
      if statement_type == 'income':
         data, _ = fd.get_income_statement_annual(symbol)
      elif statement_type == 'balance':
         data, _ = fd.get_balance_sheet_annual(symbol)
      elif statement_type == 'cashflow':
         data, _ = fd.get_cash_flow_annual(symbol)
      if data.empty:
         st.error(f"No {statement_type} data available for {symbol}.")
         return None
      return data
   except Exception as e:
      logger.error(f"Error fetching {statement_type} statement for {symbol}: {str(e)}")
      st.error(f"Error fetching {statement_type} statement for {symbol}: {str(e)}")
      return None


def analyze_sentiment(text):
   """Analyze sentiment of text, handling long inputs by chunking."""
   if not sentiment_pipeline:
      return {'label': 'NEUTRAL', 'score': 0.5}
   if not isinstance(text, str) or not text.strip():
      return {'label': 'NEUTRAL', 'score': 0.5}
   try:
      max_length = 512
      chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)]
      results = [sentiment_pipeline(chunk)[0] for chunk in chunks]
      scores = [r['score'] if r['label'] == 'POSITIVE' else 1 - r['score'] for r in results]
      avg_score = sum(scores) / len(scores)
      avg_label = 'POSITIVE' if avg_score > 0.5 else 'NEGATIVE' if avg_score < 0.5 else 'NEUTRAL'
      return {'label': avg_label, 'score': avg_score}
   except Exception as e:
      logger.error(f"Sentiment analysis failed: {str(e)}")
      st.error(f"Sentiment analysis failed: {str(e)}")
      return {'label': 'NEUTRAL', 'score': 0.5}


@retry(stop=stop_after_attempt(3), wait=wait_fixed(60))
def get_news_sentiment(symbol):
   """Fetch and analyze news sentiment from Alpha Vantage."""
   try:
      url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={ALPHA_VANTAGE_KEY}'
      response = requests.get(url)
      response.raise_for_status()
      data = response.json()
      if 'feed' in data:
         sentiments = [analyze_sentiment(item.get('summary', ''))['label'] for item in data['feed'][:5]]
         if sentiments:
            positive = sentiments.count('POSITIVE') / len(sentiments)
            negative = sentiments.count('NEGATIVE') / len(sentiments)
            neutral = sentiments.count('NEUTRAL') / len(sentiments)
            return positive, negative, neutral
      return 0.33, 0.33, 0.34
   except Exception as e:
      logger.error(f"News sentiment analysis failed for {symbol}: {str(e)}")
      st.error(f"News sentiment analysis failed for {symbol}: {str(e)}")
      return 0.33, 0.33, 0.34


def calculate_technicals(data):
   """Calculate technical indicators for stock data."""
   if data.empty:
      return data
   data = data.copy()
   delta = data['Close'].diff()
   gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
   loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
   rs = gain / loss.where(loss != 0, np.inf)
   data['RSI'] = 100 - (100 / (1 + rs)).where(rs != np.inf, 50)
   ema12 = data['Close'].ewm(span=12, adjust=False).mean()
   ema26 = data['Close'].ewm(span=26, adjust=False).mean()
   data['MACD'] = ema12 - ema26
   data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
   data['BB_Middle'] = data['Close'].rolling(window=20).mean()
   data['BB_Std'] = data['Close'].rolling(window=20).std()
   data['BB_Upper'] = data['BB_Middle'] + 2 * data['BB_Std']
   data['BB_Lower'] = data['BB_Middle'] - 2 * data['BB_Std']
   high_low = data['High'] - data['Low']
   high_close = np.abs(data['High'] - data['Close'].shift())
   low_close = np.abs(data['Low'] - data['Close'].shift())
   tr = high_low.combine(high_close, max).combine(low_close, max)
   data['ATR'] = tr.rolling(window=14).mean()
   data['Lowest_14'] = data['Low'].rolling(window=14).min()
   data['Highest_14'] = data['High'].rolling(window=14).max()
   denominator = data['Highest_14'] - data['Lowest_14']
   data['Stoch_K'] = 100 * (data['Close'] - data['Lowest_14']) / denominator.where(denominator != 0, 1)
   data['Stoch_D'] = data['Stoch_K'].rolling(window=3).mean()
   data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
   return data


def get_dividend_analysis(symbol):
   """Calculate current and historical dividend yield."""
   try:
      stock = yf.Ticker(symbol)
      dividends = stock.dividends
      if not dividends.empty:
         history = stock.history(period='1d')
         if history.empty:
            return 0, pd.Series()
         current_price = history['Close'].iloc[-1]
         annual_dividend = dividends.resample('Y').sum().iloc[-1]
         yield_current = annual_dividend / current_price * 100
         yield_history = (dividends.resample('Y').sum() / stock.history(period='max')['Close'].resample(
            'Y').last() * 100).dropna()
         return yield_current, yield_history
      return 0, pd.Series()
   except Exception as e:
      logger.error(f"Error calculating dividend yield for {symbol}: {str(e)}")
      st.error(f"Error calculating dividend yield for {symbol}: {str(e)}")
      return 0, pd.Series()


def get_insider_institutional(symbol):
   """Fetch institutional and major holders data."""
   try:
      stock = yf.Ticker(symbol)
      return stock.institutional_holders, stock.major_holders
   except Exception as e:
      logger.error(f"Error fetching ownership data for {symbol}: {str(e)}")
      st.error(f"Error fetching ownership data for {symbol}: {str(e)}")
      return None, None


def get_options_chain(symbol):
   """Fetch options chain data."""
   try:
      stock = yf.Ticker(symbol)
      expiration_dates = stock.options
      if expiration_dates:
         opt = stock.option_chain(expiration_dates[0])
         calls = opt.calls[['strike', 'lastPrice', 'volume', 'openInterest', 'impliedVolatility']]
         puts = opt.puts[['strike', 'lastPrice', 'volume', 'openInterest', 'impliedVolatility']]
         return calls, puts, expiration_dates[0]
      return None, None, None
   except Exception as e:
      logger.error(f"Error fetching options chain for {symbol}: {str(e)}")
      st.error(f"Error fetching options chain for {symbol}: {str(e)}")
      return None, None, None


def compare_peers(symbol):
   """Compare stock performance with peers."""
   try:
      stock = yf.Ticker(symbol)
      peers = stock.info.get('relatedCompanies', [symbol, 'SPY'])[:5]
      data = yf.download(peers, period='1y')['Adj Close']
      if data.empty:
         st.error(f"No peer data available for {symbol}.")
         return pd.DataFrame()
      return data.pct_change().cumsum() * 100
   except Exception as e:
      logger.error(f"Error in peer comparison for {symbol}: {str(e)}")
      st.error(f"Error in peer comparison for {symbol}: {str(e)}")
      return pd.DataFrame()


def backtest_sma_strategy(data, short_window=50, long_window=200):
   """Backtest SMA crossover strategy."""
   if data.empty:
      logger.warning("Backtest aborted: Input data is empty")
      return pd.Series()

   # Ensure data is a DataFrame with expected columns
   if not isinstance(data, pd.DataFrame) or 'Close' not in data.columns:
      logger.error("Invalid input data: Must be a DataFrame with 'Close' column")
      st.error("Invalid stock data for backtesting. Please try a different symbol or period.")
      return pd.Series()

   data = data.copy()
   logger.info(f"Backtesting SMA strategy with short_window={short_window}, long_window={long_window}")

   # Calculate SMAs
   data['Short_SMA'] = data['Close'].rolling(window=short_window).mean()
   data['Long_SMA'] = data['Close'].rolling(window=long_window).mean()

   # Generate trading signals
   data['Signal'] = np.where(data['Short_SMA'] > data['Long_SMA'], 1, 0)
   data['Position'] = data['Signal'].shift()

   # Calculate daily returns
   daily_returns = data['Close'].pct_change()
   data['Returns'] = daily_returns * data['Position']

   # Calculate cumulative returns
   cumulative_returns = data['Returns'].cumsum()

   # Log data shape for debugging
   logger.info(f"Data shape after calculations: {data.shape}")

   return cumulative_returns


def lstm_price_prediction(data, symbol, days_ahead=30):
   """LSTM-based stock price prediction with model caching."""
   if data.empty:
      st.error("No data available for price prediction.")
      return pd.Series()
   model_path = f"lstm_model_{symbol}.h5"
   close_prices = data['Close'].values
   scaler = MinMaxScaler()
   scaled_data = scaler.fit_transform(close_prices.reshape(-1, 1))

   look_back = 60
   X, y = [], []
   for i in range(look_back, len(scaled_data)):
      X.append(scaled_data[i - look_back:i, 0])
      y.append(scaled_data[i, 0])
   X, y = np.array(X), np.array(y)
   X = np.reshape(X, (X.shape[0], X.shape[1], 1))

   train_size = int(len(X) * 0.8)
   X_train, X_test = X[:train_size], X[train_size:]
   y_train, y_test = y[:train_size], y[train_size:]

   if os.path.exists(model_path):
      model = tf.keras.models.load_model(model_path)
      st.info("Loaded pre-trained LSTM model.")
   else:
      model = Sequential()
      model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
      model.add(Dropout(0.2))
      model.add(LSTM(50))
      model.add(Dropout(0.2))
      model.add(Dense(1))
      model.compile(optimizer='adam', loss='mean_squared_error')
      model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
      model.save(model_path)
      st.info("Trained and saved new LSTM model.")

   last_sequence = scaled_data[-look_back:]
   future_predictions = []
   for _ in range(days_ahead):
      pred = model.predict(last_sequence.reshape(1, look_back, 1), verbose=0)
      future_predictions.append(pred[0, 0])
      last_sequence = np.append(last_sequence[1:], pred[0, 0])

   future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
   future_dates = pd.date_range(start=data.index[-1], periods=days_ahead + 1, freq='B')[1:]
   return pd.Series(future_predictions.flatten(), index=future_dates)


def stock_health_score(data, symbol):
   """Calculate heuristic-based stock health score."""
   if data.empty:
      return 0
   technicals = calculate_technicals(data)
   latest = technicals.iloc[-1]
   income = get_financial_statements(symbol, 'income')
   pos, neg, neu = get_news_sentiment(symbol)

   features = {
      'RSI': latest['RSI'] if not pd.isna(latest['RSI']) else 50,
      'MACD': latest['MACD'] if not pd.isna(latest['MACD']) else 0,
      'ATR': latest['ATR'] if not pd.isna(latest['ATR']) else 0,
      'Volume': latest['Volume'] if not pd.isna(latest['Volume']) else 0,
      'Sentiment_Pos': pos,
      'Revenue': float(income['totalRevenue'].iloc[0]) if income is not None and not income.empty else 0,
      'NetIncome': float(income['netIncome'].iloc[0]) if income is not None and not income.empty else 0
   }
   df = pd.DataFrame([features])
   scaler = StandardScaler()
   X = scaler.fit_transform(df)

   health_score = np.clip(np.mean(X[0]) * 100 + 50, 0, 100)
   st.warning("Health score is based on a heuristic model and should be used for informational purposes only.")
   return health_score


def detect_price_anomalies(data):
   """Detect anomalies in price movements."""
   if data.empty:
      return pd.DataFrame()
   df = data[['Close']].pct_change().dropna()
   if df.empty:
      return pd.DataFrame()
   clf = IsolationForest(contamination=0.05, random_state=42)
   df['Anomaly'] = clf.fit_predict(df)
   anomalies = data.loc[df[df['Anomaly'] == -1].index]
   return anomalies


def classify_trend(data):
   """Classify market trend as Bullish, Bearish, or Neutral."""
   if data.empty:
      return 'Neutral'
   returns = data['Close'].pct_change().dropna()
   X = pd.DataFrame({
      'Returns': returns,
      'Volatility': returns.rolling(20).std(),
      'RSI': calculate_technicals(data)['RSI']
   }).dropna()
   if X.empty:
      return 'Neutral'
   labels = np.where(X['Returns'].rolling(20).mean() > 0.01, 1,
                     np.where(X['Returns'].rolling(20).mean() < -0.01, -1, 0))
   model = xgb.XGBClassifier(random_state=42)
   model.fit(X.iloc[:-1], labels[1:])
   latest = X.iloc[-1].values.reshape(1, -1)
   pred = model.predict(latest)[0]
   st.warning("Trend prediction is based on a simplified model and should be used cautiously.")
   return {1: 'Bullish', -1: 'Bearish', 0: 'Neutral'}[pred]


# ======================
# AUDITING TOOLS CLASS
# ======================

class AdvancedAuditTools:
   def __init__(self):
      self.gst_rules = {
         'india': {'standard_rate': 0.18, 'threshold': 2000000},
         'singapore': {'standard_rate': 0.07, 'threshold': 1000000},
         'us': {'standard_rate': 0.0, 'threshold': 0}
      }

   def gst_compliance_check(self, transactions):
      """Check GST compliance for transactions."""
      try:
         df = pd.DataFrame(transactions)
         df['date'] = pd.to_datetime(df['date'], errors='coerce')
         if df['date'].isna().any():
            st.error("Invalid date values in transactions.")
            return None
         df['day_of_week'] = df['date'].dt.dayofweek
         df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
         df['amount_zscore'] = stats.zscore(df['amount'])
         clf = IsolationForest(contamination=0.05, random_state=42)
         features = ['amount', 'day_of_week', 'is_weekend']
         df['anomaly_score'] = clf.fit_predict(df[features])
         country = st.selectbox("Select Country", list(self.gst_rules.keys()))
         rate = self.gst_rules[country]['standard_rate']
         threshold = self.gst_rules[country]['threshold']
         df['gst_amount'] = df['amount'] * rate
         df['gst_compliant'] = df['amount'] > threshold
         return df
      except Exception as e:
         logger.error(f"GST analysis failed: {str(e)}")
         st.error(f"GST analysis failed: {str(e)}")
         return None

   def accounting_standard_check(self, financials):
      """Classify financials as IFRS or GAAP."""
      try:
         X = financials[['revenue_growth', 'asset_turnover', 'debt_ratio']]
         scaler = StandardScaler()
         X_scaled = scaler.fit_transform(X)
         kmeans = KMeans(n_clusters=2, random_state=42)
         clusters = kmeans.fit_predict(X_scaled)
         standards = ['IFRS', 'GAAP']
         financials['predicted_standard'] = [standards[c] for c in clusters]
         return financials
      except Exception as e:
         logger.error(f"Accounting standard analysis failed: {str(e)}")
         st.error(f"Accounting standard analysis failed: {str(e)}")
         return None

   def inventory_audit(self, inventory_data):
      """Audit inventory data for anomalies."""
      try:
         inventory_data['date'] = pd.to_datetime(inventory_data['date'], errors='coerce')
         if inventory_data['date'].isna().any():
            st.error("Invalid date values in inventory data.")
            return None
         ts = inventory_data.set_index('date')['quantity']
         model = sm.tsa.SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
         results = model.fit(disp=False)
         predictions = results.get_prediction()
         ts_pred = predictions.predicted_mean
         residuals = ts - ts_pred
         threshold = residuals.std() * 3
         anomalies = (residuals.abs() > threshold).astype(int)
         return {'actual': ts, 'predicted': ts_pred, 'anomalies': anomalies, 'model_summary': results.summary()}
      except Exception as e:
         logger.error(f"Inventory analysis failed: {str(e)}")
         st.error(f"Inventory analysis failed: {str(e)}")
         return None

   def receivables_analysis(self, invoices):
      """Analyze receivables aging and payment probability."""
      try:
         invoices = invoices.copy()
         invoices['due_date'] = pd.to_datetime(invoices['due_date'], errors='coerce')
         if invoices['due_date'].isna().any():
            st.error("Invalid due_date values detected.")
            return None
         now = pd.Timestamp.now(tz='UTC')
         invoices['days_outstanding'] = (now - invoices['due_date']).dt.days
         invoices['is_paid'] = invoices['paid_amount'] > 0
         kmf = KaplanMeierFitter()
         kmf.fit(invoices['days_outstanding'], invoices['is_paid'])
         payment_prob = kmf.survival_function_
         X = invoices[['amount', 'days_outstanding']]
         scaler = StandardScaler()
         X_scaled = scaler.fit_transform(X)
         invoices['risk_cluster'] = KMeans(n_clusters=3).fit_predict(X_scaled)
         return {'payment_probability': payment_prob, 'invoices': invoices,
                 'high_risk': invoices[invoices['risk_cluster'] == 2]}
      except Exception as e:
         logger.error(f"Receivables analysis failed: {str(e)}")
         st.error(f"Receivables analysis failed: {str(e)}")
         return None


# ======================
# STREAMLIT UI
# ======================

def validate_symbol(symbol):
   """Validate stock symbol."""
   if not symbol or not symbol.isalnum():
      st.error("Please enter a valid stock symbol (alphanumeric characters only).")
      return False
   try:
      stock = yf.Ticker(symbol)
      stock.info  # Test if symbol is valid
      return True
   except Exception:
      st.error(f"Invalid stock symbol: {symbol}. Please try a different symbol.")
      return False


def main():
   st.sidebar.title("Navigation")
   page = st.sidebar.radio("Go to", [
      "Company Overview", "Financial Statements", "Stock Analysis", "Sentiment Analysis",
      "Portfolio Builder", "Auditing Tools", "Options Analysis", "Backtesting", "AI Predictions"
   ])

   symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, MSFT)", "AAPL").upper().strip()
   if not validate_symbol(symbol):
      return

   st.sidebar.subheader("Set Alerts")
   alert_type = st.sidebar.selectbox("Alert Type", ["Price", "RSI", "News"])
   threshold = st.sidebar.number_input("Threshold", min_value=0.0, value=100.0)
   if st.sidebar.button("Set Alert"):
      if threshold <= 0:
         st.error("Threshold must be positive.")
      else:
         st.session_state[f"alert_{symbol}_{alert_type}"] = threshold
         st.sidebar.success(f"Alert set for {alert_type} at {threshold}")

   if page == "Company Overview":
      st.header("🏢 Company Overview")
      info = get_company_info(symbol)
      if info:
         col1, col2 = st.columns([1, 2])
         with col1:
            st.subheader("Basic Information")
            st.metric("Company Name", info['name'])
            st.write(f"**Sector:** {info['sector']}")
            st.write(f"**Industry:** {info['industry']}")
            st.write(f"**Country:** {info['country']}")
            st.write(f"**Employees:** {info['employees']:,}")
            st.metric("Market Cap",
                      f"${info['market_cap']:,}" if isinstance(info['market_cap'], int) else info['market_cap'])
         with col2:
            st.subheader("Business Summary")
            st.write(info['summary'])
         st.subheader("Dividend Analysis")
         yield_current, yield_history = get_dividend_analysis(symbol)
         st.metric("Current Dividend Yield", f"{yield_current:.2f}%")
         if not yield_history.empty:
            fig = px.line(yield_history, title="Historical Dividend Yield")
            st.plotly_chart(fig)
         st.subheader("Ownership Insights")
         inst_holders, major_holders = get_insider_institutional(symbol)
         if inst_holders is not None:
            st.write("Institutional Holders")
            st.dataframe(inst_holders)
         if major_holders is not None:
            st.write("Major Holders")
            st.dataframe(major_holders)
      else:
         st.error("Could not fetch company information.")

   elif page == "Financial Statements":
      st.header("📑 Financial Statements")
      tab1, tab2, tab3 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])
      with tab1:
         st.subheader("Income Statement")
         income_stmt = get_financial_statements(symbol, 'income')
         if income_stmt is not None:
            numeric_cols = income_stmt.select_dtypes(include=[np.number]).columns
            format_dict = {col: "{:,.0f}" for col in numeric_cols}
            st.dataframe(income_stmt.style.format(format_dict))
      with tab2:
         st.subheader("Balance Sheet")
         balance_sheet = get_financial_statements(symbol, 'balance')
         if balance_sheet is not None:
            numeric_cols = balance_sheet.select_dtypes(include=[np.number]).columns
            format_dict = {col: "{:,.0f}" for col in numeric_cols}
            st.dataframe(balance_sheet.style.format(format_dict))
      with tab3:
         st.subheader("Cash Flow Statement")
         cash_flow = get_financial_statements(symbol, 'cashflow')
         if cash_flow is not None:
            numeric_cols = cash_flow.select_dtypes(include=[np.number]).columns
            format_dict = {col: "{:,.0f}" for col in numeric_cols}
            st.dataframe(cash_flow.style.format(format_dict))

   elif page == "Stock Analysis":
      st.header("📈 Stock Analysis")
      period_options = {'1 Month': '1mo', '3 Months': '3mo', '6 Months': '6mo', '1 Year': '1y', '2 Years': '2y',
                        '5 Years': '5y', 'Max': 'max'}
      selected_period = st.selectbox("Select Time Period", list(period_options.keys()))
      data = get_stock_data(symbol, period_options[selected_period])

      if not data.empty:
         data = calculate_technicals(data.dropna())
         if f"alert_{symbol}_Price" in st.session_state and data['Close'].iloc[-1] >= st.session_state[
            f"alert_{symbol}_Price"]:
            st.warning(
               f"Alert: {symbol} price hit {st.session_state[f'alert_{symbol}_Price']}! Current: {data['Close'].iloc[-1]:.2f}")
         if f"alert_{symbol}_RSI" in st.session_state and data['RSI'].iloc[-1] >= st.session_state[
            f"alert_{symbol}_RSI"]:
            st.warning(f"Alert: RSI hit {st.session_state[f'alert_{symbol}_RSI']}! Current: {data['RSI'].iloc[-1]:.2f}")

         # Candlestick Chart
         st.subheader(f"{symbol} Candlestick Chart")
         fig = go.Figure(data=[
            go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'])])
         fig.update_layout(title=f"{symbol} Stock Price", yaxis_title="Price ($)", height=600)
         st.plotly_chart(fig)

         # Technical Indicators
         st.subheader("Technical Indicators")
         indicators = st.multiselect("Select Indicators",
                                     ['RSI', 'MACD', 'Bollinger Bands', 'ATR', 'Stochastic', 'VWAP'],
                                     default=['RSI', 'MACD'])
         fig = go.Figure()
         fig.add_trace(
            go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'],
                           name='Price'))
         for ind in indicators:
            if ind == 'RSI':
               fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', yaxis='y2'))
            elif ind == 'MACD':
               fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD'))
               fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], name='MACD Signal'))
            elif ind == 'Bollinger Bands':
               fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], name='BB Upper'))
               fig.add_trace(go.Scatter(x=data.index, y=data['BB_Middle'], name='BB Middle'))
               fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], name='BB Lower'))
            elif ind == 'ATR':
               fig.add_trace(go.Scatter(x=data.index, y=data['ATR'], name='ATR', yaxis='y2'))
            elif ind == 'Stochastic':
               fig.add_trace(go.Scatter(x=data.index, y=data['Stoch_K'], name='Stoch %K', yaxis='y2'))
               fig.add_trace(go.Scatter(x=data.index, y=data['Stoch_D'], name='Stoch %D', yaxis='y2'))
            elif ind == 'VWAP':
               fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], name='VWAP'))
         fig.update_layout(yaxis2=dict(overlaying='y', side='right'), title=f"{symbol} Technical Analysis", height=600)
         st.plotly_chart(fig)

         # Peer Comparison
         st.subheader("Peer Comparison")
         peer_data = compare_peers(symbol)
         if not peer_data.empty:
            fig = px.line(peer_data, title=f"{symbol} vs Peers (Cumulative Returns)")
            st.plotly_chart(fig)

   elif page == "Sentiment Analysis":
      st.header("😊 Sentiment Analysis")
      st.subheader("News Sentiment")
      positive, negative, neutral = get_news_sentiment(symbol)
      cols = st.columns(3)
      cols[0].metric("Positive", f"{positive * 100:.1f}%", delta=f"{(positive - 0.33) * 100:.1f}%")
      cols[1].metric("Negative", f"{negative * 100:.1f}%", delta=f"{(negative - 0.33) * 100:.1f}%",
                     delta_color="inverse")
      cols[2].metric("Neutral", f"{neutral * 100:.1f}%", delta=f"{(neutral - 0.34) * 100:.1f}%")
      fig = go.Figure(data=[
         go.Pie(labels=['Positive', 'Negative', 'Neutral'], values=[positive, negative, neutral], hole=0.3)
      ])
      fig.update_layout(title="Sentiment Distribution")
      st.plotly_chart(fig)

   elif page == "Portfolio Builder":
      st.header("💼 Portfolio Builder")
      if 'portfolio' not in st.session_state:
         st.session_state.portfolio = {}
      st.subheader("Add Stocks to Your Portfolio")
      col1, col2 = st.columns(2)
      with col1:
         new_symbol = st.text_input("Stock Symbol", "").upper().strip()
      with col2:
         shares = st.number_input("Shares", min_value=1, value=100)
      if st.button("Add to Portfolio") and new_symbol:
         if not validate_symbol(new_symbol):
            return
         if shares <= 0:
            st.error("Shares must be a positive integer.")
            return
         if new_symbol in st.session_state.portfolio:
            st.session_state.portfolio[new_symbol] += shares
         else:
            st.session_state.portfolio[new_symbol] = shares
         st.success(f"Added {shares} shares of {new_symbol} to portfolio")
      if st.session_state.portfolio:
         st.subheader("Your Portfolio")
         portfolio_data = []
         total_value = 0
         for sym, shares in st.session_state.portfolio.items():
            try:
               stock = yf.Ticker(sym)
               current_price = stock.history(period='1d')['Close'].iloc[-1]
               value = current_price * shares
               total_value += value
               prev_close = stock.history(period='2d')['Close'].iloc[0]
               daily_change = (current_price - prev_close) / prev_close
               portfolio_data.append({'Symbol': sym, 'Shares': shares, 'Price': current_price, 'Value': value,
                                      'Daily Change': daily_change})
            except Exception as e:
               logger.error(f"Error processing portfolio stock {sym}: {str(e)}")
               portfolio_data.append(
                  {'Symbol': sym, 'Shares': shares, 'Price': 'N/A', 'Value': 'N/A', 'Daily Change': 'N/A'})
         df = pd.DataFrame(portfolio_data)

         def safe_format(x, format_str):
            try:
               if pd.isna(x) or x == 'N/A':
                  return 'N/A'
               return format_str.format(float(x))
            except:
               return str(x)

         styled_df = df.style.format({
            'Price': lambda x: safe_format(x, '${:,.2f}'),
            'Value': lambda x: safe_format(x, '${:,.2f}'),
            'Daily Change': lambda x: safe_format(x, '{:.2%}')
         })
         st.dataframe(styled_df)
         st.metric("Total Portfolio Value", f"${total_value:,.2f}")
         if total_value > 0:
            fig = go.Figure(data=[
               go.Pie(labels=df[df['Value'] != 'N/A']['Symbol'], values=df[df['Value'] != 'N/A']['Value'], hole=0.3)
            ])
            fig.update_layout(title="Portfolio Allocation")
            st.plotly_chart(fig)

   elif page == "Auditing Tools":
      st.header("🔍 Advanced Auditing Tools")
      auditor = AdvancedAuditTools()
      audit_function = st.selectbox("Select Audit Function", [
         "GST/TDS Compliance Check", "IFRS/GAAP Classifier", "Inventory Audit", "Receivables Aging Analysis"
      ])
      if audit_function == "GST/TDS Compliance Check":
         st.subheader("💰 GST/TDS Compliance Check")
         dates = pd.date_range('2023-01-01', periods=100)
         amounts = np.random.lognormal(mean=8, sigma=1.5, size=100)
         vendors = np.random.choice(['Vendor A', 'Vendor B', 'Vendor C', 'Vendor D'], 100)
         transactions = pd.DataFrame({'date': dates, 'amount': amounts, 'vendor': vendors})
         if st.button("Run Compliance Check"):
            with st.spinner("Analyzing transactions..."):
               result = auditor.gst_compliance_check(transactions)
               if result is not None:
                  st.success("Analysis completed!")
                  col1, col2 = st.columns(2)
                  col1.metric("Total GST Liability", f"${result['gst_amount'].sum():,.2f}")
                  col2.metric("Anomaly Rate", f"{len(result[result['anomaly_score'] == -1]) / len(result):.1%}")
                  normal = result[result['anomaly_score'] == 1]
                  anomalies = result[result['anomaly_score'] == -1]
                  fig = go.Figure()
                  fig.add_trace(go.Scatter(x=normal['date'], y=normal['amount'], mode='markers', name='Normal',
                                           marker=dict(color='blue')))
                  fig.add_trace(go.Scatter(x=anomalies['date'], y=anomalies['amount'], mode='markers', name='Anomaly',
                                           marker=dict(color='red')))
                  fig.update_layout(title="Transaction Analysis", height=600)
                  st.plotly_chart(fig)
                  st.write("### Anomaly Details")
                  st.dataframe(anomalies)
      elif audit_function == "IFRS/GAAP Classifier":
         st.subheader("🌐 Accounting Standard Classifier")
         companies = [f"Company {chr(65 + i)}" for i in range(10)]
         financials = pd.DataFrame({
            'company': companies,
            'revenue_growth': np.random.normal(0.08, 0.03, 10),
            'asset_turnover': np.random.uniform(0.5, 1.5, 10),
            'debt_ratio': np.random.uniform(0.2, 0.7, 10)
         })
         if st.button("Classify Standards"):
            with st.spinner("Analyzing financial patterns..."):
               result = auditor.accounting_standard_check(financials)
               if result is not None:
                  st.write("### Classification Results")
                  st.dataframe(result)
                  fig = go.Figure()
                  for standard in ['IFRS', 'GAAP']:
                     subset = result[result['predicted_standard'] == standard]
                     fig.add_trace(go.Scatter3d(
                        x=subset['revenue_growth'], y=subset['asset_turnover'], z=subset['debt_ratio'],
                        mode='markers', name=standard,
                        marker=dict(size=5, opacity=0.8)
                     ))
                  fig.update_layout(
                     title="Financial Profile Clustering",
                     scene=dict(
                        xaxis_title="Revenue Growth",
                        yaxis_title="Asset Turnover",
                        zaxis_title="Debt Ratio"
                     ),
                     height=600
                  )
                  st.plotly_chart(fig)
      elif audit_function == "Inventory Audit":
         st.subheader("📦 Inventory Anomaly Detection")
         dates = pd.date_range('2022-01-01', '2023-01-01', freq='D')
         base = 100 + 20 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365)
         noise = np.random.normal(0, 5, len(dates))
         quantities = np.maximum(10, base + noise).astype(int)
         quantities[30] = 200
         quantities[150:155] = 10
         inventory_data = pd.DataFrame({'date': dates, 'quantity': quantities})
         if st.button("Analyze Inventory"):
            with st.spinner("Running time series analysis..."):
               result = auditor.inventory_audit(inventory_data)
               if result is not None:
                  st.write("### Inventory Analysis Results")
                  fig = go.Figure()
                  fig.add_trace(
                     go.Scatter(x=result['actual'].index, y=result['actual'], name='Actual', line=dict(color='blue')))
                  fig.add_trace(go.Scatter(x=result['predicted'].index, y=result['predicted'], name='Predicted',
                                           line=dict(color='green', dash='dash')))
                  anomalies = result['actual'][result['anomalies'] == 1]
                  fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies, mode='markers', name='Anomaly',
                                           marker=dict(color='red', size=10)))
                  fig.update_layout(title="Inventory Level Analysis", height=600)
                  st.plotly_chart(fig)
                  st.write("### Model Summary")
                  st.text(result['model_summary'])
      elif audit_function == "Receivables Aging Analysis":
         st.subheader("⏳ Receivables Aging Analysis")
         due_dates = pd.date_range('2022-01-01', periods=50, freq='7D')
         paid_dates = [d + timedelta(days=np.random.randint(0, 90)) if np.random.random() > 0.3 else None for d in
                       due_dates]
         invoices = pd.DataFrame({
            'invoice_id': range(1001, 1051),
            'due_date': due_dates,
            'paid_date': paid_dates,
            'amount': np.random.uniform(100, 5000, 50),
            'paid_amount': [a if pd.notnull(d) else 0 for a, d in zip(np.random.uniform(100, 5000, 50), paid_dates)],
            'customer': np.random.choice(['Customer A', 'Customer B', 'Customer C'], 50)
         })
         if st.button("Analyze Receivables"):
            with st.spinner("Running receivables analysis..."):
               result = auditor.receivables_analysis(invoices)
               if result is not None:
                  st.write("### Payment Probability Analysis")
                  fig = go.Figure()
                  fig.add_trace(go.Scatter(x=result['payment_probability'].index, y=result['payment_probability'],
                                           name='Non-Payment Probability'))
                  fig.update_layout(
                     title="Payment Probability Curve",
                     xaxis_title="Days Outstanding",
                     yaxis_title="Probability of Non-Payment",
                     height=500
                  )
                  st.plotly_chart(fig)
                  st.write("### High Risk Receivables")
                  st.dataframe(result['high_risk'])
                  st.write("### Aging Summary")
                  aging = pd.cut(result['invoices']['days_outstanding'], bins=[0, 30, 60, 90, np.inf],
                                 labels=['0-30', '31-60', '61-90', '90+'])
                  fig = go.Figure(data=[
                     go.Bar(x=aging.value_counts().index, y=aging.value_counts().values)
                  ])
                  fig.update_layout(title="Receivables Aging Distribution", height=400)
                  st.plotly_chart(fig)

   elif page == "Options Analysis":
      st.header("📊 Options Analysis")
      calls, puts, exp_date = get_options_chain(symbol)
      if calls is not None:
         st.subheader(f"Calls (Expiration: {exp_date})")
         st.dataframe(calls)
         st.subheader(f"Puts (Expiration: {exp_date})")
         st.dataframe(puts)
         fig = px.scatter(calls, x='strike', y='impliedVolatility', title="Implied Volatility vs Strike (Calls)")
         st.plotly_chart(fig)

   elif page == "Backtesting":
      st.header("🔄 Backtesting")
      period_options = {'1 Year': '1y', '2 Years': '2y', '5 Years': '5y'}
      selected_period = st.selectbox("Select Time Period", list(period_options.keys()))
      data = get_stock_data(symbol, period_options[selected_period])
      if not data.empty:
         st.subheader("SMA Crossover Strategy")
         short_w = st.slider("Short SMA Window", 10, 100, 50)
         long_w = st.slider("Long SMA Window", 50, 300, 200)
         if st.button("Run Backtest"):
            returns = backtest_sma_strategy(data, short_w, long_w)
            if not returns.empty:
               st.write(f"Strategy Return: {returns.iloc[-1]:.2f}%")
               fig = px.line(returns, title="Cumulative Returns")
               st.plotly_chart(fig)
            else:
               st.error("Backtest failed: No valid returns generated.")

   elif page == "AI Predictions":
      st.header("🤖 AI Predictions")
      period_options = {'1 Year': '1y', '2 Years': '2y', '5 Years': '5y'}
      selected_period = st.selectbox("Select Time Period", list(period_options.keys()))
      data = get_stock_data(symbol, period_options[selected_period])

      if not data.empty:
         data = calculate_technicals(data.dropna())

         # LSTM Price Prediction
         st.subheader("Price Forecast (LSTM)")
         days_ahead = st.slider("Days Ahead", 5, 90, 30)
         if st.button("Generate Forecast"):
            with st.spinner("Generating price forecast..."):
               forecast = lstm_price_prediction(data, symbol, days_ahead)
               if not forecast.empty:
                  fig = go.Figure()
                  fig.add_trace(go.Candlestick(
                     x=data.index[-60:], open=data['Open'][-60:], high=data['High'][-60:],
                     low=data['Low'][-60:], close=data['Close'][-60:], name='Historical'
                  ))
                  fig.add_trace(go.Scatter(x=forecast.index, y=forecast, name='Forecast', line=dict(color='orange')))
                  fig.update_layout(title=f"{symbol} Price Forecast", yaxis_title="Price ($)", height=600)
                  st.plotly_chart(fig)

         # Stock Health Score
         st.subheader("Stock Health Score")
         score = stock_health_score(data, symbol)
         st.metric("Health Score", f"{score:.1f}/100")

         # Anomaly Detection
         st.subheader("Price Anomaly Detection")
         anomalies = detect_price_anomalies(data)
         if not anomalies.empty:
            st.write("Detected Anomalies")
            st.dataframe(anomalies)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Price', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies['Close'], mode='markers', name='Anomaly',
                                     marker=dict(color='red', size=10)))
            fig.update_layout(title="Price Anomaly Detection", height=600)
            st.plotly_chart(fig)
         else:
            st.write("No anomalies detected.")

         # Trend Classification
         st.subheader("Market Trend Prediction")
         trend = classify_trend(data)
         st.metric("Predicted Trend", trend)


if __name__ == "__main__":
   main()