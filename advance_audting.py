import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from alpha_vantage.fundamentaldata import FundamentalData
from transformers import pipeline
import requests
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import statsmodels.api as sm
from scipy import stats
import holidays
from lifelines import KaplanMeierFitter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import xgboost as xgb

# Set page config
st.set_page_config(page_title="Finance Analytics Suite", page_icon="📊", layout="wide")

# API Keys
ALPHA_VANTAGE_KEY = 'S9N79HUNKBK1GWPW'

# Initialize APIs
fd = FundamentalData(key=ALPHA_VANTAGE_KEY)
sentiment_pipeline = pipeline("sentiment-analysis")

# ======================
# HELPER FUNCTIONS
# ======================

@st.cache_data
def get_company_info(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        return {
            'name': info.get('longName', symbol), 'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'), 'country': info.get('country', 'N/A'),
            'employees': info.get('fullTimeEmployees', 'N/A'),
            'summary': info.get('longBusinessSummary', 'No description available'),
            'market_cap': info.get('marketCap', 'N/A')
        }
    except Exception as e:
        st.error(f"Error fetching company info: {str(e)}")
        return None

@st.cache_data
def get_stock_data(symbol, start_date=None, end_date=None, period='1y'):
    try:
        if start_date and end_date:
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            data = yf.download(symbol, start=start_str, end=end_str)
        else:
            data = yf.download(symbol, period=period)
        if data.empty:
            st.warning(f"No data found for {symbol}. Trying max period...")
            data = yf.download(symbol, period='max')
            if data.empty:
                return pd.DataFrame()
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            st.error(f"Missing required columns in data for {symbol}")
            return pd.DataFrame()
        if data.isnull().any().any():
            data = data.fillna(method='ffill')
        return data
    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def get_financial_statements(symbol, statement_type):
    try:
        if statement_type == 'income':
            data, _ = fd.get_income_statement_annual(symbol)
        elif statement_type == 'balance':
            data, _ = fd.get_balance_sheet_annual(symbol)
        elif statement_type == 'cashflow':
            data, _ = fd.get_cash_flow_annual(symbol)
        return data
    except Exception as e:
        st.error(f"Error fetching {statement_type} statement: {str(e)}")
        return None

def analyze_sentiment(text):
    if not text or str(text).strip() == '':
        return {'label': 'NEUTRAL', 'score': 0.5}
    try:
        return sentiment_pipeline(text[:512])[0]
    except Exception as e:
        st.error(f"Sentiment analysis failed: {str(e)}")
        return {'label': 'NEUTRAL', 'score': 0.5}

def get_news_sentiment(symbol):
    try:
        url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={ALPHA_VANTAGE_KEY}'
        response = requests.get(url)
        data = response.json()
        if 'feed' in data:
            sentiments = [analyze_sentiment(item['summary'])['label'] for item in data['feed'][:5] if 'summary' in item]
            if sentiments:
                positive = sentiments.count('POSITIVE') / len(sentiments)
                negative = sentiments.count('NEGATIVE') / len(sentiments)
                neutral = sentiments.count('NEUTRAL') / len(sentiments)
                return positive, negative, neutral
        return 0.33, 0.33, 0.34
    except Exception as e:
        st.error(f"News sentiment analysis failed: {str(e)}")
        return 0.33, 0.33, 0.34

def calculate_technicals(data):
    if data.empty:
        return data
    high_low = data['High'] - data['Low']
    high_close = abs(data['High'] - data['Close'].shift())
    low_close = abs(data['Low'] - data['Close'].shift())
    # Fixed: Use DataFrame and .max() for True Range
    tr = pd.DataFrame({
        'high_low': high_low,
        'high_close': high_close,
        'low_close': low_close
    }).max(axis=1)
    data['ATR'] = tr.rolling(window=14).mean()
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['SMA'] = data['Close'].rolling(window=20).mean()
    data['BB_std'] = data['Close'].rolling(window=20).std()
    data['BB_upper'] = data['SMA'] + (data['BB_std'] * 2)
    data['BB_lower'] = data['SMA'] - (data['BB_std'] * 2)
    return data

def get_dividend_analysis(symbol):
    stock = yf.Ticker(symbol)
    dividends = stock.dividends
    if not dividends.empty:
        current_price = stock.history(period='1d')['Close'].iloc[-1]
        annual_dividend = dividends.resample('Y').sum().iloc[-1]
        yield_current = annual_dividend / current_price * 100
        yield_history = (dividends.resample('Y').sum() / stock.history(period='max')['Close'].resample('Y').last() * 100).dropna()
        return yield_current, yield_history
    return 0, pd.Series()

def get_insider_institutional(symbol):
    stock = yf.Ticker(symbol)
    return stock.institutional_holders, stock.major_holders

def get_options_chain(symbol):
    stock = yf.Ticker(symbol)
    expiration_dates = stock.options
    if expiration_dates:
        opt = stock.option_chain(expiration_dates[0])
        calls = opt.calls[['strike', 'lastPrice', 'volume', 'openInterest', 'impliedVolatility']]
        puts = opt.puts[['strike', 'lastPrice', 'volume', 'openInterest', 'impliedVolatility']]
        return calls, puts, expiration_dates[0]
    return None, None, None

def compare_peers(symbol):
    stock = yf.Ticker(symbol)
    peers = stock.info.get('relatedCompanies', [symbol, 'SPY'])[:5]
    data = yf.download(peers, period='1y')['Adj Close'].pct_change().cumsum() * 100
    return data

def backtest_sma_strategy(data, short_window, long_window):
    if data.empty:
        return pd.DataFrame()
    data['SMA_short'] = data['Close'].rolling(window=short_window).mean()
    data['SMA_long'] = data['Close'].rolling(window=long_window).mean()
    data['Signal'] = 0
    data.loc[data['SMA_short'] > data['SMA_long'], 'Signal'] = 1
    data.loc[data['SMA_short'] < data['SMA_long'], 'Signal'] = -1
    data['Position'] = data['Signal'].shift(1)
    data['Price_Change'] = data['Close'].pct_change()
    data['Strategy_Returns'] = data['Price_Change'] * data['Position']
    data['Cumulative_Returns'] = (1 + data['Strategy_Returns']).cumprod()
    return data

def create_interactive_chart(data, forecast=None):
    if data.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Price'))
    if forecast is not None:
        fig.add_trace(go.Scatter(x=forecast.index, y=forecast, name='Forecast', line=dict(color='orange', width=2)))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA'], name='20-Day Average', line=dict(color='#ff7f0e')))
    fig.update_layout(
        title='Stock Price Trend', yaxis_title='Price', template='plotly_white', 
        hovermode='x unified', height=500, margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig

# Unified AI Recommendation Function
def get_ai_recommendation(data, symbol, days_ahead=30):
    if data.empty:
        return None, None, None, None
    
    # LSTM Prediction
    close_prices = data['Close'].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_prices.reshape(-1, 1))
    look_back = 60
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    train_size = int(len(X) * 0.8)
    X_train = X[:train_size]
    y_train = y[:train_size]
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    last_sequence = scaled_data[-look_back:]
    future_predictions = []
    for _ in range(days_ahead):
        pred = model.predict(last_sequence.reshape(1, look_back, 1), verbose=0)
        future_predictions.append(pred[0, 0])
        last_sequence = np.append(last_sequence[1:], pred[0, 0])
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    forecast = pd.Series(future_predictions.flatten(), index=pd.date_range(start=data.index[-1], periods=days_ahead + 1, freq='B')[1:])
    price_trend = (forecast.iloc[-1] - data['Close'].iloc[-1]) / data['Close'].iloc[-1]

    # Stock Health Score
    technicals = calculate_technicals(data)
    latest = technicals.iloc[-1]
    income = get_financial_statements(symbol, 'income')
    pos, neg, neu = get_news_sentiment(symbol)
    features = {
        'RSI': latest['RSI'], 'MACD': latest['MACD'], 'ATR': latest['ATR'], 'Volume': latest['Volume'],
        'Sentiment_Pos': pos, 'Revenue': float(income['totalRevenue'].iloc[0]) if income is not None else 0,
        'NetIncome': float(income['netIncome'].iloc[0]) if income is not None else 0
    }
    df = pd.DataFrame([features])
    scaler = StandardScaler()
    X = scaler.fit_transform(df)
    health_score = np.clip(np.mean(X[0]) * 100 + 50, 0, 100)

    # Anomaly Detection
    df_anomaly = data[['Close']].pct_change().dropna()
    clf = IsolationForest(contamination=0.05, random_state=42)
    df_anomaly['Anomaly'] = clf.fit_predict(df_anomaly)
    anomalies = data.loc[df_anomaly[df_anomaly['Anomaly'] == -1].index]
    anomaly_factor = len(anomalies) / len(data)

    # Trend Classification
    returns = data['Close'].pct_change().dropna()
    X_trend = pd.DataFrame({
        'Returns': returns, 'Volatility': returns.rolling(20).std(), 'RSI': technicals['RSI']
    }).dropna()
    labels = np.where(X_trend['Returns'].rolling(20).mean() > 0.01, 1, np.where(X_trend['Returns'].rolling(20).mean() < -0.01, -1, 0))
    model = xgb.XGBClassifier(random_state=42)
    model.fit(X_trend.iloc[:-1], labels[1:])
    latest = X_trend.iloc[-1].values.reshape(1, -1)
    trend_pred = model.predict(latest)[0]
    trend = {1: 'Bullish', -1: 'Bearish', 0: 'Neutral'}[trend_pred]

    # Unified Recommendation
    score = (price_trend * 30) + (health_score * 0.4) - (anomaly_factor * 100) + (50 if trend == 'Bullish' else -50 if trend == 'Bearish' else 0)
    confidence = min(95, max(50, abs(score) * 0.5))
    if score > 50:
        recommendation = "Buy"
        reason = f"The stock is trending up with a strong forecast (${forecast.iloc[-1]:.2f} in {days_ahead} days) and solid fundamentals."
    elif score < -50:
        recommendation = "Sell"
        reason = f"The stock shows a downward trend and potential risks ahead."
    else:
        recommendation = "Hold"
        reason = f"The stock is stable but lacks a clear direction right now."
    
    return recommendation, confidence, reason, {'forecast': forecast, 'health_score': health_score, 'anomalies': anomalies, 'trend': trend}

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
        try:
            df = pd.DataFrame(transactions)
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
            st.error(f"GST analysis failed: {str(e)}")
            return None

    def accounting_standard_check(self, financials):
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
            st.error(f"Accounting standard analysis failed: {str(e)}")
            return None

    def inventory_audit(self, inventory_data):
        try:
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
            st.error(f"Inventory analysis failed: {str(e)}")
            return None

    def receivables_analysis(self, invoices):
        try:
            invoices['days_outstanding'] = (datetime.now() - invoices['due_date']).dt.days
            invoices['is_paid'] = invoices['paid_amount'] > 0
            kmf = KaplanMeierFitter()
            kmf.fit(invoices['days_outstanding'], invoices['is_paid'])
            payment_prob = kmf.survival_function_
            X = invoices[['amount', 'days_outstanding']]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            invoices['risk_cluster'] = KMeans(n_clusters=3).fit_predict(X_scaled)
            return {'payment_probability': payment_prob, 'invoices': invoices, 'high_risk': invoices[invoices['risk_cluster'] == 2]}
        except Exception as e:
            st.error(f"Receivables analysis failed: {str(e)}")
            return None

# ======================
# STREAMLIT UI
# ======================

def main():
    st.title("Finance Analytics Suite")
    
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "Company Overview", "Financial Statements", "Stock Chart", "Sentiment Analysis",
        "Portfolio Builder", "Auditing Tools", "Options Analysis", "Backtesting", "Smart Insights"
    ])
    
    # Sidebar Settings
    st.sidebar.header("Settings")
    symbol = st.sidebar.text_input("Stock Symbol", "AAPL").upper()
    start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
    end_date = st.sidebar.date_input("End Date", datetime.now())
    
    # Alerts System
    st.sidebar.subheader("Set Alerts")
    alert_type = st.sidebar.selectbox("Alert Type", ["Price", "News"])
    threshold = st.sidebar.number_input("Price Threshold" if alert_type == "Price" else "News Sentiment (0-1)", value=100.0 if alert_type == "Price" else 0.7)
    if st.sidebar.button("Set Alert"):
        st.session_state[f"alert_{symbol}_{alert_type}"] = threshold
        st.sidebar.success(f"Alert set for {alert_type} at {threshold}")

    # Fetch data
    data = get_stock_data(symbol, start_date, end_date)
    if not data.empty:
        data = calculate_technicals(data.dropna())
        if page not in ["Smart Insights", "Financial Statements", "Auditing Tools"]:
            st.subheader(f"{symbol} Overview")
            fig = create_interactive_chart(data)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    # Page-specific content
    if page == "Company Overview":
        st.header("🏢 Company Overview")
        info = get_company_info(symbol)
        if info:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("Key Info")
                st.write(f"**Name:** {info['name']}")
                st.write(f"**Sector:** {info['sector']}")
                st.write(f"**Industry:** {info['industry']}")
                st.write(f"**Country:** {info['country']}")
                st.write(f"**Employees:** {info['employees']:,}")
                st.write(f"**Market Cap:** ${info['market_cap']:,}" if isinstance(info['market_cap'], int) else info['market_cap'])
            with col2:
                st.subheader("About the Company")
                st.write(info['summary'])
            st.subheader("Dividends")
            yield_current, yield_history = get_dividend_analysis(symbol)
            st.write(f"**Current Yield:** {yield_current:.2f}%")
            if not yield_history.empty:
                fig = px.line(yield_history, title="Dividend Yield Over Time")
                st.plotly_chart(fig)

    elif page == "Financial Statements":
        st.header("📑 Financial Statements")
        tab1, tab2, tab3 = st.tabs(["Income", "Balance Sheet", "Cash Flow"])
        with tab1:
            income_stmt = get_financial_statements(symbol, 'income')
            if income_stmt is not None:
                st.subheader("Income Statement")
                numeric_cols = income_stmt.select_dtypes(include=[np.number]).columns
                format_dict = {col: "{:,.0f}" for col in numeric_cols}
                st.dataframe(income_stmt.style.format(format_dict))
        with tab2:
            balance_sheet = get_financial_statements(symbol, 'balance')
            if balance_sheet is not None:
                st.subheader("Balance Sheet")
                numeric_cols = balance_sheet.select_dtypes(include=[np.number]).columns
                format_dict = {col: "{:,.0f}" for col in numeric_cols}
                st.dataframe(balance_sheet.style.format(format_dict))
        with tab3:
            cash_flow = get_financial_statements(symbol, 'cashflow')
            if cash_flow is not None:
                st.subheader("Cash Flow")
                numeric_cols = cash_flow.select_dtypes(include=[np.number]).columns
                format_dict = {col: "{:,.0f}" for col in numeric_cols}
                st.dataframe(cash_flow.style.format(format_dict))

    elif page == "Stock Chart":
        st.header("📈 Stock Chart")
        if not data.empty:
            if f"alert_{symbol}_Price" in st.session_state and data['Close'].iloc[-1] >= st.session_state[f"alert_{symbol}_Price"]:
                st.warning(f"Price Alert: {symbol} hit ${data['Close'].iloc[-1]:.2f}!")
            st.subheader("Key Metrics")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Latest Price:** ${data['Close'].iloc[-1]:.2f}")
                st.write(f"**20-Day Average:** ${data['SMA'].iloc[-1]:.2f}")
            with col2:
                st.write(f"**Volatility (ATR):** {data['ATR'].iloc[-1]:.2f}")
                st.write(f"**Momentum (RSI):** {data['RSI'].iloc[-1]:.2f}")

    elif page == "Sentiment Analysis":
        st.header("😊 Market Sentiment")
        st.subheader("News Mood")
        positive, negative, neutral = get_news_sentiment(symbol)
        if f"alert_{symbol}_News" in st.session_state and positive >= st.session_state[f"alert_{symbol}_News"]:
            st.warning(f"News Alert: Positive sentiment hit {positive:.2f}!")
        cols = st.columns(3)
        cols[0].metric("Positive", f"{positive * 100:.1f}%")
        cols[1].metric("Negative", f"{negative * 100:.1f}%")
        cols[2].metric("Neutral", f"{neutral * 100:.1f}%")
        fig = go.Figure(go.Bar(x=['Positive', 'Negative', 'Neutral'], y=[positive, negative, neutral], 
                              marker_color=['#4CAF50', '#F44336', '#9E9E9E']))
        fig.update_layout(title="News Sentiment Breakdown", yaxis_title="Percentage", height=400)
        st.plotly_chart(fig)

    elif page == "Portfolio Builder":
        st.header("💼 My Portfolio")
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = {}
        st.subheader("Add a Stock")
        col1, col2 = st.columns(2)
        with col1:
            new_symbol = st.text_input("Stock Symbol", "").upper()
        with col2:
            shares = st.number_input("Shares", min_value=1, value=100)
        if st.button("Add Stock") and new_symbol:
            if new_symbol in st.session_state.portfolio:
                st.session_state.portfolio[new_symbol] += shares
            else:
                st.session_state.portfolio[new_symbol] = shares
            st.success(f"Added {shares} shares of {new_symbol}")
        if st.session_state.portfolio:
            st.subheader("Portfolio Summary")
            portfolio_data = []
            total_value = 0
            for symbol, shares in st.session_state.portfolio.items():
                try:
                    stock = yf.Ticker(symbol)
                    current_price = stock.history(period='1d')['Close'].iloc[-1]
                    value = current_price * shares
                    total_value += value
                    daily_change = (current_price - stock.history(period='2d')['Close'].iloc[0]) / stock.history(period='2d')['Close'].iloc[0]
                    portfolio_data.append({'Symbol': symbol, 'Shares': shares, 'Price': current_price, 'Value': value, 'Change': daily_change})
                except:
                    portfolio_data.append({'Symbol': symbol, 'Shares': shares, 'Price': 'N/A', 'Value': 'N/A', 'Change': 'N/A'})
            df = pd.DataFrame(portfolio_data)
            styled_df = df.style.format({'Price': '${:,.2f}', 'Value': '${:,.2f}', 'Change': '{:.2%}'}, na_rep='N/A')
            st.dataframe(styled_df)
            st.write(f"**Total Value:** ${total_value:,.2f}")
            if total_value > 0:
                fig = px.pie(df[df['Value'] != 'N/A'], values='Value', names='Symbol', title="Portfolio Breakdown")
                st.plotly_chart(fig)

    elif page == "Auditing Tools":
        st.header("🔍 Auditing Tools")
        auditor = AdvancedAuditTools()
        audit_function = st.selectbox("Choose Tool", ["GST Compliance", "Accounting Standards", "Inventory Check", "Receivables Aging"])
        if audit_function == "GST Compliance":
            st.subheader("💰 GST Compliance")
            dates = pd.date_range('2023-01-01', periods=100)
            amounts = np.random.lognormal(mean=8, sigma=1.5, size=100)
            vendors = np.random.choice(['Vendor A', 'Vendor B', 'Vendor C', 'Vendor D'], 100)
            transactions = pd.DataFrame({'date': dates, 'amount': amounts, 'vendor': vendors})
            if st.button("Check Compliance"):
                with st.spinner("Analyzing..."):
                    result = auditor.gst_compliance_check(transactions)
                    if result is not None:
                        st.success("Done!")
                        st.write(f"**Total GST:** ${result['gst_amount'].sum():,.2f}")
                        st.write(f"**Anomalies:** {len(result[result['anomaly_score'] == -1])} transactions")
                        fig = px.scatter(result, x='date', y='amount', color='anomaly_score', color_continuous_scale=['blue', 'red'], title="Transaction Check")
                        st.plotly_chart(fig)
        elif audit_function == "Accounting Standards":
            st.subheader("🌐 Accounting Standards")
            companies = [f"Company {chr(65 + i)}" for i in range(10)]
            financials = pd.DataFrame({'company': companies, 'revenue_growth': np.random.normal(0.08, 0.03, 10), 'asset_turnover': np.random.uniform(0.5, 1.5, 10), 'debt_ratio': np.random.uniform(0.2, 0.7, 10)})
            if st.button("Classify"):
                with st.spinner("Analyzing..."):
                    result = auditor.accounting_standard_check(financials)
                    st.dataframe(result)
        elif audit_function == "Inventory Check":
            st.subheader("📦 Inventory Check")
            dates = pd.date_range('2022-01-01', '2023-01-01', freq='D')
            quantities = np.maximum(10, 100 + 20 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365) + np.random.normal(0, 5, len(dates))).astype(int)
            quantities[30] = 200
            quantities[150:155] = 10
            inventory_data = pd.DataFrame({'date': dates, 'quantity': quantities})
            if st.button("Analyze Inventory"):
                with st.spinner("Analyzing..."):
                    result = auditor.inventory_audit(inventory_data)
                    if result is not None:
                        fig = px.line(x=result['actual'].index, y=result['actual'], title="Inventory Levels")
                        fig.add_scatter(x=result['actual'].index[result['anomalies'] == 1], y=result['actual'][result['anomalies'] == 1], mode='markers', name='Anomalies', marker=dict(color='red'))
                        st.plotly_chart(fig)
        elif audit_function == "Receivables Aging":
            st.subheader("⏳ Receivables Aging")
            due_dates = pd.date_range('2022-01-01', periods=50, freq='7D')
            paid_dates = [d + timedelta(days=np.random.randint(0, 90)) if np.random.random() > 0.3 else None for d in due_dates]
            invoices = pd.DataFrame({
                'invoice_id': range(1001, 1051), 'due_date': due_dates, 'paid_date': paid_dates,
                'amount': np.random.uniform(100, 5000, 50),
                'paid_amount': [a if pd.notnull(d) else 0 for a, d in zip(np.random.uniform(100, 5000, 50), paid_dates)],
                'customer': np.random.choice(['Customer A', 'Customer B', 'Customer C'], 50)
            })
            if st.button("Analyze Receivables"):
                with st.spinner("Analyzing..."):
                    result = auditor.receivables_analysis(invoices)
                    if result is not None:
                        fig = px.line(result['payment_probability'], title="Payment Likelihood")
                        st.plotly_chart(fig)
                        st.write("**High-Risk Receivables:**")
                        st.dataframe(result['high_risk'])

    elif page == "Options Analysis":
        st.header("📊 Options Analysis")
        calls, puts, exp_date = get_options_chain(symbol)
        if calls is not None:
            st.subheader(f"Calls (Expires: {exp_date})")
            st.dataframe(calls)
            st.subheader(f"Puts (Expires: {exp_date})")
            st.dataframe(puts)
            fig = px.scatter(calls, x='strike', y='impliedVolatility', title="Call Options Volatility")
            st.plotly_chart(fig)

    elif page == "Backtesting":
        st.header("🔄 Backtesting")
        period_options = {'1 Year': '1y', '2 Years': '2y', '5 Years': '5y'}
        selected_period = st.selectbox("Time Period", list(period_options.keys()))
        data = get_stock_data(symbol, period=period_options[selected_period])
        if not data.empty:
            st.subheader("Moving Average Strategy")
            short_w = st.slider("Short Window", 10, 100, 50)
            long_w = st.slider("Long Window", 50, 300, 200)
            if st.button("Test Strategy"):
                results = backtest_sma_strategy(data, short_w, long_w)
                if not results.empty:
                    final_return = (results['Cumulative_Returns'].iloc[-1] - 1) * 100
                    st.write(f"**Strategy Gain:** {final_return:.2f}%")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=results.index, y=results['Cumulative_Returns'], name='Strategy'))
                    fig.add_trace(go.Scatter(x=results.index, y=results['Close'] / results['Close'].iloc[0], name='Hold'))
                    fig.update_layout(title="Strategy vs. Hold", yaxis_title="Growth")
                    st.plotly_chart(fig)

    elif page == "Smart Insights":
        st.header("🤖 Smart Insights")
        if not data.empty:
            st.subheader(f"What to Do with {symbol}")
            days_ahead = st.slider("Look Ahead (Days)", 5, 90, 30)
            if st.button("Get Insight"):
                with st.spinner("Thinking..."):
                    recommendation, confidence, reason, details = get_ai_recommendation(data, symbol, days_ahead)
                    if recommendation:
                        color = '#4CAF50' if recommendation == "Buy" else '#F44336' if recommendation == "Sell" else '#9E9E9E'
                        st.markdown(f"<h2 style='color:{color}'>{recommendation}</h2>", unsafe_allow_html=True)
                        st.write(f"**Confidence:** {confidence:.0f}%")
                        st.write(f"**Why:** {reason}")
                        with st.expander("More Details"):
                            st.write(f"**Forecasted Price:** ${details['forecast'].iloc[-1]:.2f} in {days_ahead} days")
                            st.write(f"**Health Score:** {details['health_score']:.0f}/100")
                            st.write(f"**Trend:** {details['trend']}")
                            st.write(f"**Unusual Moves:** {len(details['anomalies'])} detected")
                            fig = create_interactive_chart(data[-60:], details['forecast'])
                            fig.update_layout(title=f"{symbol} Forecast")
                            st.plotly_chart(fig)

    st.sidebar.markdown("---")
    st.sidebar.info("Finance Analytics Suite v3.0\n\nPowered by Yahoo Finance & Alpha Vantage\nFor demo purposes only.")

if __name__ == "__main__":
    main()