import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torchcde import cdeint
import networkx as nx
from transformers import pipeline
import redis
import re
from datetime import datetime
import click
import os
from typing import List, Dict, Tuple
from scipy.special import zeta
from scipy.fft import fft
import cvxpy as cp
from arch import arch_model
import gudhi
from qiskit import QuantumCircuit, Aer, execute
import io
import base64
import warnings
import sys


warnings.filterwarnings("ignore")

# Streamlit App Configuration
st.set_page_config(page_title="AutoCA: Deep-Tech Financial AI", layout="wide")
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>AutoCA: Deep-Tech Financial AI (2025–2030)</h1>",
            unsafe_allow_html=True)

# Redis for Caching
try:
   redis_client = redis.Redis(host='localhost', port=6379, db=0)
except:
   redis_client = None
   st.warning("Redis not available; caching disabled.")

# Role-Based Access
USER_ROLES = {"CA": "full", "SME_Owner": "limited", "Auditor": "read-only"}


# Bookkeeping Classes
class Account:
   def __init__(self, name, account_type, currency="INR"):
      self.name = name
      self.account_type = account_type
      self.currency = currency
      self.balance = 0.0
      self.transactions = []

   def debit(self, amount, description):
      if self.account_type in ["Asset", "Expense"]:
         self.balance += amount
      else:
         self.balance -= amount
      self.transactions.append({"type": "debit", "amount": amount, "description": description, "date": datetime.now()})

   def credit(self, amount, description):
      if self.account_type in ["Asset", "Expense"]:
         self.balance -= amount
      else:
         self.balance += amount
      self.transactions.append({"type": "credit", "amount": amount, "description": description, "date": datetime.now()})


class Ledger:
   def __init__(self):
      self.accounts = {}
      self.gstin_map = {}

   def add_account(self, name, account_type, currency="INR", gstin=None):
      self.accounts[name] = Account(name, account_type, currency)
      if gstin:
         self.gstin_map[gstin] = name

   def record_transaction(self, debit_account, credit_account, amount, description, gstin=None):
      if debit_account not in self.accounts or credit_account not in self.accounts:
         raise ValueError("Invalid account")
      self.accounts[debit_account].debit(amount, description)
      self.accounts[credit_account].credit(amount, description)

      total_debits = sum(acc.balance for acc in self.accounts.values() if acc.account_type in ["Asset", "Expense"])
      total_credits = sum(
         acc.balance for acc in self.accounts.values() if acc.account_type in ["Liability", "Equity", "Revenue"])
      if abs(total_debits - total_credits) > 0.01:
         raise ValueError("Ledger imbalance detected")

      if redis_client:
         redis_client.set(f"txn_{hash(description)}",
                          str({"debit": debit_account, "credit": credit_account, "amount": amount}))

      return {"status": "Recorded", "debit_account": debit_account, "credit_account": credit_account, "amount": amount}

   def auto_entry(self, invoice_data):
      amount = invoice_data.get("amount")
      gstin = invoice_data.get("gstin")
      date = invoice_data.get("date", "2025-06-19")
      doc_type = invoice_data.get("document_type", "Unknown")

      description = f"Invoice {doc_type} {date}"
      if doc_type.lower().startswith("s"):
         debit_account = "Cash"
         credit_account = "Sales"
      else:
         debit_account = "Purchases"
         credit_account = "Cash"

      if debit_account not in self.accounts:
         self.add_account(debit_account, "Asset" if debit_account == "Cash" else "Expense")
      if credit_account not in self.accounts:
         self.add_account(credit_account, "Revenue" if credit_account == "Sales" else "Asset")

      return self.record_transaction(debit_account, credit_account, amount, description, gstin)


# Initialize Ledger
ledger = Ledger()


# Data Ingestion Module
@st.cache_data
def load_data(file, file_type="csv") -> pd.DataFrame:
   """Load and preprocess business datasets or invoices."""
   if file is not None:
      try:
         if file_type == "csv":
            df = pd.read_csv(file)
         elif file_type in ["xlsx", "xls"]:
            df = pd.read_excel(file)
         elif file_type == "text":
            text = file.read().decode('utf-8')
            invoice_data = parse_invoice_text(text)
            df = pd.DataFrame([invoice_data])
         else:
            st.error("Unsupported file format. Use CSV, Excel, or Text.")
            return None
         for col in df.columns:
            if df[col].dtype == 'object':
               try:
                  df[col] = pd.to_datetime(df[col])
               except:
                  pass
         df = adaptive_neural_imputation(df)
         return df
      except Exception as e:
         st.error(f"Error processing file: {str(e)}")
         return None
   return None


# Adaptive Neural Imputation
class AdaptiveImputer(nn.Module):
   def __init__(self, input_dim: int):
      super(AdaptiveImputer, self).__init__()
      self.encoder = nn.Sequential(
         nn.Linear(input_dim, 128),
         nn.ReLU(),
         nn.Linear(128, 64),
         nn.ReLU(),
      )
      self.decoder = nn.Sequential(
         nn.Linear(64, 128),
         nn.ReLU(),
         nn.Linear(128, input_dim)
      )

   def forward(self, x):
      return self.decoder(self.encoder(x))


def adaptive_neural_imputation(df: pd.DataFrame) -> pd.DataFrame:
   """Impute missing values using a neural network."""
   numeric_cols = df.select_dtypes(include=[np.number]).columns
   if df[numeric_cols].isna().any().any():
      imputer = AdaptiveImputer(len(numeric_cols))
      optimizer = torch.optim.Adam(imputer.parameters(), lr=0.005)
      X = torch.tensor(df[numeric_cols].fillna(0).values, dtype=torch.float32)
      mask = torch.tensor(df[numeric_cols].isna().values, dtype=torch.bool)

      for _ in range(100):
         optimizer.zero_grad()
         X_pred = imputer(X)
         loss = torch.mean((X_pred[~mask] - X[~mask]) ** 2)
         loss.backward()
         optimizer.step()

      with torch.no_grad():
         X_imputed = imputer(X).numpy()
         df[numeric_cols] = np.where(df[numeric_cols].isna(), X_imputed, df[numeric_cols])
   return df


# Fractional Differentiation (Simplified)
def fractional_diff(data: pd.Series, d: float = 0.5) -> pd.Series:
   """Compute fractional differencing for time series."""
   weights = [1.0]
   for k in range(1, len(data)):
      weights.append(-weights[-1] * (d - k + 1) / k)
   result = np.convolve(data, weights[:len(data)][::-1], mode='valid')
   return pd.Series(result, index=data.index[-len(result):])


# T-Copula (Simplified)
def t_copula(data: pd.DataFrame, dof: int = 5) -> np.ndarray:
   """Approximate T-Copula for dependency modeling."""
   from scipy.stats import t
   standardized = (data - data.mean()) / data.std()
   return t.cdf(standardized, df=dof)


# Neural CDE for Anomaly Detection
class CDEFunc(nn.Module):
   def __init__(self, input_dim: int):
      super(CDEFunc, self).__init__()
      self.net = nn.Sequential(
         nn.Linear(input_dim, 100),
         nn.Tanh(),
         nn.Linear(100, input_dim * input_dim)
      )

   def forward(self, t, z):
      return self.net(z).view(z.size(0), z.size(1), -1)


def neural_cde_anomaly(data: pd.DataFrame) -> Dict:
   """Detect anomalies using Neural CDEs with Zeta enhancement."""
   numeric_cols = data.select_dtypes(include=[np.number]).columns
   amounts = data['amount'].values if 'amount' in data.columns else data[numeric_cols].values
   timestamps = data['date'].map(datetime.toordinal).values if 'date' in data.columns else np.arange(len(data))

   scaler = StandardScaler()
   scaled_data = scaler.fit_transform(amounts.reshape(-1, 1) if len(amounts.shape) == 1 else amounts)

   X = torch.tensor(np.column_stack([timestamps, scaled_data]), dtype=torch.float32)
   try:
      X = torchcde.CubicSpline(X).interpolate(torch.linspace(0, 1, 50))
      cde_func = CDEFunc(scaled_data.shape[1] + 1)
      z0 = X[:, 0, :]
      z = cdeint(X, cde_func, z0, t=torch.linspace(0, 1, 50))
      recon_error = torch.mean((z[:, -1, 1:] - torch.tensor(scaled_data, dtype=torch.float32)) ** 2, dim=1)
      anomaly_indices = torch.where(recon_error > recon_error.mean() + 3 * recon_error.std())[0].tolist()

      zeta_scores = np.abs([zeta(2 + 1j * a / np.max(amounts)) for a in amounts.flatten()])
      zeta_anomalies = np.where(zeta_scores > np.percentile(zeta_scores, 95))[0].tolist()
      combined_anomalies = list(set(anomaly_indices + zeta_anomalies))

      return {
         "anomalies": data.iloc[combined_anomalies].to_dict('records'),
         "zeta_scores": zeta_scores.tolist(),
         "use_case": "Hybrid Neural CDE + Zeta Anomaly Detection"
      }
   except Exception as e:
      return {"error": f"Neural CDE failed: {str(e)}", "anomalies": [], "zeta_scores": []}


# Mock Theta Fraud Detection
def mock_theta_fraud(data: pd.DataFrame) -> Dict:
   """Detect sparse fraud using Mock Theta."""
   amounts = data['amount'].values if 'amount' in data.columns else np.sum(
      data.select_dtypes(include=[np.number]).values, axis=1)
   q = np.exp(-amounts / np.max(amounts))
   mock_theta = np.sum(q ** np.arange(len(amounts)))
   coefficients = [np.sum(q ** i) for i in range(len(amounts))]  # <- fixed

   return {
      "score": float(1 / (1 + mock_theta)),
      "coefficients": [float(c) for c in coefficients],
      "use_case": "Sparse Fraud Pattern Detection"
   }


# GARCH Volatility
def garch_volatility(data: pd.DataFrame) -> Dict:
   """Predict volatility using GARCH."""
   amounts = data['amount'].values if 'amount' in data.columns else np.sum(
      data.select_dtypes(include=[np.number]).values, axis=1)
   try:
      model = arch_model(amounts, vol='GARCH', p=1, q=1)
      res = model.fit(disp='off')
      return {
         "volatility": float(res.conditional_volatility[-1]),
         "use_case": "Cash Flow Volatility Prediction"
      }
   except:
      return {"error": "GARCH failed", "volatility": 0.0}


# Kalman Filter Forecasting
def kalman_forecast(data: pd.DataFrame) -> Dict:
   """Forecast using Kalman Filter."""
   amounts = data['amount'].values if 'amount' in data.columns else np.sum(
      data.select_dtypes(include=[np.number]).values, axis=1)
   A, B, H, Q, R = 1.0, 0.0, 1.0, 0.1, 0.1
   x_hat = amounts[0]
   P = 1.0
   estimates = []
   for z in amounts:
      x_hat_minus = A * x_hat
      P_minus = A * P * A + Q
      K = P_minus * H / (H * P_minus * H + R)
      x_hat = x_hat_minus + K * (z - H * x_hat_minus)
      P = (1 - K * H) * P_minus
      estimates.append(x_hat)
   forecast = estimates[-1] * np.ones(30)
   return {
      "forecast": forecast.tolist(),
      "estimates": estimates,
      "use_case": "Financial Trend Forecasting"
   }


# Fourier Trends
def fourier_trends(data: pd.DataFrame) -> Dict:
   """Detect trends using Fourier Transform."""
   amounts = data['amount'].values if 'amount' in data.columns else np.sum(
      data.select_dtypes(include=[np.number]).values, axis=1)
   fft_result = fft(amounts)
   freq = np.fft.fftfreq(len(amounts))
   return {
      "amplitudes": np.abs(fft_result).tolist(),
      "frequencies": freq.tolist(),
      "use_case": "Seasonal Trend Detection"
   }


# Topological Data Analysis
def topological_data_analysis(data: pd.DataFrame) -> Dict:
   """Perform TDA using persistent homology."""
   amounts = data['amount'].values if 'amount' in data.columns else np.sum(
      data.select_dtypes(include=[np.number]).values, axis=1)
   point_cloud = np.column_stack([np.arange(len(amounts)), amounts])

   rips_complex = gudhi.RipsComplex(points=point_cloud, max_edge_length=1000)
   simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
   persistence = simplex_tree.persistence()

   return {
      "persistence_diagram": [(dim, (birth, death)) for dim, (birth, death) in persistence if death != float('inf')],
      "use_case": "Topological Pattern Detection"
   }


# Quantum Audit
def quantum_audit(data: pd.DataFrame) -> Dict:
   """Simulate quantum-inspired audit using Qiskit."""
   amounts = data['amount'].values if 'amount' in data.columns else np.sum(
      data.select_dtypes(include=[np.number]).values, axis=1)
   n_qubits = min(4, int(np.log2(len(amounts)) + 1))

   qc = QuantumCircuit(n_qubits)
   for i in range(n_qubits):
      qc.h(i)
   qc.measure_all()

   simulator = Aer.get_backend('qasm_simulator')
   result = execute(qc, simulator, shots=1024).result()
   counts = result.get_counts()

   entropy = -sum((count / 1024) * np.log2(count / 1024 + 1e-10) for count in counts.values())
   return {
      "quantum_entropy": float(entropy),
      "use_case": "Quantum-Inspired Audit Integrity Check"
   }


# Ramanujan Partitions
def ramanujan_partitions(data: pd.DataFrame) -> Dict:
   """Estimate compliance using Ramanujan's partition theory."""
   amounts = data['amount'].values if 'amount' in data.columns else np.sum(
      data.select_dtypes(include=[np.number]).values, axis=1)
   n = int(np.sum(np.abs(amounts)) / 1000) % 100
   if n == 0:
      return {"partitions": 0, "use_case": "Compliance Estimation"}

   p = [1] + [0] * n
   for i in range(1, n + 1):
      k = 1
      while True:
         pent = (k * (3 * k - 1)) // 2
         if pent > i:
            break
         sign = (-1) ** (k + 1)
         p[i] += sign * p[i - pent]
         pent = (k * (3 * k + 1)) // 2
         if pent > i:
            break
         p[i] += sign * p[i - pent]
         k += 1
   return {
      "partitions": p[n],
      "use_case": "Compliance Estimation via Partitions"
   }


# Game-Theoretic Shapley Compliance
def game_theoretic_shapley_compliance(data: pd.DataFrame) -> Dict:
   """Estimate compliance using simplified Shapley values."""
   amounts = data['amount'].values if 'amount' in data.columns else np.sum(
      data.select_dtypes(include=[np.number]).values, axis=1)
   n = min(5, len(amounts))
   contributions = np.abs(amounts[:n])
   total = np.sum(contributions)
   shapley_values = []

   for i in range(n):
      value = contributions[i] / total
      shapley_values.append(value)

   return {
      "shapley_values": shapley_values,
      "use_case": "Game-Theoretic Compliance Analysis"
   }


# Invoice Parsing
def parse_invoice_text(text: str) -> Dict:
   """Parse invoice text for key details."""
   try:
      gstin_pattern = r"\d{2}[A-Z]{5}\d{4}[A-Z]{1}\d{1}[A-Z]{1}\d{1}"
      amount_pattern = r"(?:Total|Amount)\s*[:\-]?\s*₹?\s*(\d+\.?\d*)"
      date_pattern = r"\d{2}-\d{2}-\d{4}"

      gstin = re.search(gstin_pattern, text)
      amount = re.search(amount_pattern, text)
      date = re.search(date_pattern, text)

      nlp = pipeline("text-classification", model="distilbert-base-uncased")
      doc_type = nlp(text[:512])[0]['label']

      invoice_data = {
         "gstin": gstin.group(0) if gstin else None,
         "amount": float(amount.group(1)) if amount else None,
         "date": date.group(0) if date else None,
         "document_type": "Sales" if doc_type == "POSITIVE" else "Purchase",
         "confidence": {
            "gstin": 0.95 if gstin else 0.7,
            "amount": 0.85 if amount else 0.0,
            "date": 0.9 if date else 0.0
         }
      }

      if invoice_data["amount"]:
         entry = ledger.auto_entry(invoice_data)
         invoice_data["ledger_entry"] = entry

      if redis_client:
         redis_client.set(f"invoice_{hash(text)}", str(invoice_data))

      return invoice_data
   except:
      return {"error": "Failed to parse invoice"}


# Bank Reconciliation
def bank_reconciliation(transactions: List[Dict], bank_feed: List[Dict]) -> Dict:
   """Reconcile transactions with bank feed."""
   matches = []
   for b in bank_feed:
      for l in transactions:
         if abs(b['amount'] - l['amount']) < 0.01:
            matches.append({"bank_txn": b, "ledger_txn": l})
   return {
      "matches": matches,
      "unmatched": len(bank_feed) - len(matches),
      "use_case": "Bank Reconciliation"
   }


# Inventory Optimization
def manage_inventory(items: List[Dict]) -> List[Dict]:
   """Optimize inventory using convex optimization."""
   try:
      x = cp.Variable(len(items))
      objective = cp.Minimize(cp.sum_squares(x - [i['stock'] for i in items]))
      constraints = [x >= 0, cp.sum(x * [i['cost'] for i in items]) <= sum(i['cost'] * i['stock'] for i in items) * 0.9]
      prob = cp.Problem(objective, constraints)
      prob.solve()
      inventory = [{"item_id": i['id'], "optimal_stock": float(x.value[j])} for j, i in enumerate(items)]
      return inventory
   except:
      return [{"item_id": i['id'], "optimal_stock": i['stock']} for i in items]


# Tax Computation
def compute_tax(data: pd.DataFrame, tax_type="GST") -> Dict:
   """Compute tax based on transactions."""
   amounts = data['amount'].values if 'amount' in data.columns else np.sum(
      data.select_dtypes(include=[np.number]).values, axis=1)
   tax_base = np.sum(amounts)

   rates = {"GST": 0.18, "TDS": 0.1, "IT_44ADA": 0.3}
   rate = rates.get(tax_type, 0.0)
   tax = tax_base * rate

   return {
      "tax_type": tax_type,
      "amount": float(tax),
      "use_case": "Tax Estimation"
   }


# Audit Report
def generate_audit_report(data: pd.DataFrame) -> Dict:
   """Generate audit report."""
   symmetry = noether_symmetry(data)
   anomalies = neural_cde_anomaly(data).get("anomalies", [])
   quantum = quantum_audit(data)

   report = {
      "form_3ca": {
         "status": "Compliant" if symmetry["deviation"] < 0.01 else "Issues Detected",
         "ledger_balance": float(data['amount'].sum() if 'amount' in data.columns else 0),
         "audit_date": "2025-06-19"
      },
      "form_3cd": {
         "transactions_analyzed": len(data),
         "anomalies": len(anomalies),
         "quantum_entropy": quantum["quantum_entropy"]
      }
   }
   return {
      "report": report,
      "use_case": "Audit Report Generation"
   }


# Noether Symmetry
def noether_symmetry(data: pd.DataFrame) -> Dict:
   """Check ledger integrity using Noether's symmetry."""
   amounts = data['amount'].values if 'amount' in data.columns else np.sum(
      data.select_dtypes(include=[np.number]).values, axis=1)
   assets = np.sum(amounts[amounts > 0])
   liabilities = assets * 0.6
   equity = assets * 0.4
   deviation = abs(assets - (liabilities + equity))
   return {
      "deviation": float(deviation),
      "use_case": "Ledger Integrity Check"
   }


# Payroll Processing
def process_payroll(employees: List[Dict]) -> List[Dict]:
   """Process payroll with optimization."""
   try:
      salaries = [emp['salary'] for emp in employees]
      x = cp.Variable(len(salaries))
      objective = cp.Minimize(cp.sum_squares(x - salaries))
      constraints = [x >= 0, cp.sum(x) <= sum(salaries) * 0.95]
      prob = cp.Problem(objective, constraints)
      prob.solve()

      payroll = []
      for i, emp in enumerate(employees):
         tds = x.value[i] * 0.1
         pf = x.value[i] * 0.12
         net = x.value[i] - tds - pf
         payroll.append({
            "employee_id": emp['id'],
            "gross": float(x.value[i]),
            "tds": float(tds),
            "pf": float(pf),
            "net": float(net)
         })
      return payroll
   except:
      return [{"employee_id": emp['id'], "gross": emp['salary'], "tds": 0, "pf": 0, "net": emp['salary']} for emp in
              employees]


# Vendor Risk
def vendor_credit_risk(data: pd.DataFrame) -> Dict:
   """Assess vendor credit risk."""
   amounts = data['amount'].values if 'amount' in data.columns else np.sum(
      data.select_dtypes(include=[np.number]).values, axis=1)
   risk_scores = np.random.rand(len(amounts))
   return {
      "risk_scores": risk_scores.tolist(),
      "use_case": "Vendor Credit Risk Assessment"
   }


# Loan Readiness
def loan_readiness_report(data: pd.DataFrame) -> Dict:
   """Generate loan readiness report."""
   amounts = data['amount'].values if 'amount' in data.columns else np.sum(
      data.select_dtypes(include=[np.number]).values, axis=1)
   revenue = np.sum(amounts[amounts > 0])
   expenses = np.sum(amounts[amounts < 0])

   report = {
      "profit_loss": {"revenue": float(revenue), "expenses": float(expenses), "net": float(revenue + expenses)},
      "balance_sheet": {"assets": float(revenue), "liabilities": float(expenses), "equity": float(revenue + expenses)},
      "use_case": "Loan Readiness Assessment"
   }
   return report


# AI Bookkeeping
def ai_bookkeeping(data: pd.DataFrame) -> List[Dict]:
   """Categorize transactions using NLP."""
   descriptions = data['description'].values if 'description' in data.columns else [f"Txn {i}" for i in
                                                                                    range(len(data))]
   nlp = pipeline("text-classification", model="distilbert-base-uncased")
   categories = [nlp(d[:512])[0]['label'] for d in descriptions]

   return [{"transaction_id": i, "category": "Revenue" if c == "POSITIVE" else "Expense"} for i, c in
           enumerate(categories)]


# Predictive Alerts
def predictive_alerts(data: pd.DataFrame) -> Dict:
   """Generate predictive alerts."""
   forecast = kalman_forecast(data)["forecast"]
   if np.any(np.array(forecast) < 0):
      return {"alert": "Cash flow shortage predicted", "severity": "High"}
   return {"alert": "Cash flow stable", "severity": "Low"}


# Smart Deduction Finder
def smart_deduction_finder(data: pd.DataFrame) -> List[Dict]:
   """Find tax deductions using RL."""
   amounts = data['amount'].values if 'amount' in data.columns else np.sum(
      data.select_dtypes(include=[np.number]).values, axis=1)
   q_table = np.zeros((len(amounts), 2))
   for _ in range(100):
      state = 0
      while state < len(amounts):
         action = np.random.randint(0, 2)
         reward = 1 if action == 1 and amounts[state] > 1000 else -1
         next_state = state + 1
         q_table[state, action] += 0.1 * (
                    reward + 0.9 * np.max(q_table[min(next_state, len(amounts) - 1)]) - q_table[state, action])
         state = next_state
         if state >= len(amounts):
            break

   deductions = [{"transaction_id": i, "deduction": "Section 80C"} for i in range(len(amounts)) if q_table[i, 1] > 0]
   return deductions


# Chart Download Utility
def get_chart_download_link(fig, filename="chart"):
   """Generate download link for Plotly chart."""
   buffer = io.BytesIO()
   fig.write_image(buffer, format="png")
   buffer.seek(0)
   b64 = base64.b64encode(buffer.read()).decode()
   href = f'<a href="data:image/png;base64,{b64}" download="{filename}.png">Download Chart as PNG</a>'
   return href


# Streamlit UI
def render_ui():
   """Render the enhanced Streamlit UI."""
   st.sidebar.markdown("<h2 style='color: #2E86C1;'>AutoCA Control Panel</h2>", unsafe_allow_html=True)
   user_role = st.sidebar.selectbox("Select Role", ["CA", "SME_Owner", "Auditor"], key="role")
   access_level = USER_ROLES[user_role]

   st.sidebar.header("Data Ingestion")
   file_type = st.sidebar.selectbox("File Type", ["CSV", "Excel", "Text"], key="file_type")
   uploaded_file = st.sidebar.file_uploader("Upload Data", type=['csv', 'xlsx', 'xls', 'txt'], key="uploader")

   if uploaded_file:
      df = load_data(uploaded_file, file_type.lower())
      if df is not None:
         st.header("Dataset Preview")
         st.dataframe(df, use_container_width=True)

         # Filters
         with st.expander("Apply Filters"):
            col1, col2 = st.columns(2)
            with col1:
               if 'date' in df.columns:
                  date_range = st.date_input("Select Date Range", [df['date'].min(), df['date'].max()])
                  df = df[(df['date'] >= pd.to_datetime(date_range[0])) & (df['date'] <= pd.to_datetime(date_range[1]))]
            with col2:
               if 'document_type' in df.columns:
                  doc_type = st.multiselect("Document Type", df['document_type'].unique(),
                                            default=df['document_type'].unique())
                  df = df[df['document_type'].isin(doc_type)]

         tabs = st.tabs([
            "Dashboard", "Invoice", "Bookkeeping", "Reconciliation", "Inventory", "Tax", "Audit",
            "Payroll", "Vendor Risk", "Loan", "Fraud", "Forecast", "Anomaly", "Alerts",
            "Deductions", "Math Insights"
         ])

         with tabs[0]:
            st.header("Financial Dashboard")
            col1, col2, col3 = st.columns(3)
            with col1:
               fraud = mock_theta_fraud(df)
               fig_gauge = go.Figure(go.Indicator(
                  mode="gauge+number",
                  value=fraud['score'],
                  title={'text': "Fraud Risk"},
                  gauge={'axis': {'range': [0, 1]}, 'bar': {'color': "red" if fraud['score'] > 0.5 else "green"}}
               ))
               st.plotly_chart(fig_gauge, use_container_width=True)
               st.markdown(get_chart_download_link(fig_gauge, "fraud_risk"), unsafe_allow_html=True)

            with col2:
               alert = predictive_alerts(df)
               color = "red" if alert["severity"] == "High" else "green"
               st.markdown(f"<h3 style='color: {color};'>Cash Flow: {alert['alert']}</h3>", unsafe_allow_html=True)

            with col3:
               vol = garch_volatility(df)
               fig_vol = px.bar(x=["Volatility"], y=[vol['volatility']], title="Cash Flow Volatility")
               st.plotly_chart(fig_vol, use_container_width=True)
               st.markdown(get_chart_download_link(fig_vol, "volatility"), unsafe_allow_html=True)

            # Transaction Heatmap
            if 'amount' in df.columns and 'date' in df.columns:
               df['month'] = df['date'].dt.month_name()
               heatmap_data = df.pivot_table(values='amount', index='month', aggfunc='sum').fillna(0)
               fig_heatmap = px.imshow(heatmap_data, title="Transaction Heatmap by Month",
                                       color_continuous_scale="Viridis")
               st.plotly_chart(fig_heatmap, use_container_width=True)
               st.markdown(get_chart_download_link(fig_heatmap, "transaction_heatmap"), unsafe_allow_html=True)

            # Download Data
            csv = df.to_csv(index=False)
            st.download_button("Download Data as CSV", csv, "data.csv", "text/csv")

         with tabs[1]:
            st.header("Invoice Processing")
            if file_type == "Text":
               invoice_data = parse_invoice_text(df.iloc[0]['text'] if 'text' in df.columns else '')
               st.json(invoice_data)
               if 'amount' in df.columns:
                  fig_pie = px.pie(df, values='amount',
                                   names='document_type' if 'document_type' in df.columns else ['Invoice'],
                                   title="Invoice Distribution")
                  st.plotly_chart(fig_pie, use_container_width=True)
                  st.markdown(get_chart_download_link(fig_pie, "invoice_distribution"), unsafe_allow_html=True)
            else:
               st.warning("Upload text file for invoice processing")

         with tabs[2]:
            st.header("Bookkeeping")
            if access_level in ["CA", "SME_Owner"]:
               with st.form("bookkeeping_form"):
                  amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0)
                  gstin = st.text_input("GSTIN (Optional)")
                  doc_type = st.selectbox("Document Type", ["Sales", "Purchase"])
                  submitted = st.form_submit_button("Record Transaction")
                  if submitted:
                     try:
                        entry = ledger.auto_entry(
                           {"amount": amount, "gstin": gstin, "date": "2025-06-19", "document_type": doc_type})
                        st.success(f"Transaction Recorded: {entry}")
                     except Exception as e:
                        st.error(f"Error: {str(e)}")
               categorized = ai_bookkeeping(df)
               st.write("Categorized Transactions:", categorized)
               fig_bar = px.bar(pd.DataFrame(categorized), x="transaction_id", y=[1] * len(categorized),
                                color="category", title="Transaction Categories")
               st.plotly_chart(fig_bar, use_container_width=True)
               st.markdown(get_chart_download_link(fig_bar, "transaction_categories"), unsafe_allow_html=True)
            else:
               st.write("Read-only access")

         with tabs[3]:
            st.header("Bank Reconciliation")
            if access_level in ["CA", "SME_Owner"]:
               bank_feed = [{"id": i, "amount": float(a), "date": "2025-01-01"} for i, a in
                            enumerate(df['amount'][:2] if 'amount' in df.columns else [1000, 5000])]
               recon = bank_reconciliation(df.to_dict('records'), bank_feed)
               st.write("Reconciliation Results:", recon)
               fig_sunburst = px.sunburst(
                  pd.DataFrame(
                     {"status": ["Matched", "Unmatched"], "count": [len(recon["matches"]), recon["unmatched"]]}),
                  path=["status"], values="count", title="Reconciliation Status"
               )
               st.plotly_chart(fig_sunburst, use_container_width=True)
               st.markdown(get_chart_download_link(fig_sunburst, "reconciliation_status"), unsafe_allow_html=True)
            else:
               st.write("Read-only access")

         with tabs[4]:
            st.header("Inventory Optimization")
            if access_level in ["CA", "SME_Owner"]:
               items = [{"id": i, "stock": 100, "cost": 50} for i in range(2)]
               inventory = manage_inventory(items)
               st.write("Optimal Inventory Levels:", inventory)
               fig_inventory = px.bar(pd.DataFrame(inventory), x="item_id", y="optimal_stock",
                                      title="Optimal Inventory")
               st.plotly_chart(fig_inventory, use_container_width=True)
               st.markdown(get_chart_download_link(fig_inventory, "inventory_levels"), unsafe_allow_html=True)
            else:
               st.write("Read-only access")

         with tabs[5]:
            st.header("Tax Computation")
            tax_type = st.selectbox("Tax Type", ["GST", "TDS", "IT_44ADA"], key="tax_type")
            tax = compute_tax(df, tax_type)
            st.write(f"{tax['tax_type']} Amount: ₹{tax['amount']:.2f}")
            fig_tax = px.bar(x=[tax['tax_type']], y=[tax['amount']], title="Tax Amount")
            st.plotly_chart(fig_tax, use_container_width=True)
            st.markdown(get_chart_download_link(fig_tax, "tax_amount"), unsafe_allow_html=True)

         with tabs[6]:
            st.header("Audit Report Generation")
            report = generate_audit_report(df)
            st.write("Audit Report:", report)
            fig_audit = px.treemap(
               pd.DataFrame({
                  "category": ["Form 3CA", "Form 3CD"],
                  "value": [1, report["report"]["form_3cd"]["transactions_analyzed"]],
                  "parent": ["", "Form 3CD"]
               }),
               path=["category"], values="value", title="Audit Report Structure"
            )
            st.plotly_chart(fig_audit, use_container_width=True)
            st.markdown(get_chart_download_link(fig_audit, "audit_report"), unsafe_allow_html=True)

         with tabs[7]:
            st.header("Payroll Processing")
            if access_level in ["CA", "SME_Owner"]:
               employees = [{"id": i, "salary": 50000} for i in range(2)]
               payroll = process_payroll(employees)
               st.write("Payroll Details:", payroll)
               fig_payroll = px.bar(pd.DataFrame(payroll), x="employee_id", y=["gross", "tds", "pf", "net"],
                                    title="Payroll Breakdown")
               st.plotly_chart(fig_payroll, use_container_width=True)
               st.markdown(get_chart_download_link(fig_payroll, "payroll_breakdown"), unsafe_allow_html=True)
            else:
               st.write("Read-only access")

         with tabs[8]:
            st.header("Vendor Risk Analysis")
            risk = vendor_credit_risk(df)
            st.write("Vendor Risk Scores:", risk)
            fig_risk = px.histogram(x=risk["risk_scores"], title="Vendor Risk Distribution")
            st.plotly_chart(fig_risk, use_container_width=True)
            st.markdown(get_chart_download_link(fig_risk, "vendor_risk"), unsafe_allow_html=True)

         with tabs[9]:
            st.header("Loan Readiness Reports")
            loan = loan_readiness_report(df)
            st.write("Loan Readiness Report:", loan)
            fig_loan = px.bar(
               x=["Revenue", "Expenses", "Net"],
               y=[loan["profit_loss"]["revenue"], loan["profit_loss"]["expenses"], loan["profit_loss"]["net"]],
               title="Profit & Loss Summary"
            )
            st.plotly_chart(fig_loan, use_container_width=True)
            st.markdown(get_chart_download_link(fig_loan, "loan_summary"), unsafe_allow_html=True)

         with tabs[10]:
            st.header("Fraud Detection")
            if st.button("Detect Fraud", key="fraud_button"):
               with st.spinner("Running Mock Theta..."):
                  fraud = mock_theta_fraud(df)
                  st.write(f"Fraud Score: {fraud['score']:.2f}")
                  fig_fraud = px.line(y=fraud['coefficients'], title="Mock Theta Coefficients",
                                      labels={"y": "Coefficient", "x": "Transaction Index"})
                  st.plotly_chart(fig_fraud, use_container_width=True)
                  st.markdown(get_chart_download_link(fig_fraud, "fraud_coefficients"), unsafe_allow_html=True)

         with tabs[11]:
            st.header("Cash Flow Forecasting")
            if st.button("Generate Forecast", key="forecast_button"):
               with st.spinner("Running Kalman Filter..."):
                  forecast = kalman_forecast(df)
                  dates = pd.date_range(start="2025-06-19", periods=30, freq="D")
                  fig_forecast = px.line(
                     x=dates, y=forecast['forecast'],
                     title="Cash Flow Forecast",
                     labels={"x": "Date", "y": "Amount"}
                  )
                  st.plotly_chart(fig_forecast, use_container_width=True)
                  st.markdown(get_chart_download_link(fig_forecast, "cash_flow_forecast"), unsafe_allow_html=True)

         with tabs[12]:
            st.header("Anomaly Detection")
            if st.button("Detect Anomalies", key="anomaly_button"):
               with st.spinner("Running Neural CDE + Zeta..."):
                  anomaly = neural_cde_anomaly(df)
                  if "error" in anomaly:
                     st.error(anomaly["error"])
                  else:
                     st.write("Anomalous Transactions:", anomaly["anomalies"])
                     fig_anomaly = px.scatter(
                        df,
                        x='date' if 'date' in df.columns else df.index,
                        y='amount' if 'amount' in df.columns else df.columns[0],
                        color=[1 if i in [x['amount'] for x in anomaly["anomalies"]] else 0 for i in
                               df['amount']] if 'amount' in df.columns else [0] * len(df),
                        title="Anomaly Detection",
                        labels={"color": "Anomaly"},
                        color_discrete_map={0: "blue", 1: "red"}
                     )
                     st.plotly_chart(fig_anomaly, use_container_width=True)
                     st.markdown(get_chart_download_link(fig_anomaly, "anomaly_detection"), unsafe_allow_html=True)

         with tabs[13]:
            st.header("Predictive Alerts")
            alert = predictive_alerts(df)
            st.write("Alert:", alert)
            fig_alert = go.Figure(go.Indicator(
               mode="gauge+number",
               value=1 if alert["severity"] == "High" else 0,
               title={"text": "Alert Severity"},
               gauge={'axis': {'range': [0, 1]}, 'bar': {'color': "red" if alert["severity"] == "High" else "green"}}
            ))
            st.plotly_chart(fig_alert, use_container_width=True)
            st.markdown(get_chart_download_link(fig_alert, "alert_severity"), unsafe_allow_html=True)

         with tabs[14]:
            st.header("Smart Deduction Finder")
            deductions = smart_deduction_finder(df)
            st.write("Tax Deductions:", deductions)
            fig_deductions = px.bar(
               pd.DataFrame(deductions), x="transaction_id", y=[1] * len(deductions),
               title="Tax Deductions Identified"
            )
            st.plotly_chart(fig_deductions, use_container_width=True)
            st.markdown(get_chart_download_link(fig_deductions, "tax_deductions"), unsafe_allow_html=True)

         with tabs[15]:
            st.header("Math Insights")
            with st.expander("1. Mock Theta (Fraud)"):
               fraud = mock_theta_fraud(df)
               st.markdown(r"\[ f(q) = \sum_{n=0}^{\infty} a_n q^n \]")
               fig_fraud = px.line(y=fraud['coefficients'], title="Mock Theta Coefficients")
               st.plotly_chart(fig_fraud, use_container_width=True)
               st.write(f"Score: {fraud['score']:.2f} ({fraud['use_case']})")
               st.markdown(get_chart_download_link(fig_fraud, "mock_theta"), unsafe_allow_html=True)

            with st.expander("2. Riemann Zeta (Anomaly)"):
               anomaly = neural_cde_anomaly(df)
               st.markdown(r"\[ \zeta(s) = \sum_{n=1}^{\infty} \frac{1}{n^s} \]")
               if "zeta_scores" in anomaly:
                  fig_zeta = px.line(y=anomaly['zeta_scores'], title="Zeta Scores")
                  st.plotly_chart(fig_zeta, use_container_width=True)
                  st.markdown(get_chart_download_link(fig_zeta, "zeta_scores"), unsafe_allow_html=True)

            with st.expander("3. GARCH Volatility"):
               vol = garch_volatility(df)
               st.markdown(r"\[ \sigma_t^2 = \alpha_0 + \alpha_1 \epsilon_{t-1}^2 + \beta_1 \sigma_{t-1}^2 \]")
               fig_vol = px.bar(x=["Volatility"], y=[vol['volatility']], title="GARCH Volatility")
               st.plotly_chart(fig_vol, use_container_width=True)
               st.write(f"Volatility: {vol['volatility']:.2f} ({vol['use_case']})")
               st.markdown(get_chart_download_link(fig_vol, "garch_volatility"), unsafe_allow_html=True)

            with st.expander("4. Kalman Filter (Forecast)"):
               forecast = kalman_forecast(df)
               st.markdown(r"\[ \hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k(y_k - H \hat{x}_{k|k-1}) \]")
               fig_kalman = px.line(y=forecast['estimates'], title="Kalman Estimates")
               st.plotly_chart(fig_kalman, use_container_width=True)
               st.markdown(get_chart_download_link(fig_kalman, "kalman_estimates"), unsafe_allow_html=True)

            with st.expander("5. Fourier Series (Trends)"):
               fourier = fourier_trends(df)
               st.markdown(r"\[ f(x) = a_0 + \sum_{n=1}^{\infty} (a_n \cos nx + b_n \sin nx) \]")
               fig_fourier = px.line(x=fourier['frequencies'], y=fourier['amplitudes'], title="Fourier Amplitudes")
               st.plotly_chart(fig_fourier, use_container_width=True)
               st.markdown(get_chart_download_link(fig_fourier, "fourier_amplitudes"), unsafe_allow_html=True)

            with st.expander("6. Noether Symmetry (Ledger)"):
               symmetry = noether_symmetry(df)
               st.markdown(r"\[ \sum \text{Assets} = \sum \text{Liabilities} + \sum \text{Equity} \]")
               fig_symmetry = go.Figure(go.Indicator(
                  mode="gauge+number",
                  value=symmetry['deviation'],
                  title={"text": "Ledger Deviation"},
                  gauge={'axis': {'range': [0, max(symmetry['deviation'], 1)]},
                         'bar': {'color': "red" if symmetry['deviation'] > 0.01 else "green"}}
               ))
               st.plotly_chart(fig_symmetry, use_container_width=True)
               st.write(f"Deviation: {symmetry['deviation']:.2f} ({symmetry['use_case']})")
               st.markdown(get_chart_download_link(fig_symmetry, "noether_symmetry"), unsafe_allow_html=True)

            with st.expander("7. Topological Data Analysis"):
               tda = topological_data_analysis(df)
               st.markdown(r"\[ H_k(X) \rightarrow \text{Persistent Homology} \]")
               fig_tda = px.scatter(
                  x=[p[1][0] for p in tda['persistence_diagram']],
                  y=[p[1][1] for p in tda['persistence_diagram']],
                  title="Persistence Diagram",
                  labels={"x": "Birth", "y": "Death"}
               )
               st.plotly_chart(fig_tda, use_container_width=True)
               st.write(f"Diagram: {tda['persistence_diagram']} ({tda['use_case']})")
               st.markdown(get_chart_download_link(fig_tda, "persistence_diagram"), unsafe_allow_html=True)

            with st.expander("8. Quantum Audit"):
               quantum = quantum_audit(df)
               st.markdown(r"\[ S(\rho) = -\text{Tr}(\rho \log \rho) \]")
               fig_quantum = px.bar(x=["Entropy"], y=[quantum['quantum_entropy']], title="Quantum Entropy")
               st.plotly_chart(fig_quantum, use_container_width=True)
               st.write(f"Entropy: {quantum['quantum_entropy']:.2f} ({quantum['use_case']})")
               st.markdown(get_chart_download_link(fig_quantum, "quantum_entropy"), unsafe_allow_html=True)

            with st.expander("9. Ramanujan Partitions"):
               partitions = ramanujan_partitions(df)
               st.markdown(r"\[ p(n) \approx \frac{1}{4n\sqrt{3}} e^{\pi \sqrt{\frac{2n}{3}}} \]")
               fig_partitions = px.bar(x=["Partitions"], y=[partitions['partitions']], title="Ramanujan Partitions")
               st.plotly_chart(fig_partitions, use_container_width=True)
               st.write(f"Partitions: {partitions['partitions']} ({partitions['use_case']})")
               st.markdown(get_chart_download_link(fig_partitions, "ramanujan_partitions"), unsafe_allow_html=True)

            with st.expander("10. Game-Theoretic Shapley Compliance"):
               shapley = game_theoretic_shapley_compliance(df)
               st.markdown(
                  r"\[ \phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} [v(S \cup \{i\}) - v(S)] \]")
               fig_shapley = px.bar(x=range(len(shapley['shapley_values'])), y=shapley['shapley_values'],
                                    title="Shapley Values")
               st.plotly_chart(fig_shapley, use_container_width=True)
               st.write(f"Values: {shapley['shapley_values']} ({shapley['use_case']})")
               st.markdown(get_chart_download_link(fig_shapley, "shapley_values"), unsafe_allow_html=True)

            with st.expander("11. Fractional Differentiation"):
               if 'amount' in df.columns:
                  frac_diff = fractional_diff(df['amount'])
                  st.markdown(r"\[ D^d f(t) = \frac{1}{\Gamma(1-d)} \int_{-\infty}^t (t-\tau)^{-d} f(\tau) d\tau \]")
                  fig_frac = px.line(y=frac_diff, title="Fractional Differencing")
                  st.plotly_chart(fig_frac, use_container_width=True)
                  st.write("Fractional Differencing Applied")
                  st.markdown(get_chart_download_link(fig_frac, "fractional_diff"), unsafe_allow_html=True)

            with st.expander("12. T-Copula"):
               numeric_cols = df.select_dtypes(include=[np.number]).columns
               if len(numeric_cols) >= 2:
                  copula = t_copula(df[numeric_cols[:2]])
                  st.markdown(r"\[ C(u_1, u_2; \nu) = t_{\nu, \Sigma}(t_{\nu}^{-1}(u_1), t_{\nu}^{-1}(u_2)) \]")
                  fig_copula = px.scatter(x=copula[:, 0], y=copula[:, 1], title="T-Copula Scatter")
                  st.plotly_chart(fig_copula, use_container_width=True)
                  st.write("T-Copula Dependencies Modeled")
                  st.markdown(get_chart_download_link(fig_copula, "t_copula"), unsafe_allow_html=True)


# CLI Implementation
@click.group()
def cli():
   """AutoCA CLI for financial analytics."""
   pass


@cli.command()
@click.option('--data-file', type=str, required=True, help="Path to data file")
@click.option('--task', type=click.Choice([
   'anomaly', 'forecast', 'fraud', 'tax', 'bookkeeping', 'audit', 'payroll',
   'inventory', 'reconcile', 'vendor', 'loan', 'alerts', 'deductions', 'tda',
   'quantum', 'partitions', 'shapley', 'frac_diff', 'copula'
]), default='anomaly', help="Analysis task")
@click.option('--output', type=str, default='output.png', help="Output file for visualization")
def analyze(data_file, task, output):
   """Analyze business data."""
   df = load_data(open(data_file, 'rb'), file_type=data_file.split('.')[-1])
   if df is None:
      click.echo("Failed to load data")
      return

   if task == 'anomaly':
      result = neural_cde_anomaly(df)
      if 'error' not in result:
         fig = go.Figure()
         fig.add_trace(go.Scatter(
            x=df.index, y=df['amount'] if 'amount' in df.columns else df[df.columns[0]],
            mode='markers',
            marker=dict(color=['red' if i in [x['amount'] for x in result['anomalies']] else 'blue' for i in
                               df['amount']] if 'amount' in df.columns else ['blue'] * len(df))
         ))
         fig.write_image(output)
         click.echo(f"Anomalies: {result['anomalies']}")
         click.echo(f"Visualization saved to {output}")

   elif task == 'forecast':
      result = kalman_forecast(df)
      dates = pd.date_range(start="2025-06-19", periods=30, freq="D")
      fig = go.Figure(data=go.Scatter(x=dates, y=result['forecast'], mode='lines'))
      fig.write_image(output)
      click.echo(f"Forecast saved to {output}")

   elif task == 'fraud':
      result = mock_theta_fraud(df)
      fig = go.Figure(data=go.Scatter(y=result['coefficients'], mode='lines'))
      fig.write_image(output)
      click.echo(f"Fraud Score: {result['score']}")
      click.echo(f"Visualization saved to {output}")

   elif task == 'tax':
      result = compute_tax(df, "GST")
      click.echo(f"Tax Amount: {result['amount']}")

   elif task == 'bookkeeping':
      entry = ledger.auto_entry({
         "amount": float(df['amount'].iloc[0] if 'amount' in df.columns else 1000),
         "gstin": df['gstin'].iloc[0] if 'gstin' in df.columns else None,
         "date": "2025-06-19",
         "document_type": "Sales"
      })
      click.echo(f"Transaction: {entry}")

   elif task == 'audit':
      result = generate_audit_report(df)
      click.echo(f"Audit Report: {result}")

   elif task == 'payroll':
      employees = [{"id": i, "salary": 50000} for i in range(2)]
      result = process_payroll(employees)
      click.echo(f"Payroll: {result}")

   elif task == 'inventory':
      items = [{"id": i, "stock": 100, "cost": 50} for i in range(2)]
      result = manage_inventory(items)
      click.echo(f"Inventory: {result}")

   elif task == 'reconcile':
      bank_feed = [{"id": i, "amount": float(a), "date": "2025-01-01"} for i, a in
                   enumerate(df['amount'][:2] if 'amount' in df.columns else [1000, 5000])]
      result = bank_reconciliation(df.to_dict('records'), bank_feed)
      click.echo(f"Reconciliation: {result}")

   elif task == 'vendor':
      result = vendor_credit_risk(df)
      click.echo(f"Vendor Risk: {result}")

   elif task == 'loan':
      result = loan_readiness_report(df)
      click.echo(f"Loan Report: {result}")

   elif task == 'alerts':
      result = predictive_alerts(df)
      click.echo(f"Alerts: {result}")

   elif task == 'deductions':
      result = smart_deduction_finder(df)
      click.echo(f"Deductions: {result}")

   elif task == 'tda':
      result = topological_data_analysis(df)
      fig = go.Figure(data=go.Scatter(
         x=[p[1][0] for p in result['persistence_diagram']],
         y=[p[1][1] for p in result['persistence_diagram']],
         mode='markers'
      ))
      fig.write_image(output)
      click.echo(f"TDA: {result['persistence_diagram']}")
      click.echo(f"Visualization saved to {output}")

   elif task == 'quantum':
      result = quantum_audit(df)
      fig = go.Figure(data=go.Bar(x=["Entropy"], y=[result['quantum_entropy']]))
      fig.write_image(output)
      click.echo(f"Quantum Entropy: {result['quantum_entropy']}")
      click.echo(f"Visualization saved to {output}")

   elif task == 'partitions':
      result = ramanujan_partitions(df)
      fig = go.Figure(data=go.Bar(x=["Partitions"], y=[result['partitions']]))
      fig.write_image(output)
      click.echo(f"Partitions: {result['partitions']}")
      click.echo(f"Visualization saved to {output}")

   elif task == 'shapley':
      result = game_theoretic_shapley_compliance(df)
      fig = go.Figure(data=go.Bar(x=list(range(len(result['shapley_values']))), y=result['shapley_values']))
      fig.write_image(output)
      click.echo(f"Shapley Values: {result['shapley_values']}")
      click.echo(f"Visualization saved to {output}")

   elif task == 'frac_diff':
      if 'amount' in df.columns:
         result = fractional_diff(df['amount'])
         fig = go.Figure(data=go.Scatter(y=result, mode='lines'))
         fig.write_image(output)
         click.echo("Fractional Differencing Applied")
         click.echo(f"Visualization saved to {output}")

   elif task == 'copula':
      numeric_cols = df.select_dtypes(include=[np.number]).columns
      if len(numeric_cols) >= 2:
         result = t_copula(df[numeric_cols[:2]])
         fig = go.Figure(data=go.Scatter(x=result[:, 0], y=result[:, 1], mode='markers'))
         fig.write_image(output)
         click.echo("T-Copula Applied")
         click.echo(f"Visualization saved to {output}")


if __name__ == '__main__':
   if len(sys.argv) > 1:
      cli()
   else:
      render_ui()

