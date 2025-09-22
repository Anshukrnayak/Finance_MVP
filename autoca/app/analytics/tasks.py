import logging
import numpy as np
from celery import shared_task
from sklearn.ensemble import IsolationForest
from scipy.fft import fft
from django.db import transaction
from django.utils import timezone

from invoices.models import Invoice
from core.models import AuditLog

logger = logging.getLogger(__name__)

@shared_task(
    bind=True,
    max_retries=2,
    time_limit=180,
    soft_time_limit=150
)
def raman_fraud_detection_task(self, invoice_id: int):
    """
    Task 4: Raman Fraud Detection with Advanced Analytics
    """
    try:
        invoice = Invoice.objects.get(id=invoice_id)
        if not invoice.extracted_data.get('amount'):
            return {'status': 'skipped', 'reason': 'No amount data'}

        amount = float(invoice.extracted_data['amount'])

        # Get recent transactions for context
        recent_amounts = _get_recent_transaction_amounts(invoice.user, limit=100)
        amounts_array = np.array(recent_amounts + [amount]).reshape(-1, 1)

        # Mock theta function simulation (simplified)
        q = np.exp(-amount / (np.max(amounts_array) + 1e-10))
        mock_theta = np.sum([q ** i for i in range(len(amounts_array))])
        theta_score = 1 / (1 + mock_theta)

        # Isolation Forest anomaly detection
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        iso_forest.fit(amounts_array[:-1])  # Train on historical data
        iso_score = -iso_forest.score_samples([[amount]])[0]

        # Frequency analysis
        fft_result = np.abs(fft(amounts_array.flatten()))
        freq_anomaly = np.mean(fft_result > np.percentile(fft_result, 95))

        # Entropy calculation
        bins, _ = np.histogram(amounts_array, bins='auto', density=True)
        entropy = -np.sum(bins * np.log2(bins + 1e-10))

        # Combined fraud score
        fraud_score = 0.4 * theta_score + 0.3 * iso_score + 0.2 * freq_anomaly + 0.1 * entropy

        # Update invoice
        invoice.fraud_score = fraud_score
        if fraud_score > 0.8:
            invoice.fraud_risk_level = 'critical'
        elif fraud_score > 0.6:
            invoice.fraud_risk_level = 'high'
        elif fraud_score > 0.4:
            invoice.fraud_risk_level = 'medium'
        else:
            invoice.fraud_risk_level = 'low'
        invoice.save()

        logger.info(f"Fraud detection completed for invoice {invoice_id}: score {fraud_score:.3f}")
        return {'status': 'success', 'fraud_score': fraud_score}

    except Exception as e:
        logger.error(f"Fraud detection failed for invoice {invoice_id}: {str(e)}")
        raise self.retry(exc=e)

@shared_task
def quantum_audit_simulation_task(self, invoice_id: int):
    """
    Task 5: Quantum Audit Simulation (Simplified Classical Simulation)
    """
    try:
        invoice = Invoice.objects.get(id=invoice_id)
        amount = float(invoice.extracted_data.get('amount', 0))

        # Simulate quantum probability with classical randomness
        risk_factors = [
            amount > 100000,  # Large transaction
            invoice.fraud_score > 0.6 if invoice.fraud_score else False,
            not invoice.is_gstin_valid if hasattr(invoice, 'is_gstin_valid') else False
        ]

        # Calculate audit risk based on factors
        risk_score = sum(risk_factors) / len(risk_factors) if risk_factors else 0
        risk_score += np.random.normal(0, 0.1)  # Add some randomness

        # Store result (for demo purposes)
        quantum_data = {
            'risk_score': min(max(risk_score, 0), 1),
            'factors_considered': len(risk_factors),
            'simulation_type': 'classical_approximation'
        }

        if 'advanced_analytics' not in invoice.extracted_data:
            invoice.extracted_data['advanced_analytics'] = {}
        invoice.extracted_data['advanced_analytics']['quantum_audit'] = quantum_data
        invoice.save()

        return {'status': 'success', 'quantum_risk': risk_score}

    except Exception as e:
        logger.error(f"Quantum audit simulation failed: {str(e)}")
        return {'status': 'failed', 'error': str(e)}

@shared_task
def tda_anomaly_detection_task(self, invoice_id: int):
    """
    Task 7: Topological Data Analysis Anomaly Detection
    """
    try:
        invoice = Invoice.objects.get(id=invoice_id)

        # Get historical transaction data
        historical_data = _get_transaction_patterns(invoice.user)

        if len(historical_data) < 10:
            return {'status': 'skipped', 'reason': 'Insufficient data'}

        # Simplified TDA-like anomaly detection
        current_amount = float(invoice.extracted_data.get('amount', 0))
        amounts = historical_data + [current_amount]

        # Calculate persistence-like features (simplified)
        mean_amount = np.mean(amounts[:-1])
        std_amount = np.std(amounts[:-1])

        # Z-score based anomaly detection
        z_score = (current_amount - mean_amount) / (std_amount + 1e-10)
        anomaly_score = min(1.0, abs(z_score) / 5.0)  # Normalize to 0-1

        # Store results
        tda_data = {
            'anomaly_score': anomaly_score,
            'z_score': z_score,
            'mean_reference': mean_amount,
            'std_reference': std_amount
        }

        if 'advanced_analytics' not in invoice.extracted_data:
            invoice.extracted_data['advanced_analytics'] = {}
        invoice.extracted_data['advanced_analytics']['tda_analysis'] = tda_data
        invoice.save()

        return {'status': 'success', 'anomaly_score': anomaly_score}

    except Exception as e:
        logger.error(f"TDA anomaly detection failed: {str(e)}")
        return {'status': 'failed', 'error': str(e)}

def _get_recent_transaction_amounts(user, limit=100):
    """Get recent transaction amounts for contextual analysis"""
    recent_invoices = Invoice.objects.filter(
        user=user,
        processing_status='processed',
        extracted_data__has_key='amount'
    ).order_by('-processed_at')[:limit]

    amounts = []
    for inv in recent_invoices:
        try:
            amount = float(inv.extracted_data['amount'])
            amounts.append(amount)
        except (ValueError, TypeError):
            continue

    return amounts

def _get_transaction_patterns(user):
    """Get transaction patterns for TDA analysis"""
    return _get_recent_transaction_amounts(user, limit=50)
