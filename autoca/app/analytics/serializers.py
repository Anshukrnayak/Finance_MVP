from rest_framework import serializers
from django.db.models import Avg, Max, Min, Sum
from invoices.models import Invoice
from compliance.models import GSTRFiling

class FraudAnalysisSerializer(serializers.Serializer):
    """Serializer for fraud analysis results"""
    invoice_id = serializers.IntegerField()
    fraud_score = serializers.FloatField(min_value=0, max_value=1)
    fraud_risk_level = serializers.CharField()
    confidence = serializers.FloatField(min_value=0, max_value=1)
    anomalies_detected = serializers.ListField(child=serializers.CharField())
    explanation = serializers.CharField()
    recommendations = serializers.ListField(child=serializers.CharField())

    class Meta:
        fields = (
            'invoice_id', 'fraud_score', 'fraud_risk_level', 'confidence',
            'anomalies_detected', 'explanation', 'recommendations'
        )

class PredictiveAnalyticsSerializer(serializers.Serializer):
    """Serializer for predictive analytics results"""
    period = serializers.CharField()
    predicted_revenue = serializers.FloatField(min_value=0)
    confidence_interval_lower = serializers.FloatField(min_value=0)
    confidence_interval_upper = serializers.FloatField(min_value=0)
    growth_rate = serializers.FloatField()
    seasonality_factor = serializers.FloatField()
    model_used = serializers.CharField()
    model_accuracy = serializers.FloatField(min_value=0, max_value=1)

    class Meta:
        fields = (
            'period', 'predicted_revenue', 'confidence_interval_lower',
            'confidence_interval_upper', 'growth_rate', 'seasonality_factor',
            'model_used', 'model_accuracy'
        )

class ComplianceRiskSerializer(serializers.Serializer):
    """Serializer for compliance risk assessment"""
    risk_score = serializers.FloatField(min_value=0, max_value=1)
    risk_level = serializers.CharField()
    high_risk_transactions = serializers.IntegerField(min_value=0)
    compliance_gaps = serializers.ListField(child=serializers.CharField())
    audit_probability = serializers.FloatField(min_value=0, max_value=1)
    recommended_actions = serializers.ListField(child=serializers.CharField())

    class Meta:
        fields = (
            'risk_score', 'risk_level', 'high_risk_transactions',
            'compliance_gaps', 'audit_probability', 'recommended_actions'
        )

class MLModelPerformanceSerializer(serializers.Serializer):
    """Serializer for ML model performance metrics"""
    model_name = serializers.CharField()
    accuracy = serializers.FloatField(min_value=0, max_value=1)
    precision = serializers.FloatField(min_value=0, max_value=1)
    recall = serializers.FloatField(min_value=0, max_value=1)
    f1_score = serializers.FloatField(min_value=0, max_value=1)
    training_date = serializers.DateTimeField()
    inference_time_ms = serializers.FloatField(min_value=0)
    feature_importance = serializers.DictField(child=serializers.FloatField())

    class Meta:
        fields = (
            'model_name', 'accuracy', 'precision', 'recall', 'f1_score',
            'training_date', 'inference_time_ms', 'feature_importance'
        )

class DashboardMetricsSerializer(serializers.Serializer):
    """Serializer for dashboard metrics"""
    total_invoices = serializers.IntegerField(min_value=0)
    processed_invoices = serializers.IntegerField(min_value=0)
    success_rate = serializers.FloatField(min_value=0, max_value=100)
    total_revenue = serializers.FloatField(min_value=0)
    avg_invoice_amount = serializers.FloatField(min_value=0)
    fraud_rate = serializers.FloatField(min_value=0, max_value=100)
    compliance_rate = serializers.FloatField(min_value=0, max_value=100)
    monthly_trend = serializers.DictField(child=serializers.FloatField())

    class Meta:
        fields = (
            'total_invoices', 'processed_invoices', 'success_rate',
            'total_revenue', 'avg_invoice_amount', 'fraud_rate',
            'compliance_rate', 'monthly_trend'
        )

    def to_representation(self, instance):
        data = super().to_representation(instance)
        # Format percentages
        for field in ['success_rate', 'fraud_rate', 'compliance_rate']:
            if field in data and data[field] is not None:
                data[field] = f"{data[field]:.1f}%"
        return data

    