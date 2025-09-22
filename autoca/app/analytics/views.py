import logging
import pandas as pd
import numpy as np
from django.http import JsonResponse
from django.db.models import Sum, Count, Avg
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from ratelimit.decorators import ratelimit

from core.views import BaseAPIView
from invoices.models import Invoice
from invoices.tasks import (
    raman_fraud_detection_task,
    quantum_audit_simulation_task,
    tda_anomaly_detection_task
)
from compliance.tasks import file_gstr3b_task, generate_xbrl_report_task
from compliance.models import GSTRFiling

logger = logging.getLogger(__name__)

class AnalyticsDashboardView(BaseAPIView):
    """Comprehensive analytics dashboard"""

    @ratelimit(key='user', rate='10/m', method='GET')
    def get(self, request):
        try:
            # Basic statistics
            total_invoices = Invoice.objects.filter(
                user=request.user,
                is_deleted=False
            ).count()

            processed_invoices = Invoice.objects.filter(
                user=request.user,
                processing_status='processed',
                is_deleted=False
            ).count()

            # Financial metrics
            financial_data = Invoice.objects.filter(
                user=request.user,
                processing_status='processed',
                extracted_data__has_key='amount',
                is_deleted=False
            ).aggregate(
                total_amount=Sum('extracted_data__amount'),
                avg_amount=Avg('extracted_data__amount'),
                invoice_count=Count('id')
            )

            # Fraud statistics
            fraud_stats = Invoice.objects.filter(
                user=request.user,
                is_deleted=False
            ).exclude(fraud_risk_level__isnull=True).values(
                'fraud_risk_level'
            ).annotate(
                count=Count('id'),
                avg_score=Avg('fraud_score')
            )

            # Monthly trends
            monthly_trends = Invoice.objects.filter(
                user=request.user,
                processing_status='processed',
                is_deleted=False
            ).extra(
                select={'month': "DATE_TRUNC('month', uploaded_at)"}
            ).values('month').annotate(
                count=Count('id'),
                total_amount=Sum('extracted_data__amount')
            ).order_by('month')

            return Response({
                "overview": {
                    "total_invoices": total_invoices,
                    "processed_invoices": processed_invoices,
                    "success_rate": (processed_invoices / total_invoices * 100) if total_invoices > 0 else 0
                },
                "financials": financial_data,
                "fraud_analysis": list(fraud_stats),
                "monthly_trends": list(monthly_trends)
            })

        except Exception as e:
            logger.error(f"Analytics dashboard failed: {e}")
            return Response(
                {"error": "Failed to load analytics"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class FraudDetectionView(BaseAPIView):
    """Trigger advanced fraud detection"""

    @ratelimit(key='user', rate='5/m', method='POST')
    def post(self, request, invoice_id):
        try:
            # Trigger all fraud detection tasks in chain
            chain = (
                    raman_fraud_detection_task.si(invoice_id)
                    | quantum_audit_simulation_task.si(invoice_id)
                    | tda_anomaly_detection_task.si(invoice_id)
            )
            result = chain.apply_async()

            return Response({
                "message": "Fraud detection started",
                "task_id": result.id
            }, status=status.HTTP_202_ACCEPTED)

        except Exception as e:
            logger.error(f"Fraud detection trigger failed: {e}")
            return Response(
                {"error": "Failed to start fraud detection"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class BatchFraudAnalysisView(BaseAPIView):
    """Batch fraud analysis for multiple invoices"""

    @ratelimit(key='user', rate='2/m', method='POST')
    def post(self, request):
        try:
            invoice_ids = request.data.get('invoice_ids', [])
            if not invoice_ids:
                return Response(
                    {"error": "No invoice IDs provided"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            if len(invoice_ids) > 100:
                return Response(
                    {"error": "Maximum 100 invoices per batch analysis"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Create group of tasks
            tasks = []
            for invoice_id in invoice_ids:
                task_chain = (
                        raman_fraud_detection_task.si(invoice_id)
                        | quantum_audit_simulation_task.si(invoice_id)
                        | tda_anomaly_detection_task.si(invoice_id)
                )
                tasks.append(task_chain)

            # Execute in parallel
            from celery import group
            job = group(tasks)
            result = job.apply_async()

            return Response({
                "message": "Batch fraud analysis started",
                "task_id": result.id,
                "invoice_count": len(invoice_ids)
            }, status=status.HTTP_202_ACCEPTED)

        except Exception as e:
            logger.error(f"Batch fraud analysis failed: {e}")
            return Response(
                {"error": "Failed to start batch analysis"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class PredictiveAnalyticsView(BaseAPIView):
    """Advanced predictive analytics"""

    @ratelimit(key='user', rate='5/m', method='GET')
    def get(self, request):
        try:
            # Get historical data for predictions
            invoices = Invoice.objects.filter(
                user=request.user,
                processing_status='processed',
                extracted_data__has_key='amount',
                is_deleted=False
            ).values('uploaded_at', 'extracted_data__amount')[:1000]

            if not invoices:
                return Response({"message": "Insufficient data for analysis"})

            # Create time series data
            df = pd.DataFrame(list(invoices))
            df['date'] = pd.to_datetime(df['uploaded_at'])
            df['amount'] = pd.to_numeric(df['extracted_data__amount'], errors='coerce')
            df = df.dropna()

            # Basic time series analysis
            monthly = df.set_index('date').resample('M')['amount'].agg(['sum', 'count', 'mean'])

            # Simple forecasting (could be enhanced with proper ML models)
            forecast = self._simple_forecast(monthly['sum'])

            return Response({
                "historical_trends": monthly.to_dict(),
                "forecast": forecast,
                "statistics": {
                    "total_periods": len(monthly),
                    "avg_monthly_volume": monthly['sum'].mean(),
                    "growth_rate": self._calculate_growth_rate(monthly['sum'])
                }
            })

        except Exception as e:
            logger.error(f"Predictive analytics failed: {e}")
            return Response(
                {"error": "Failed to generate predictive analytics"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def _simple_forecast(self, series):
        """Simple moving average forecast"""
        if len(series) < 3:
            return {"message": "Insufficient data for forecasting"}

        # Last 3 months moving average
        last_values = series[-3:].values
        forecast_value = np.mean(last_values)

        return {
            "next_period_forecast": forecast_value,
            "confidence_interval": {
                "lower": forecast_value * 0.8,
                "upper": forecast_value * 1.2
            },
            "method": "3-month moving average"
        }

    def _calculate_growth_rate(self, series):
        """Calculate monthly growth rate"""
        if len(series) < 2:
            return 0
        return ((series.iloc[-1] / series.iloc[-2]) - 1) * 100