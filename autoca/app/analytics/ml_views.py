from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import authentication_classes, permission_classes
from ratelimit.decorators import ratelimit

from core.authentication import MLFeatureAuthentication
from core.permissions import APIKeyPermission
from .serializers import (
    FraudAnalysisSerializer,
    PredictiveAnalyticsSerializer,
    ComplianceRiskSerializer,
    MLModelPerformanceSerializer
)

class MLBaseView(APIView):
    """Base view for ML system integration"""
    authentication_classes = [MLFeatureAuthentication]
    permission_classes = [APIKeyPermission]
    required_permissions = ['ml_access']

class FraudDetectionMLView(MLBaseView):
    """Endpoint for ML system to submit fraud detection results"""

    @ratelimit(key='ip', rate='100/m', method='POST')
    def post(self, request, invoice_id):
        try:
            serializer = FraudAnalysisSerializer(data=request.data)
            if serializer.is_valid():
                # Update invoice with ML results
                from invoices.models import Invoice
                from django.utils import timezone

                invoice = Invoice.objects.get(id=invoice_id)
                ml_data = serializer.validated_data

                # Update invoice with ML results
                if 'advanced_analytics' not in invoice.extracted_data:
                    invoice.extracted_data['advanced_analytics'] = {}

                invoice.extracted_data['advanced_analytics'].update({
                    'fraud_analysis': ml_data,
                    'ml_processed_at': timezone.now().isoformat()
                })

                invoice.fraud_score = ml_data['fraud_score']
                invoice.fraud_risk_level = ml_data['fraud_risk_level']
                invoice.save()

                return Response(
                    {"message": "Fraud analysis results stored successfully"},
                    status=status.HTTP_200_OK
                )

            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        except Invoice.DoesNotExist:
            return Response(
                {"error": "Invoice not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            return Response(
                {"error": f"Failed to process ML results: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class PredictiveAnalyticsMLView(MLBaseView):
    """Endpoint for predictive analytics results"""

    @ratelimit(key='ip', rate='50/m', method='POST')
    def post(self, request, user_id):
        try:
            serializer = PredictiveAnalyticsSerializer(data=request.data)
            if serializer.is_valid():
                # Store predictive analytics results
                from core.models import UserProfile
                from django.utils import timezone

                profile = UserProfile.objects.get(user_id=user_id)
                predictive_data = serializer.validated_data

                # Update user profile with predictions
                if 'analytics' not in profile.notification_preferences:
                    profile.notification_preferences['analytics'] = {}

                profile.notification_preferences['analytics'].update({
                    'predictive_analytics': predictive_data,
                    'last_updated': timezone.now().isoformat()
                })

                profile.save()

                return Response(
                    {"message": "Predictive analytics stored successfully"},
                    status=status.HTTP_200_OK
                )

            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        except UserProfile.DoesNotExist:
            return Response(
                {"error": "User profile not found"},
                status=status.HTTP_404_NOT_FOUND
            )

class ModelPerformanceMLView(MLBaseView):
    """Endpoint for ML model performance metrics"""

    @ratelimit(key='ip', rate='20/m', method='POST')
    def post(self, request):
        try:
            serializer = MLModelPerformanceSerializer(data=request.data)
            if serializer.is_valid():
                # Store model performance metrics
                from django.core.cache import cache

                model_data = serializer.validated_data
                model_name = model_data['model_name']

                # Cache model performance for monitoring
                cache_key = f'model_performance_{model_name}'
                cache.set(cache_key, model_data, timeout=3600)  # 1 hour

                # Also store in database for historical analysis
                from analytics.models import ModelPerformanceHistory
                ModelPerformanceHistory.objects.create(
                    model_name=model_name,
                    performance_data=model_data
                )

                return Response(
                    {"message": "Model performance metrics stored successfully"},
                    status=status.HTTP_200_OK
                )

            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        except Exception as e:
            return Response(
                {"error": f"Failed to store model metrics: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

