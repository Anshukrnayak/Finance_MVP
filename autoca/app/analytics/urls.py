from django.urls import path
from .views import (
    AnalyticsDashboardView,
    FraudDetectionView,
    BatchFraudAnalysisView,
    PredictiveAnalyticsView
)

from .ml_views import (
    FraudDetectionMLView,
    PredictiveAnalyticsMLView,
    ModelPerformanceMLView
)


urlpatterns = [

    
    path('dashboard/', AnalyticsDashboardView.as_view(), name='analytics-dashboard'),
    path('fraud-detection/<int:invoice_id>/', FraudDetectionView.as_view(), name='fraud-detection'),
    path('fraud-detection/batch/', BatchFraudAnalysisView.as_view(), name='batch-fraud-analysis'),
    path('predictive/', PredictiveAnalyticsView.as_view(), name='predictive-analytics'),

    path('ml/fraud-detection/<int:invoice_id>/', FraudDetectionMLView.as_view(), name='ml-fraud-detection'),
    path('ml/predictive-analytics/<int:user_id>/', PredictiveAnalyticsMLView.as_view(), name='ml-predictive-analytics'),
    path('ml/model-performance/', ModelPerformanceMLView.as_view(), name='ml-model-performance'),
]
