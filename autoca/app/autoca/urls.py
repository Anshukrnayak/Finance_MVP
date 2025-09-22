from django.urls import path, include
from django.contrib import admin
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/auth/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/auth/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),

    path('api/core/', include('core.urls')),
    path('api/invoices/', include('invoices.urls')),
    path('api/analytics/', include('analytics.urls')),
    path('api/compliance/', include('compliance.urls')),
    path('api/billing/', include('billing.urls')),
]