from django.urls import path, include
from rest_framework.routers import DefaultRouter
from rest_framework_simplejwt.views import TokenRefreshView
from . import views

router = DefaultRouter()
router.register(r'users', views.UserViewSet, basename='user')
router.register(r'profiles', views.UserProfileViewSet, basename='profile')
router.register(r'subscriptions', views.SubscriptionViewSet, basename='subscription')
router.register(r'audit-logs', views.AuditLogViewSet, basename='audit-log')
router.register(r'api-keys', views.APIKeyViewSet, basename='api-key')

urlpatterns = [
    # Authentication endpoints
    path('auth/register/', views.UserRegistrationView.as_view(), name='user-register'),
    path('auth/login/', views.CustomTokenObtainPairView.as_view(), name='token-obtain-pair'),
    path('auth/token/refresh/', TokenRefreshView.as_view(), name='token-refresh'),
    path('auth/logout/', views.UserLogoutView.as_view(), name='user-logout'),
    path('auth/change-password/', views.ChangePasswordView.as_view(), name='change-password'),
    path('auth/reset-password/', views.ResetPasswordRequestView.as_view(), name='reset-password-request'),
    path('auth/reset-password/confirm/', views.ResetPasswordConfirmView.as_view(), name='reset-password-confirm'),

    # User management
    path('users/me/', views.CurrentUserView.as_view(), name='current-user'),
    path('users/<uuid:pk>/activate/', views.UserActivationView.as_view(), name='user-activate'),
    path('users/<uuid:pk>/deactivate/', views.UserDeactivationView.as_view(), name='user-deactivate'),

    # Profile management
    path('profiles/me/', views.CurrentUserProfileView.as_view(), name='current-profile'),
    path('profiles/me/avatar/', views.UserAvatarUploadView.as_view(), name='user-avatar-upload'),

    # Subscription management
    path('subscriptions/me/', views.CurrentUserSubscriptionView.as_view(), name='current-subscription'),
    path('subscriptions/me/upgrade/', views.SubscriptionUpgradeView.as_view(), name='subscription-upgrade'),
    path('subscriptions/me/cancel/', views.SubscriptionCancelView.as_view(), name='subscription-cancel'),
    path('subscriptions/me/invoice/', views.SubscriptionInvoiceView.as_view(), name='subscription-invoice'),

    # API key management
    path('api-keys/me/', views.UserAPIKeysView.as_view(), name='user-api-keys'),
    path('api-keys/<uuid:pk>/regenerate/', views.APIKeyRegenerateView.as_view(), name='api-key-regenerate'),

    # System endpoints
    path('system/health/', views.SystemHealthView.as_view(), name='system-health'),
    path('system/stats/', views.SystemStatisticsView.as_view(), name='system-stats'),
    path('system/backup/', views.SystemBackupView.as_view(), name='system-backup'),

    # Include router URLs
    path('', include(router.urls)),
]

# Optional: Add debug endpoints for development
try:
    from django.conf import settings
    if settings.DEBUG:
        urlpatterns += [
            path('debug/users/', views.DebugUserListView.as_view(), name='debug-user-list'),
            path('debug/subscriptions/', views.DebugSubscriptionView.as_view(), name='debug-subscription'),
        ]
except ImportError:
    pass

