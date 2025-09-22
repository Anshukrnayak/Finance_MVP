from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework import exceptions
from django.utils.translation import gettext_lazy as _
from .models import APIKey

class JWTAuthenticationWithML(JWTAuthentication):
    """JWT authentication with ML context"""

    def authenticate(self, request):
        try:
            user_auth_tuple = super().authenticate(request)
            if user_auth_tuple:
                user, auth = user_auth_tuple
                # Add ML context to request
                request.ml_context = self._get_ml_context(user)
                return user, auth
        except exceptions.AuthenticationFailed:
            # Fall back to API key authentication
            api_key = request.headers.get('X-API-Key')
            api_secret = request.headers.get('X-API-Secret')

            if api_key and api_secret:
                try:
                    api_key_obj = APIKey.objects.get(
                        key=api_key,
                        secret=api_secret,
                        is_deleted=False
                    )

                    if api_key_obj.is_expired():
                        raise exceptions.AuthenticationFailed('API key expired')

                    user = api_key_obj.user
                    if not user.is_active:
                        raise exceptions.AuthenticationFailed('User inactive')

                    # Update last used
                    api_key_obj.last_used = timezone.now()
                    api_key_obj.save()

                    # Add ML context
                    request.ml_context = self._get_ml_context(user)
                    request.api_key = api_key_obj

                    return user, None

                except APIKey.DoesNotExist:
                    raise exceptions.AuthenticationFailed('Invalid API credentials')

        return None

    def _get_ml_context(self, user):
        """Get ML context for the user"""
        try:
            # Get user's typical transaction patterns for ML context
            from invoices.models import Invoice
            recent_invoices = Invoice.objects.filter(
                user=user,
                processing_status='processed'
            )[:100]

            amounts = [inv.extracted_data.get('amount', 0) for inv in recent_invoices
                       if inv.extracted_data and 'amount' in inv.extracted_data]

            return {
                'user_id': user.id,
                'avg_transaction_amount': sum(amounts) / len(amounts) if amounts else 0,
                'transaction_count': len(amounts),
                'business_category': user.profile.company_name if hasattr(user, 'profile') else 'unknown',
                'ml_features_available': len(amounts) >= 10  # Minimum data for ML
            }
        except Exception:
            return {
                'user_id': user.id,
                'ml_features_available': False
            }

class MLFeatureAuthentication:
    """Authentication for ML feature endpoints"""

    def authenticate(self, request):
        ml_api_key = request.headers.get('X-ML-API-Key')

        if not ml_api_key:
            return None

        # Validate ML API key (in production, use environment variable)
        if ml_api_key != settings.ML_API_KEY:
            raise exceptions.AuthenticationFailed('Invalid ML API key')

        return (None, None)  # No user, just API key validation

    def authenticate_header(self, request):
        return 'X-ML-API-Key'
    