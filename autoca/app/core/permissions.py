from rest_framework import permissions
from rest_framework.exceptions import PermissionDenied
from .models import UserSubscription

class HasMLAccess(permissions.BasePermission):
    """Check if user has access to ML features"""

    def has_permission(self, request, view):
        if not request.user.is_authenticated:
            return False

        # Check subscription for ML features
        try:
            subscription = request.user.subscription
            if not subscription.is_active:
                raise PermissionDenied("Active subscription required for ML features")

            # Check if plan includes ML features
            plan = subscription.plan
            if plan and plan.features.get('ml_analytics', False):
                return True

            raise PermissionDenied("ML features not included in your plan")

        except UserSubscription.DoesNotExist:
            raise PermissionDenied("Subscription required for ML features")

class HasSufficientData(permissions.BasePermission):
    """Check if user has sufficient data for ML analysis"""

    def has_permission(self, request, view):
        from invoices.models import Invoice

        # Count processed invoices
        processed_count = Invoice.objects.filter(
            user=request.user,
            processing_status='processed'
        ).count()

        # Minimum 10 invoices for meaningful ML analysis
        if processed_count < 10:
            raise PermissionDenied("Insufficient data for ML analysis. Minimum 10 processed invoices required.")

        return True

class CanUseAdvancedML(permissions.BasePermission):
    """Check if user can use advanced ML features"""

    def has_permission(self, request, view):
        # Check subscription tier
        try:
            subscription = request.user.subscription
            plan = subscription.plan

            if plan and plan.features.get('advanced_ml', False):
                return True

            raise PermissionDenied("Advanced ML features require premium subscription")

        except UserSubscription.DoesNotExist:
            raise PermissionDenied("Premium subscription required for advanced ML features")

class APIKeyPermission(permissions.BasePermission):
    """Permission for API key access"""

    def has_permission(self, request, view):
        if hasattr(request, 'api_key'):
            # Check API key permissions
            required_permissions = getattr(view, 'required_permissions', [])
            api_key_permissions = request.api_key.permissions or {}

            for perm in required_permissions:
                if not api_key_permissions.get(perm, False):
                    raise PermissionDenied(f"API key missing permission: {perm}")

            return True
        return False

class IsOwnerOrMLSystem(permissions.BasePermission):
    """Allow access to owner or ML system"""

    def has_object_permission(self, request, view, obj):
        # ML system can access any object
        if hasattr(request, 'ml_system') and request.ml_system:
            return True

        # Owner can access their objects
        if hasattr(obj, 'user'):
            return obj.user == request.user

        return False

    