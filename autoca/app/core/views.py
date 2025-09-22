import logging
import json
from django.http import JsonResponse, HttpResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.db import transaction
from django.core.exceptions import PermissionDenied, ValidationError
from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions
from rest_framework.decorators import api_view, permission_classes
from ratelimit.decorators import ratelimit
from ratelimit.exceptions import Ratelimited

from .models import BaseUser, UserProfile, AuditLog
from .serializers import UserSerializer, UserProfileSerializer
from .authentication import JWTAuthentication
from .permissions import IsOwner, IsAdminUser, HasSubscriptionAccess

logger = logging.getLogger(__name__)

class BaseAPIView(APIView):
    """Base API view with common functionality"""
    authentication_classes = [JWTAuthentication]
    permission_classes = [permissions.IsAuthenticated]

    def handle_exception(self, exc):
        """Global exception handler for API views"""
        if isinstance(exc, PermissionDenied):
            logger.warning(f"Permission denied for user: {self.request.user}")
            return Response(
                {"error": "Permission denied"},
                status=status.HTTP_403_FORBIDDEN
            )
        elif isinstance(exc, ValidationError):
            logger.warning(f"Validation error: {exc}")
            return Response(
                {"error": str(exc)},
                status=status.HTTP_400_BAD_REQUEST
            )
        elif isinstance(exc, Ratelimited):
            logger.warning(f"Rate limit exceeded for user: {self.request.user}")
            return Response(
                {"error": "Rate limit exceeded"},
                status=status.HTTP_429_TOO_MANY_REQUESTS
            )

        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return Response(
            {"error": "Internal server error"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

class UserProfileView(BaseAPIView):
    """User profile management"""

    @ratelimit(key='user', rate='10/m', method='GET')
    def get(self, request):
        try:
            profile = UserProfile.objects.get(user=request.user)
            serializer = UserProfileSerializer(profile)
            return Response(serializer.data)
        except UserProfile.DoesNotExist:
            return Response(
                {"error": "Profile not found"},
                status=status.HTTP_404_NOT_FOUND
            )

    @ratelimit(key='user', rate='5/m', method='POST')
    def post(self, request):
        try:
            with transaction.atomic():
                profile, created = UserProfile.objects.get_or_create(
                    user=request.user,
                    defaults=request.data
                )
                if not created:
                    serializer = UserProfileSerializer(profile, data=request.data)
                else:
                    serializer = UserProfileSerializer(profile)

                if serializer.is_valid():
                    serializer.save()
                    return Response(serializer.data, status=status.HTTP_200_OK if not created else status.HTTP_201_CREATED)
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        except Exception as e:
            logger.error(f"Profile update failed: {e}")
            return Response(
                {"error": "Failed to update profile"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

@api_view(['POST'])
@permission_classes([permissions.AllowAny])
@ratelimit(key='ip', rate='5/m', method='POST')
def register_user(request):
    """User registration endpoint"""
    try:
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            # Create user profile
            UserProfile.objects.create(user=user)

            # Log registration
            AuditLog.objects.create(
                action='user_registered',
                resource_type='user',
                resource_id=user.id,
                ip_address=request.META.get('REMOTE_ADDR'),
                user_agent=request.META.get('HTTP_USER_AGENT', '')
            )

            return Response(
                {"message": "User created successfully", "user_id": user.id},
                status=status.HTTP_201_CREATED
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    except Exception as e:
        logger.error(f"User registration failed: {e}")
        return Response(
            {"error": "Registration failed"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )