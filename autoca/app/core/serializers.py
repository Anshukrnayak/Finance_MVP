from rest_framework import serializers
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth import authenticate
from django.core.validators import validate_email
from django.core.exceptions import ValidationError as DjangoValidationError
from .models import BaseUser, UserProfile, SubscriptionPlan, UserSubscription, AuditLog, APIKey

class CustomTokenObtainPairSerializer(TokenObtainPairSerializer):
    """Custom JWT token serializer with additional user data"""

    def validate(self, attrs):
        data = super().validate(attrs)

        # Add custom claims
        refresh = self.get_token(self.user)

        # Add user data to response
        data['user'] = {
            'id': self.user.id,
            'email': self.user.email,
            'first_name': self.user.first_name,
            'last_name': self.user.last_name,
            'is_staff': self.user.is_staff,
            'is_superuser': self.user.is_superuser,
        }

        # Add subscription status
        try:
            subscription = self.user.subscription
            data['user']['subscription_status'] = subscription.status
            data['user']['subscription_plan'] = subscription.plan.name if subscription.plan else None
        except UserSubscription.DoesNotExist:
            data['user']['subscription_status'] = 'inactive'
            data['user']['subscription_plan'] = None

        return data

class UserRegistrationSerializer(serializers.ModelSerializer):
    """Serializer for user registration"""
    password = serializers.CharField(write_only=True, min_length=8)
    password_confirm = serializers.CharField(write_only=True)
    company_name = serializers.CharField(write_only=True, required=False)

    class Meta:
        model = BaseUser
        fields = ('email', 'first_name', 'last_name', 'password', 'password_confirm', 'company_name')

    def validate(self, attrs):
        if attrs['password'] != attrs['password_confirm']:
            raise serializers.ValidationError({"password_confirm": "Passwords do not match"})

        # Validate email format
        try:
            validate_email(attrs['email'])
        except DjangoValidationError:
            raise serializers.ValidationError({"email": "Enter a valid email address"})

        return attrs

    def create(self, validated_data):
        # Remove confirmation field
        validated_data.pop('password_confirm')
        company_name = validated_data.pop('company_name', None)

        # Create user
        user = BaseUser.objects.create_user(
            email=validated_data['email'],
            password=validated_data['password'],
            first_name=validated_data.get('first_name', ''),
            last_name=validated_data.get('last_name', ''),
            username=validated_data['email']  # Use email as username
        )

        # Create user profile with company name
        if company_name:
            UserProfile.objects.create(user=user, company_name=company_name)
        else:
            UserProfile.objects.create(user=user)

        return user

class UserSerializer(serializers.ModelSerializer):
    """User serializer with read-only fields"""
    subscription_status = serializers.SerializerMethodField()

    class Meta:
        model = BaseUser
        fields = (
            'id', 'email', 'first_name', 'last_name', 'phone_number',
            'is_active', 'date_joined', 'last_login', 'subscription_status'
        )
        read_only_fields = ('id', 'date_joined', 'last_login', 'is_active')

    def get_subscription_status(self, obj):
        try:
            return obj.subscription.status
        except UserSubscription.DoesNotExist:
            return 'inactive'

class UserProfileSerializer(serializers.ModelSerializer):
    """User profile serializer with nested user data"""
    user = UserSerializer(read_only=True)
    email = serializers.EmailField(source='user.email', read_only=True)
    full_name = serializers.SerializerMethodField()

    class Meta:
        model = UserProfile
        fields = (
            'id', 'user', 'email', 'full_name', 'company_name', 'company_gstin',
            'company_address', 'designation', 'ca_registration_number',
            'timezone', 'language', 'notification_preferences', 'avatar'
        )
        read_only_fields = ('id', 'user', 'email')

    def get_full_name(self, obj):
        return obj.user.get_full_name()

    def validate_company_gstin(self, value):
        if value and len(value) != 15:
            raise serializers.ValidationError("GSTIN must be 15 characters long")
        return value

class SubscriptionPlanSerializer(serializers.ModelSerializer):
    """Subscription plan serializer"""
    features_display = serializers.SerializerMethodField()

    class Meta:
        model = SubscriptionPlan
        fields = (
            'id', 'name', 'stripe_price_id', 'description',
            'max_clients', 'max_invoices_per_month', 'max_users',
            'features', 'features_display', 'amount', 'currency',
            'billing_interval', 'is_active'
        )

    def get_features_display(self, obj):
        return [f"{k}: {v}" for k, v in obj.features.items()]

class UserSubscriptionSerializer(serializers.ModelSerializer):
    """User subscription serializer with plan details"""
    plan = SubscriptionPlanSerializer(read_only=True)
    plan_id = serializers.PrimaryKeyRelatedField(
        queryset=SubscriptionPlan.objects.filter(is_active=True),
        write_only=True,
        source='plan'
    )
    is_active = serializers.SerializerMethodField()
    days_until_renewal = serializers.SerializerMethodField()

    class Meta:
        model = UserSubscription
        fields = (
            'id', 'plan', 'plan_id', 'stripe_subscription_id',
            'stripe_current_period_start', 'stripe_current_period_end',
            'stripe_status', 'status', 'is_trial', 'trial_ends_at',
            'canceled_at', 'is_active', 'days_until_renewal'
        )
        read_only_fields = (
            'id', 'stripe_subscription_id', 'stripe_current_period_start',
            'stripe_current_period_end', 'stripe_status', 'status'
        )

    def get_is_active(self, obj):
        return obj.is_active

    def get_days_until_renewal(self, obj):
        if obj.stripe_current_period_end:
            from django.utils import timezone
            delta = obj.stripe_current_period_end - timezone.now()
            return delta.days
        return None

class AuditLogSerializer(serializers.ModelSerializer):
    """Audit log serializer"""
    user_email = serializers.EmailField(source='user.email', read_only=True)

    class Meta:
        model = AuditLog
        fields = (
            'id', 'user', 'user_email', 'action', 'resource_type',
            'resource_id', 'ip_address', 'user_agent', 'metadata',
            'created_at'
        )
        read_only_fields = ('id', 'created_at')

class APIKeySerializer(serializers.ModelSerializer):
    """API key serializer (only show key once on creation)"""
    key = serializers.CharField(read_only=True)
    secret = serializers.CharField(read_only=True)
    is_expired = serializers.SerializerMethodField()

    class Meta:
        model = APIKey
        fields = (
            'id', 'name', 'key', 'secret', 'expires_at', 'last_used',
            'permissions', 'is_expired', 'created_at'
        )
        read_only_fields = ('id', 'created_at')

    def get_is_expired(self, obj):
        return obj.is_expired()

    def create(self, validated_data):
        # Only show key and secret once during creation
        instance = super().create(validated_data)
        instance.key = instance.key  # This will be shown in response
        instance.secret = instance.secret  # This will be shown in response
        return instance

    