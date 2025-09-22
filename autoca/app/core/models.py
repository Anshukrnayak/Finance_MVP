from django.db import models
from django.contrib.auth.models import AbstractUser
from django.utils.translation import gettext_lazy as _
from django.core.validators import RegexValidator
from django.conf import settings
import uuid
from datetime import datetime, timedelta
from django.utils import timezone

class TimeStampedModel(models.Model):
    """Abstract base model with created and updated timestamps"""
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True, db_index=True)

    class Meta:
        abstract = True

class SoftDeleteModel(models.Model):
    """Abstract base model for soft deletion"""
    is_deleted = models.BooleanField(default=False, db_index=True)
    deleted_at = models.DateTimeField(null=True, blank=True, db_index=True)

    class Meta:
        abstract = True

    def soft_delete(self):
        self.is_deleted = True
        self.deleted_at = timezone.now()
        self.save()

class BaseUser(AbstractUser, TimeStampedModel, SoftDeleteModel):
    """Extended User model with additional fields and soft delete"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    email = models.EmailField(_('email address'), unique=True, db_index=True)
    phone_regex = RegexValidator(
        regex=r'^\+?1?\d{9,15}$',
        message="Phone number must be entered in the format: '+999999999'. Up to 15 digits allowed."
    )
    phone_number = models.CharField(validators=[phone_regex], max_length=17, blank=True, db_index=True)

    # Stripe customer ID for payment processing
    stripe_customer_id = models.CharField(max_length=255, blank=True, db_index=True)

    # Email verification
    email_verified = models.BooleanField(default=False)
    email_verification_token = models.UUIDField(default=uuid.uuid4, editable=False)

    # Audit fields
    last_login_ip = models.GenericIPAddressField(null=True, blank=True)
    last_activity = models.DateTimeField(null=True, blank=True, db_index=True)

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username']

    class Meta:
        db_table = 'auth_user'
        indexes = [
            models.Index(fields=['email', 'is_deleted']),
            models.Index(fields=['username', 'is_deleted']),
            models.Index(fields=['created_at', 'is_deleted']),
        ]

    def __str__(self):
        return f"{self.email} ({self.get_full_name() or self.username})"

    def get_display_name(self):
        return self.get_full_name() or self.email

class UserProfile(TimeStampedModel):
    """Separate profile model for user-specific data"""
    user = models.OneToOneField(
        BaseUser,
        on_delete=models.CASCADE,
        related_name='profile',
        db_index=True
    )

    # Company information
    company_name = models.CharField(max_length=255, blank=True, db_index=True)
    company_gstin = models.CharField(max_length=15, blank=True, db_index=True)
    company_address = models.JSONField(default=dict, blank=True)

    # Professional details
    designation = models.CharField(max_length=100, blank=True)
    ca_registration_number = models.CharField(max_length=50, blank=True, db_index=True)

    # Preferences
    timezone = models.CharField(max_length=50, default='UTC')
    language = models.CharField(max_length=10, default='en')
    notification_preferences = models.JSONField(default=dict)

    # Avatar/Profile picture
    avatar = models.ImageField(upload_to='avatars/', null=True, blank=True)

    class Meta:
        db_table = 'user_profiles'
        indexes = [
            models.Index(fields=['company_name', 'company_gstin']),
            models.Index(fields=['ca_registration_number']),
        ]

    def __str__(self):
        return f"Profile of {self.user.email}"

class SubscriptionPlan(TimeStampedModel):
    """Subscription plans for different user tiers"""
    name = models.CharField(max_length=100, db_index=True)
    stripe_price_id = models.CharField(max_length=255, db_index=True)
    description = models.TextField(blank=True)

    # Limits and features
    max_clients = models.PositiveIntegerField(default=10)
    max_invoices_per_month = models.PositiveIntegerField(default=100)
    max_users = models.PositiveIntegerField(default=1)
    features = models.JSONField(default=dict)

    # Pricing
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    currency = models.CharField(max_length=3, default='USD')
    billing_interval = models.CharField(
        max_length=10,
        choices=[('month', 'Monthly'), ('year', 'Yearly')],
        default='month'
    )

    # Status
    is_active = models.BooleanField(default=True, db_index=True)

    class Meta:
        db_table = 'subscription_plans'
        indexes = [
            models.Index(fields=['stripe_price_id', 'is_active']),
            models.Index(fields=['amount', 'billing_interval']),
        ]

    def __str__(self):
        return f"{self.name} - ${self.amount}/{self.billing_interval}"

class UserSubscription(TimeStampedModel):
    """User subscription management with Stripe integration"""
    user = models.OneToOneField(
        BaseUser,
        on_delete=models.CASCADE,
        related_name='subscription',
        db_index=True
    )
    plan = models.ForeignKey(
        SubscriptionPlan,
        on_delete=models.PROTECT,
        related_name='subscriptions',
        db_index=True
    )

    # Stripe subscription details
    stripe_subscription_id = models.CharField(max_length=255, db_index=True)
    stripe_current_period_start = models.DateTimeField()
    stripe_current_period_end = models.DateTimeField()
    stripe_status = models.CharField(max_length=50, db_index=True)

    # Local status management
    status = models.CharField(
        max_length=20,
        choices=[
            ('active', 'Active'),
            ('past_due', 'Past Due'),
            ('canceled', 'Canceled'),
            ('incomplete', 'Incomplete'),
            ('trialing', 'Trialing')
        ],
        db_index=True
    )

    # Trial information
    is_trial = models.BooleanField(default=False)
    trial_ends_at = models.DateTimeField(null=True, blank=True)

    # Cancellation
    canceled_at = models.DateTimeField(null=True, blank=True, db_index=True)

    class Meta:
        db_table = 'user_subscriptions'
        indexes = [
            models.Index(fields=['stripe_subscription_id', 'status']),
            models.Index(fields=['status', 'stripe_current_period_end']),
            models.Index(fields=['user', 'status']),
        ]

    def __str__(self):
        return f"{self.user.email} - {self.plan.name}"

    @property
    def is_active(self):
        return self.status == 'active' or (self.is_trial and self.trial_ends_at > timezone.now())

    def can_add_client(self, current_count):
        return current_count < self.plan.max_clients

    def can_process_invoice(self, current_month_count):
        return current_month_count < self.plan.max_invoices_per_month

class AuditLog(TimeStampedModel):
    """Comprehensive audit logging for user actions"""
    user = models.ForeignKey(
        BaseUser,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        db_index=True
    )
    action = models.CharField(max_length=100, db_index=True)
    resource_type = models.CharField(max_length=100, db_index=True)
    resource_id = models.CharField(max_length=100, db_index=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True)
    metadata = models.JSONField(default=dict)

    class Meta:
        db_table = 'audit_logs'
        indexes = [
            models.Index(fields=['user', 'action', 'created_at']),
            models.Index(fields=['resource_type', 'resource_id']),
            models.Index(fields=['created_at', 'action']),
        ]
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.user.email if self.user else 'System'} - {self.action}"

class APIKey(TimeStampedModel, SoftDeleteModel):
    """API key management for third-party integrations"""
    user = models.ForeignKey(
        BaseUser,
        on_delete=models.CASCADE,
        related_name='api_keys',
        db_index=True
    )
    name = models.CharField(max_length=100)
    key = models.UUIDField(default=uuid.uuid4, editable=False, unique=True, db_index=True)
    secret = models.UUIDField(default=uuid.uuid4, editable=False, unique=True, db_index=True)
    expires_at = models.DateTimeField(null=True, blank=True, db_index=True)
    last_used = models.DateTimeField(null=True, blank=True, db_index=True)
    permissions = models.JSONField(default=dict)

    class Meta:
        db_table = 'api_keys'
        indexes = [
            models.Index(fields=['key', 'is_deleted']),
            models.Index(fields=['user', 'is_deleted']),
        ]

    def __str__(self):
        return f"{self.name} - {self.user.email}"

    def is_expired(self):
        return self.expires_at and self.expires_at < timezone.now()
