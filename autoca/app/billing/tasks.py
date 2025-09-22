import logging
import stripe
from celery import shared_task
from django.conf import settings
from django.utils import timezone
from core.models import UserSubscription, AuditLog

logger = logging.getLogger(__name__)

# Configure Stripe
stripe.api_key = settings.STRIPE_SECRET_KEY

@shared_task
def handle_stripe_webhook(payload, signature):
    """
    Task 9: Stripe Webhook Handling for Subscription Management
    """
    try:
        event = stripe.Webhook.construct_event(
            payload, signature, settings.STRIPE_WEBHOOK_SECRET
        )

        event_type = event['type']
        data = event['data']['object']

        if event_type == 'customer.subscription.updated':
            _handle_subscription_update(data)
        elif event_type == 'invoice.payment_failed':
            _handle_payment_failed(data)
        elif event_type == 'customer.subscription.deleted':
            _handle_subscription_cancelled(data)

        logger.info(f"Processed Stripe webhook: {event_type}")
        return {'status': 'success', 'event_type': event_type}

    except stripe.error.SignatureVerificationError as e:
        logger.error(f"Invalid webhook signature: {str(e)}")
        return {'status': 'error', 'error': 'Invalid signature'}
    except Exception as e:
        logger.error(f"Webhook processing failed: {str(e)}")
        raise

@shared_task
def sync_subscription_status():
    """
    Periodic task to sync subscription status from Stripe
    """
    try:
        # Get active subscriptions that need syncing
        subscriptions = UserSubscription.objects.filter(status__in=['active', 'past_due'])

        for subscription in subscriptions:
            try:
                stripe_sub = stripe.Subscription.retrieve(subscription.stripe_subscription_id)
                _update_subscription_from_stripe(subscription, stripe_sub)
            except stripe.error.StripeError as e:
                logger.error(f"Failed to sync subscription {subscription.id}: {str(e)}")

        logger.info("Subscription sync completed")
        return {'status': 'success', 'synced_count': len(subscriptions)}

    except Exception as e:
        logger.error(f"Subscription sync failed: {str(e)}")
        raise

# Helper functions for billing tasks
def _handle_subscription_update(subscription_data):
    """Handle subscription update from Stripe"""
    try:
        subscription = UserSubscription.objects.get(
            stripe_subscription_id=subscription_data['id']
        )
        _update_subscription_from_stripe(subscription, subscription_data)
    except UserSubscription.DoesNotExist:
        logger.warning(f"Unknown subscription: {subscription_data['id']}")

def _update_subscription_from_stripe(subscription, stripe_data):
    """Update local subscription from Stripe data"""
    subscription.stripe_status = stripe_data['status']
    subscription.status = stripe_data['status']
    subscription.stripe_current_period_start = datetime.fromtimestamp(stripe_data['current_period_start'])
    subscription.stripe_current_period_end = datetime.fromtimestamp(stripe_data['current_period_end'])

    if stripe_data.get('canceled_at'):
        subscription.canceled_at = datetime.fromtimestamp(stripe_data['canceled_at'])

    subscription.save()

    # Log the update
    AuditLog.objects.create(
        user=subscription.user,
        action='subscription_updated',
        resource_type='subscription',
        resource_id=subscription.id,
        metadata={'stripe_status': stripe_data['status']}
    )

def _handle_payment_failed(invoice_data):
    """Handle failed payment"""
    try:
        subscription = UserSubscription.objects.get(
            stripe_subscription_id=invoice_data['subscription']
        )
        subscription.status = 'past_due'
        subscription.save()

        # TODO: Send notification to user
        logger.warning(f"Payment failed for subscription {subscription.id}")

    except UserSubscription.DoesNotExist:
        logger.warning(f"Unknown subscription for failed payment: {invoice_data['subscription']}")

def _handle_subscription_cancelled(subscription_data):
    """Handle subscription cancellation"""
    try:
        subscription = UserSubscription.objects.get(
            stripe_subscription_id=subscription_data['id']
        )
        subscription.status = 'canceled'
        subscription.canceled_at = timezone.now()
        subscription.save()

        logger.info(f"Subscription canceled: {subscription.id}")

    except UserSubscription.DoesNotExist:
        logger.warning(f"Unknown subscription canceled: {subscription_data['id']}")

