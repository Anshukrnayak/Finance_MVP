import os
from celery import Celery
from celery.signals import after_setup_logger, after_setup_task_logger
import logging

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'autoca.settings.dev')

app = Celery('autoca')

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Load task modules from all registered Django apps.
app.autodiscover_tasks()

# Configure task routes for better scalability
app.conf.task_routes = {
    'invoices.tasks.process_invoice_task': {'queue': 'invoices'},
    'invoices.tasks.batch_process_invoices': {'queue': 'batches'},
    'invoices.tasks.raman_fraud_detection_task': {'queue': 'analytics'},
    'invoices.tasks.quantum_audit_simulation_task': {'queue': 'analytics'},
    'invoices.tasks.tda_anomaly_detection_task': {'queue': 'analytics'},
    'compliance.tasks.file_gstr3b_task': {'queue': 'compliance'},
    'compliance.tasks.sync_tally_ledger_task': {'queue': 'integrations'},
    'compliance.tasks.generate_xbrl_report_task': {'queue': 'reports'},
    'billing.tasks.handle_stripe_webhook': {'queue': 'billing'},
    'billing.tasks.sync_subscription_status': {'queue': 'billing'},
}

# Task timeouts and retry settings
app.conf.task_time_limit = 3600  # 1 hour max per task
app.conf.task_soft_time_limit = 3000  # 50 minutes soft limit
app.conf.task_acks_late = True  # Ack after task completion
app.conf.task_reject_on_worker_lost = True
app.conf.task_track_started = True

# Rate limiting
app.conf.task_annotations = {
    'invoices.tasks.process_invoice_task': {'rate_limit': '10/m'},
    'invoices.tasks.batch_process_invoices': {'rate_limit': '2/m'},
    'compliance.tasks.file_gstr3b_task': {'rate_limit': '5/h'},
}

@after_setup_logger.connect
def setup_logger(logger, *args, **kwargs):
    logger.setLevel(logging.INFO)

@after_setup_task_logger.connect
def setup_task_logger(logger, *args, **kwargs):
    logger.setLevel(logging.INFO)

@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')

    