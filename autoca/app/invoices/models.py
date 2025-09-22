from django.db import models
from django.core.validators import MinValueValidator
from django.utils.translation import gettext_lazy as _
from core.models import BaseUser, TimeStampedModel, SoftDeleteModel
import uuid
from datetime import datetime
import json

class Client(TimeStampedModel, SoftDeleteModel):
    """Client model with comprehensive business details"""
    user = models.ForeignKey(
        BaseUser,
        on_delete=models.CASCADE,
        related_name='clients',
        db_index=True
    )

    # Basic information
    client_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True, db_index=True)
    name = models.CharField(max_length=255, db_index=True)
    business_name = models.CharField(max_length=255, blank=True, db_index=True)

    # Contact information
    email = models.EmailField(blank=True, db_index=True)
    phone = models.CharField(max_length=20, blank=True, db_index=True)

    # Business details
    gstin = models.CharField(max_length=15, blank=True, db_index=True)
    pan = models.CharField(max_length=10, blank=True, db_index=True)
    business_type = models.CharField(
        max_length=50,
        choices=[
            ('proprietorship', 'Proprietorship'),
            ('partnership', 'Partnership'),
            ('llp', 'LLP'),
            ('pvt_ltd', 'Private Limited'),
            ('ltd', 'Public Limited'),
            ('other', 'Other')
        ],
        default='proprietorship',
        db_index=True
    )

    # Address information (structured)
    billing_address = models.JSONField(default=dict)
    shipping_address = models.JSONField(default=dict)

    # Additional metadata
    tags = models.JSONField(default=list, blank=True)
    notes = models.TextField(blank=True)
    is_active = models.BooleanField(default=True, db_index=True)

    class Meta:
        db_table = 'clients'
        indexes = [
            models.Index(fields=['user', 'is_deleted', 'is_active']),
            models.Index(fields=['gstin', 'pan']),
            models.Index(fields=['created_at', 'user']),
            models.Index(fields=['name', 'business_name']),
        ]
        unique_together = [['user', 'gstin'], ['user', 'email']]

    def __str__(self):
        return f"{self.name} ({self.business_name})"

class Invoice(TimeStampedModel, SoftDeleteModel):
    """Enhanced Invoice model with comprehensive tracking"""
    STATUS_CHOICES = [
        ('pending', 'Pending Processing'),
        ('processing', 'Processing'),
        ('processed', 'Processed Successfully'),
        ('failed', 'Processing Failed'),
        ('verified', 'Manually Verified'),
        ('filed', 'GST Filed'),
    ]

    user = models.ForeignKey(
        BaseUser,
        on_delete=models.CASCADE,
        related_name='invoices',
        db_index=True
    )
    client = models.ForeignKey(
        Client,
        on_delete=models.CASCADE,
        related_name='invoices',
        null=True,
        blank=True,
        db_index=True
    )

    # File information
    file_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True, db_index=True)
    uploaded_file = models.FileField(upload_to='invoices/%Y/%m/%d/')
    original_filename = models.CharField(max_length=255)
    file_type = models.CharField(
        max_length=10,
        choices=[('image', 'Image'), ('pdf', 'PDF')],
        db_index=True
    )
    file_size = models.PositiveIntegerField(default=0)  # in bytes
    file_hash = models.CharField(max_length=64, blank=True, db_index=True)  # SHA256 hash

    # Processing information
    processing_status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='pending',
        db_index=True
    )
    processing_attempts = models.PositiveIntegerField(default=0)
    last_processing_error = models.TextField(blank=True)

    # Extracted data (structured)
    extracted_data = models.JSONField(default=dict)
    confidence_scores = models.JSONField(default=dict)

    # Validation flags
    is_gstin_valid = models.BooleanField(null=True, db_index=True)
    is_amount_valid = models.BooleanField(null=True, db_index=True)
    is_date_valid = models.BooleanField(null=True, db_index=True)

    # Fraud detection
    fraud_score = models.FloatField(null=True, blank=True, db_index=True)
    fraud_risk_level = models.CharField(
        max_length=20,
        choices=[
            ('low', 'Low Risk'),
            ('medium', 'Medium Risk'),
            ('high', 'High Risk'),
            ('critical', 'Critical Risk')
        ],
        null=True,
        blank=True,
        db_index=True
    )

    # GST filing information
    gst_filing_status = models.CharField(
        max_length=20,
        choices=[
            ('not_filed', 'Not Filed'),
            ('ready_to_file', 'Ready to File'),
            ('filed', 'Filed'),
            ('rejected', 'Rejected')
        ],
        default='not_filed',
        db_index=True
    )
    gstr_transaction_id = models.CharField(max_length=100, blank=True, db_index=True)

    # Timestamps with indexing
    uploaded_at = models.DateTimeField(auto_now_add=True, db_index=True)
    processed_at = models.DateTimeField(null=True, blank=True, db_index=True)
    verified_at = models.DateTimeField(null=True, blank=True, db_index=True)
    filed_at = models.DateTimeField(null=True, blank=True, db_index=True)

    class Meta:
        db_table = 'invoices'
    indexes = [
        # Composite indexes for common queries
        models.Index(fields=['user', 'processing_status', 'uploaded_at']),
        models.Index(fields=['user', 'fraud_risk_level', 'uploaded_at']),
        models.Index(fields=['user', 'gst_filing_status', 'uploaded_at']),
        models.Index(fields=['client', 'uploaded_at']),
        models.Index(fields=['file_hash', 'user']),  # Duplicate detection
        models.Index(fields=['uploaded_at', 'processing_status']),

        # Partial indexes for better performance - MUST HAVE NAMES
        models.Index(
            name='high_fraud_score_idx',
            fields=['fraud_score'],
            condition=models.Q(fraud_score__gt=0.7)
        ),
        models.Index(
            name='pending_invoices_idx',
            fields=['processing_status'],
            condition=models.Q(processing_status='pending')
        ),
    ]
    ordering = ['-uploaded_at']

    def __str__(self):
        return f"Invoice {self.file_id} - {self.original_filename}"

    @property
    def extracted_amount(self):
        return self.extracted_data.get('amount')

    @property
    def extracted_gstin(self):
        return self.extracted_data.get('gstin')

    @property
    def extracted_date(self):
        return self.extracted_data.get('date')

    def mark_as_processing(self):
        self.processing_status = 'processing'
        self.processing_attempts += 1
        self.save()

    def mark_as_processed(self, extracted_data, confidence_scores):
        self.processing_status = 'processed'
        self.extracted_data = extracted_data
        self.confidence_scores = confidence_scores
        self.processed_at = timezone.now()
        self.save()

    def mark_as_failed(self, error_message):
        self.processing_status = 'failed'
        self.last_processing_error = error_message
        self.save()

class InvoiceBatch(TimeStampedModel):
    """Batch processing for multiple invoices"""
    user = models.ForeignKey(
        BaseUser,
        on_delete=models.CASCADE,
        related_name='invoice_batches',
        db_index=True
    )
    batch_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True, db_index=True)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)

    # Batch status
    status = models.CharField(
        max_length=20,
        choices=[
            ('processing', 'Processing'),
            ('completed', 'Completed'),
            ('partial', 'Partially Completed'),
            ('failed', 'Failed')
        ],
        default='processing',
        db_index=True
    )

    # Statistics
    total_invoices = models.PositiveIntegerField(default=0)
    successful_invoices = models.PositiveIntegerField(default=0)
    failed_invoices = models.PositiveIntegerField(default=0)

    # Metadata
    metadata = models.JSONField(default=dict)

    class Meta:
        db_table = 'invoice_batches'
        indexes = [
            models.Index(fields=['user', 'status', 'created_at']),
            models.Index(fields=['batch_id', 'user']),
        ]

    def __str__(self):
        return f"Batch {self.batch_id} - {self.name}"

class GSTRFiling(TimeStampedModel):
    """GSTR filing management with comprehensive tracking"""
    user = models.ForeignKey(
        BaseUser,
        on_delete=models.CASCADE,
        related_name='gstr_filings',
        db_index=True
    )
    filing_period = models.CharField(max_length=7, db_index=True)  # Format: YYYY-MM
    filing_type = models.CharField(
        max_length=10,
        choices=[
            ('gstr1', 'GSTR-1'),
            ('gstr3b', 'GSTR-3B'),
            ('gstr9', 'GSTR-9')
        ],
        db_index=True
    )

    # Filing status
    status = models.CharField(
        max_length=20,
        choices=[
            ('draft', 'Draft'),
            ('ready', 'Ready to File'),
            ('filed', 'Filed'),
            ('acknowledged', 'Acknowledged'),
            ('rejected', 'Rejected')
        ],
        default='draft',
        db_index=True
    )

    # GSTN integration
    gstn_acknowledgement_ref = models.CharField(max_length=100, blank=True, db_index=True)
    gstn_filing_date = models.DateTimeField(null=True, blank=True, db_index=True)

    # Financial summary
    total_taxable_value = models.DecimalField(max_digits=15, decimal_places=2, default=0)
    total_tax_amount = models.DecimalField(max_digits=15, decimal_places=2, default=0)
    total_liability = models.DecimalField(max_digits=15, decimal_places=2, default=0)

    # Included invoices
    invoices = models.ManyToManyField(Invoice, through='GSTRFilingItem')

    # Error handling
    error_message = models.TextField(blank=True)
    retry_count = models.PositiveIntegerField(default=0)

    class Meta:
        db_table = 'gstr_filings'
        indexes = [
            models.Index(fields=['user', 'filing_period', 'filing_type']),
            models.Index(fields=['gstn_acknowledgement_ref']),
            models.Index(fields=['status', 'created_at']),
        ]
        unique_together = [['user', 'filing_period', 'filing_type']]

    def __str__(self):
        return f"{self.filing_type} - {self.filing_period}"

class GSTRFilingItem(TimeStampedModel):
    """Linking table between GSTR filings and invoices"""
    filing = models.ForeignKey(GSTRFiling, on_delete=models.CASCADE, db_index=True)
    invoice = models.ForeignKey(Invoice, on_delete=models.CASCADE, db_index=True)
    taxable_value = models.DecimalField(max_digits=15, decimal_places=2)
    tax_amount = models.DecimalField(max_digits=15, decimal_places=2)

    class Meta:
        db_table = 'gstr_filing_items'
        indexes = [
            models.Index(fields=['filing', 'invoice']),
        ]
        unique_together = [['filing', 'invoice']]

