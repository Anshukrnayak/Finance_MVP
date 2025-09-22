import logging
import time
import json
from celery import shared_task, group, chord
from celery.exceptions import MaxRetriesExceededError, SoftTimeLimitExceeded
from django.db import transaction
from django.utils import timezone
from django.core.cache import cache
from django.conf import settings
import easyocr
from PIL import Image
import pdf2image
import numpy as np
import re
from datetime import datetime
from typing import Dict, List, Optional

from core.models import AuditLog
from invoices.models import Invoice, InvoiceBatch, Client
from .utils import validate_gstin, extract_amount, extract_date

logger = logging.getLogger(__name__)

# Initialize EasyOCR reader once (thread-safe)
reader = easyocr.Reader(['en'])

@shared_task(
    bind=True,
    max_retries=3,
    retry_backoff=True,
    retry_jitter=True,
    retry_backoff_max=600,  # Max 10 minutes between retries
    time_limit=300,  # 5 minutes per invoice
    soft_time_limit=240,  # 4 minutes soft limit
    autoretry_for=(Exception,),
    default_retry_delay=60  # 1 minute initial retry delay
)
def process_invoice_task(self, invoice_id: int, batch_id: Optional[int] = None):
    """
    Task 1: AI-Powered Invoice Processing with EasyOCR
    """
    try:
        invoice = Invoice.objects.select_related('user', 'client').get(id=invoice_id)
        invoice.mark_as_processing()

        logger.info(f"Processing invoice {invoice_id} for user {invoice.user.email}")

        # Read file content
        file_content = invoice.uploaded_file.read()

        # Extract text based on file type
        if invoice.file_type == 'pdf':
            text = _extract_text_from_pdf(file_content)
        else:
            text = _extract_text_from_image(file_content)

        # Parse extracted text
        extracted_data = _parse_invoice_text(text)

        # Validate and enhance data
        validated_data = _validate_extracted_data(extracted_data)

        # Auto-match client if possible
        client = _auto_match_client(invoice.user, validated_data.get('gstin'))
        if client:
            invoice.client = client

        # Save results
        confidence_scores = _calculate_confidence_scores(validated_data)
        invoice.mark_as_processed(validated_data, confidence_scores)

        # Trigger downstream tasks
        if validated_data.get('amount') and validated_data.get('gstin'):
            chain = (
                    raman_fraud_detection_task.si(invoice_id)
                    | quantum_audit_simulation_task.si(invoice_id)
                    | tda_anomaly_detection_task.si(invoice_id)
            )
            chain.apply_async()

        # Update batch if exists
        if batch_id:
            _update_batch_status(batch_id, success=True)

        logger.info(f"Successfully processed invoice {invoice_id}")
        return {'status': 'success', 'invoice_id': invoice_id}

    except SoftTimeLimitExceeded:
        logger.warning(f"Invoice {invoice_id} processing timed out")
        invoice.mark_as_failed("Processing timeout")
        raise self.retry(countdown=120)
    except Exception as e:
        logger.error(f"Error processing invoice {invoice_id}: {str(e)}")
        invoice.mark_as_failed(str(e))
        raise self.retry(exc=e)

@shared_task
def batch_process_invoices(batch_id: int, invoice_ids: List[int]):
    """
    Process multiple invoices in a batch with parallel processing
    """
    try:
        batch = InvoiceBatch.objects.get(id=batch_id)
        batch.total_invoices = len(invoice_ids)
        batch.save()

        # Create chord for parallel processing with callback
        header = [process_invoice_task.s(invoice_id, batch_id) for invoice_id in invoice_ids]
        callback = batch_processing_complete_task.s(batch_id)

        chord(header)(callback)

    except Exception as e:
        logger.error(f"Batch processing failed for batch {batch_id}: {str(e)}")
        batch.status = 'failed'
        batch.save()

@shared_task
def batch_processing_complete_task(results, batch_id: int):
    """
    Callback after batch processing completes
    """
    try:
        batch = InvoiceBatch.objects.get(id=batch_id)
        successful = sum(1 for r in results if r.get('status') == 'success')

        batch.successful_invoices = successful
        batch.failed_invoices = batch.total_invoices - successful
        batch.status = 'completed' if successful == batch.total_invoices else 'partial'
        batch.save()

    except Exception as e:
        logger.error(f"Batch completion processing failed: {str(e)}")

# Helper functions for invoice processing
def _extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF using pdf2image and EasyOCR"""
    try:
        images = pdf2image.convert_from_bytes(file_content)
        text_chunks = []

        for image in images:
            img_array = np.array(image)
            results = reader.readtext(img_array, detail=0)
            text_chunks.extend(results)

        return " ".join(text_chunks)
    except Exception as e:
        logger.error(f"PDF extraction failed: {str(e)}")
        raise

def _extract_text_from_image(file_content: bytes) -> str:
    """Extract text from image using EasyOCR"""
    try:
        image = Image.open(io.BytesIO(file_content))
        img_array = np.array(image)
        results = reader.readtext(img_array, detail=0)
        return " ".join(results)
    except Exception as e:
        logger.error(f"Image extraction failed: {str(e)}")
        raise

def _parse_invoice_text(text: str) -> Dict:
    """Parse extracted text to find invoice data"""
    patterns = {
        'gstin': r'\b\d{2}[A-Z]{5}\d{4}[A-Z]{1}\d{1}[A-Z]{1}\d{1}\b',
        'amount': r'(?:Total|Amount|Amt|Rs\.?|₹)\s*[:\-]?\s*([\d,]+\.?\d*)',
        'date': r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b',
        'invoice_number': r'(?:Invoice\s*No\.?|INV|Bill\s*No\.?)\s*[:\-]?\s*([A-Z0-9\-]+)',
    }

    extracted = {}
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            extracted[key] = matches[0] if isinstance(matches[0], str) else matches[0][0]

    return extracted

def _validate_extracted_data(data: Dict) -> Dict:
    """Validate and clean extracted data"""
    validated = data.copy()

    if 'gstin' in validated:
        validated['is_gstin_valid'] = validate_gstin(validated['gstin'])

    if 'amount' in validated:
        validated['amount'] = extract_amount(validated['amount'])
        validated['is_amount_valid'] = validated['amount'] is not None

    if 'date' in validated:
        validated['date'] = extract_date(validated['date'])
        validated['is_date_valid'] = validated['date'] is not None

    return validated

def _calculate_confidence_scores(data: Dict) -> Dict:
    """Calculate confidence scores for extracted data"""
    scores = {}

    if data.get('gstin'):
        scores['gstin'] = 0.95 if data.get('is_gstin_valid') else 0.7

    if data.get('amount'):
        scores['amount'] = 0.85 if data.get('is_amount_valid') else 0.5

    if data.get('date'):
        scores['date'] = 0.9 if data.get('is_date_valid') else 0.6

    return scores

def _auto_match_client(user, gstin: Optional[str]) -> Optional[Client]:
    """Auto-match client based on GSTIN"""
    if not gstin:
        return None

    try:
        return Client.objects.get(user=user, gstin=gstin, is_deleted=False)
    except Client.DoesNotExist:
        return None

def _update_batch_status(batch_id: int, success: bool):
    """Update batch processing statistics"""
    try:
        batch = InvoiceBatch.objects.get(id=batch_id)
        if success:
            batch.successful_invoices += 1
        else:
            batch.failed_invoices += 1
        batch.save()
    except Exception as e:
        logger.error(f"Failed to update batch {batch_id}: {str(e)}")

