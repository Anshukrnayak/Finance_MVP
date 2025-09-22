import logging
import uuid
from django.http import JsonResponse, HttpResponse
from django.db import transaction
from django.core.exceptions import ValidationError
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, parsers
from rest_framework.decorators import api_view, parser_classes
from ratelimit.decorators import ratelimit

from core.views import BaseAPIView
from core.models import AuditLog
from .models import Invoice, InvoiceBatch, Client
from .serializers import (
    InvoiceSerializer,
    InvoiceBatchSerializer,
    ClientSerializer
)
from .tasks import process_invoice_task, batch_process_invoices
from .utils import validate_file_type, generate_file_hash

logger = logging.getLogger(__name__)

class InvoiceUploadView(BaseAPIView):
    """Handle single invoice upload and processing"""
    parser_classes = [parsers.MultiPartParser, parsers.FormParser]

    @ratelimit(key='user', rate='10/m', method='POST')
    def post(self, request):
        try:
            if 'file' not in request.FILES:
                return Response(
                    {"error": "No file provided"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            uploaded_file = request.FILES['file']
            file_type = validate_file_type(uploaded_file.name)

            if not file_type:
                return Response(
                    {"error": "Invalid file type. Supported: PDF, JPEG, PNG"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Check subscription limits
            if not request.user.subscription.can_process_invoice(
                    request.user.invoices.filter(processing_status='processed').count()
            ):
                return Response(
                    {"error": "Subscription limit exceeded"},
                    status=status.HTTP_402_PAYMENT_REQUIRED
                )

            with transaction.atomic():
                # Check for duplicate files
                file_hash = generate_file_hash(uploaded_file.read())
                uploaded_file.seek(0)  # Reset file pointer

                duplicate = Invoice.objects.filter(
                    user=request.user,
                    file_hash=file_hash,
                    is_deleted=False
                ).first()

                if duplicate:
                    return Response(
                        {
                            "warning": "Duplicate file detected",
                            "existing_invoice_id": duplicate.id
                        },
                        status=status.HTTP_409_CONFLICT
                    )

                # Create invoice record
                invoice = Invoice.objects.create(
                    user=request.user,
                    uploaded_file=uploaded_file,
                    original_filename=uploaded_file.name,
                    file_type=file_type,
                    file_size=uploaded_file.size,
                    file_hash=file_hash
                )

                # Start processing task
                process_invoice_task.delay(invoice.id)

                # Log action
                AuditLog.objects.create(
                    user=request.user,
                    action='invoice_uploaded',
                    resource_type='invoice',
                    resource_id=invoice.id,
                    ip_address=request.META.get('REMOTE_ADDR'),
                    metadata={'filename': uploaded_file.name}
                )

                return Response(
                    {
                        "message": "Invoice uploaded and processing started",
                        "invoice_id": invoice.id,
                        "file_id": invoice.file_id
                    },
                    status=status.HTTP_202_ACCEPTED
                )

        except ValidationError as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            logger.error(f"Invoice upload failed: {e}")
            return Response(
                {"error": "Failed to process invoice"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class BatchInvoiceUploadView(BaseAPIView):
    """Handle batch invoice uploads"""
    parser_classes = [parsers.MultiPartParser, parsers.FormParser]

    @ratelimit(key='user', rate='3/m', method='POST')
    def post(self, request):
        try:
            files = request.FILES.getlist('files')
            if not files:
                return Response(
                    {"error": "No files provided"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Check batch limits
            if len(files) > 50:
                return Response(
                    {"error": "Maximum 50 files per batch"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            with transaction.atomic():
                batch = InvoiceBatch.objects.create(
                    user=request.user,
                    name=request.data.get('name', f'Batch {uuid.uuid4().hex[:8]}'),
                    description=request.data.get('description', '')
                )

                invoice_ids = []
                for uploaded_file in files:
                    file_type = validate_file_type(uploaded_file.name)
                    if file_type:
                        invoice = Invoice.objects.create(
                            user=request.user,
                            uploaded_file=uploaded_file,
                            original_filename=uploaded_file.name,
                            file_type=file_type,
                            file_size=uploaded_file.size,
                            file_hash=generate_file_hash(uploaded_file.read())
                        )
                        invoice_ids.append(invoice.id)

                # Start batch processing
                batch_process_invoices.delay(batch.id, invoice_ids)

                AuditLog.objects.create(
                    user=request.user,
                    action='batch_uploaded',
                    resource_type='batch',
                    resource_id=batch.id,
                    metadata={'file_count': len(invoice_ids)}
                )

                return Response(
                    {
                        "message": "Batch processing started",
                        "batch_id": batch.id,
                        "total_files": len(invoice_ids)
                    },
                    status=status.HTTP_202_ACCEPTED
                )

        except Exception as e:
            logger.error(f"Batch upload failed: {e}")
            return Response(
                {"error": "Failed to process batch"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class InvoiceDetailView(BaseAPIView):
    """Retrieve and manage individual invoices"""

    @ratelimit(key='user', rate='30/m', method='GET')
    def get(self, request, invoice_id):
        try:
            invoice = get_object_or_404(
                Invoice,
                id=invoice_id,
                user=request.user,
                is_deleted=False
            )
            serializer = InvoiceSerializer(invoice)
            return Response(serializer.data)

        except Exception as e:
            logger.error(f"Invoice retrieval failed: {e}")
            return Response(
                {"error": "Failed to retrieve invoice"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @ratelimit(key='user', rate='10/m', method='PUT')
    def put(self, request, invoice_id):
        try:
            invoice = get_object_or_404(
                Invoice,
                id=invoice_id,
                user=request.user,
                is_deleted=False
            )

            serializer = InvoiceSerializer(invoice, data=request.data, partial=True)
            if serializer.is_valid():
                serializer.save()

                AuditLog.objects.create(
                    user=request.user,
                    action='invoice_updated',
                    resource_type='invoice',
                    resource_id=invoice.id,
                    metadata={'updates': request.data}
                )

                return Response(serializer.data)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        except Exception as e:
            logger.error(f"Invoice update failed: {e}")
            return Response(
                {"error": "Failed to update invoice"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @ratelimit(key='user', rate='5/m', method='DELETE')
    def delete(self, request, invoice_id):
        try:
            invoice = get_object_or_404(
                Invoice,
                id=invoice_id,
                user=request.user,
                is_deleted=False
            )

            invoice.soft_delete()

            AuditLog.objects.create(
                user=request.user,
                action='invoice_deleted',
                resource_type='invoice',
                resource_id=invoice.id
            )

            return Response(
                {"message": "Invoice deleted successfully"},
                status=status.HTTP_200_OK
            )

        except Exception as e:
            logger.error(f"Invoice deletion failed: {e}")
            return Response(
                {"error": "Failed to delete invoice"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class InvoiceListView(BaseAPIView):
    """List and filter invoices"""

    @ratelimit(key='user', rate='20/m', method='GET')
    def get(self, request):
        try:
            invoices = Invoice.objects.filter(
                user=request.user,
                is_deleted=False
            ).select_related('client').order_by('-uploaded_at')

            # Apply filters
            status_filter = request.GET.get('status')
            if status_filter:
                invoices = invoices.filter(processing_status=status_filter)

            date_from = request.GET.get('date_from')
            date_to = request.GET.get('date_to')
            if date_from and date_to:
                invoices = invoices.filter(uploaded_at__date__range=[date_from, date_to])

            fraud_risk = request.GET.get('fraud_risk')
            if fraud_risk:
                invoices = invoices.filter(fraud_risk_level=fraud_risk)

            page = int(request.GET.get('page', 1))
            page_size = min(int(request.GET.get('page_size', 20)), 100)

            total_count = invoices.count()
            invoices = invoices[(page-1)*page_size : page*page_size]

            serializer = InvoiceSerializer(invoices, many=True)

            return Response({
                "invoices": serializer.data,
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "total_count": total_count,
                    "total_pages": (total_count + page_size - 1) // page_size
                }
            })

        except Exception as e:
            logger.error(f"Invoice listing failed: {e}")
            return Response(
                {"error": "Failed to retrieve invoices"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class ClientManagementView(BaseAPIView):
    """Client CRUD operations"""

    @ratelimit(key='user', rate='20/m', method='GET')
    def get(self, request):
        try:
            clients = Client.objects.filter(
                user=request.user,
                is_deleted=False,
                is_active=True
            ).order_by('name')

            serializer = ClientSerializer(clients, many=True)
            return Response(serializer.data)

        except Exception as e:
            logger.error(f"Client listing failed: {e}")
            return Response(
                {"error": "Failed to retrieve clients"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @ratelimit(key='user', rate='10/m', method='POST')
    def post(self, request):
        try:
            serializer = ClientSerializer(data=request.data)
            if serializer.is_valid():
                client = serializer.save(user=request.user)

                AuditLog.objects.create(
                    user=request.user,
                    action='client_created',
                    resource_type='client',
                    resource_id=client.id
                )

                return Response(serializer.data, status=status.HTTP_201_CREATED)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        except Exception as e:
            logger.error(f"Client creation failed: {e}")
            return Response(
                {"error": "Failed to create client"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )