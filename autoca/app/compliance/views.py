import logging
from datetime import datetime
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from ratelimit.decorators import ratelimit

from core.views import BaseAPIView
from invoices.models import Invoice
from compliance.models import GSTRFiling, GSTRFilingItem
from compliance.tasks import file_gstr3b_task, generate_xbrl_report_task
from compliance.serializers import GSTRFilingSerializer

logger = logging.getLogger(__name__)

class GSTRFilingView(BaseAPIView):
    """GSTR filing management"""

    @ratelimit(key='user', rate='10/m', method='GET')
    def get(self, request, filing_period=None):
        try:
            if filing_period:
                filing = get_object_or_404(
                    GSTRFiling,
                    user=request.user,
                    filing_period=filing_period
                )
                serializer = GSTRFilingSerializer(filing)
                return Response(serializer.data)
            else:
                filings = GSTRFiling.objects.filter(user=request.user).order_by('-filing_period')
                serializer = GSTRFilingSerializer(filings, many=True)
                return Response(serializer.data)

        except Exception as e:
            logger.error(f"GSTR filing retrieval failed: {e}")
            return Response(
                {"error": "Failed to retrieve filing data"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @ratelimit(key='user', rate='5/m', method='POST')
    def post(self, request):
        try:
            filing_period = request.data.get('filing_period')
            filing_type = request.data.get('filing_type', 'gstr3b')

            if not filing_period:
                return Response(
                    {"error": "Filing period required"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Get invoices for the period
            invoices = Invoice.objects.filter(
                user=request.user,
                processing_status='processed',
                extracted_data__has_key='date',
                gst_filing_status='not_filed',
                is_deleted=False
            ).extra(
                where=[f"EXTRACT(YEAR_MONTH FROM uploaded_at) = {filing_period.replace('-', '')}"]
            )

            if not invoices:
                return Response(
                    {"error": "No invoices found for the specified period"},
                    status=status.HTTP_404_NOT_FOUND
                )

            # Create filing record
            filing = GSTRFiling.objects.create(
                user=request.user,
                filing_period=filing_period,
                filing_type=filing_type,
                status='draft'
            )

            # Add invoices to filing
            for invoice in invoices:
                taxable_value = float(invoice.extracted_data.get('amount', 0)) / 1.18
                tax_amount = float(invoice.extracted_data.get('amount', 0)) - taxable_value

                GSTRFilingItem.objects.create(
                    filing=filing,
                    invoice=invoice,
                    taxable_value=taxable_value,
                    tax_amount=tax_amount
                )

            # Update totals
            filing.total_taxable_value = filing.gstrfilingitem_set.aggregate(Sum('taxable_value'))['taxable_value__sum'] or 0
            filing.total_tax_amount = filing.gstrfilingitem_set.aggregate(Sum('tax_amount'))['tax_amount__sum'] or 0
            filing.save()

            serializer = GSTRFilingSerializer(filing)
            return Response(serializer.data, status=status.HTTP_201_CREATED)

        except Exception as e:
            logger.error(f"GSTR filing creation failed: {e}")
            return Response(
                {"error": "Failed to create filing"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class GSTRFilingActionView(BaseAPIView):
    """Actions on GSTR filings"""

    @ratelimit(key='user', rate='3/m', method='POST')
    def post(self, request, filing_id, action):
        try:
            filing = get_object_or_404(GSTRFiling, id=filing_id, user=request.user)

            if action == 'file':
                if filing.status != 'ready':
                    return Response(
                        {"error": "Filing must be in ready status"},
                        status=status.HTTP_400_BAD_REQUEST
                    )

                # Trigger async filing task
                file_gstr3b_task.delay(filing.id)

                filing.status = 'processing'
                filing.save()

                return Response({
                    "message": "GSTR filing started",
                    "filing_id": filing.id
                }, status=status.HTTP_202_ACCEPTED)

            elif action == 'approve':
                filing.status = 'ready'
                filing.save()

                return Response({
                    "message": "Filing approved and ready for submission",
                    "filing_id": filing.id
                })

            else:
                return Response(
                    {"error": "Invalid action"},
                    status=status.HTTP_400_BAD_REQUEST
                )

        except Exception as e:
            logger.error(f"GSTR filing action failed: {e}")
            return Response(
                {"error": "Failed to process filing action"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class XBRLReportView(BaseAPIView):
    """XBRL report generation"""

    @ratelimit(key='user', rate='2/m', method='POST')
    def post(self, request):
        try:
            period = request.data.get('period')
            if not period:
                return Response(
                    {"error": "Reporting period required"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Trigger async report generation
            generate_xbrl_report_task.delay(request.user.id, period)

            return Response({
                "message": "XBRL report generation started",
                "period": period
            }, status=status.HTTP_202_ACCEPTED)

        except Exception as e:
            logger.error(f"XBRL report generation failed: {e}")
            return Response(
                {"error": "Failed to start report generation"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )