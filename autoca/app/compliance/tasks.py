import logging
import requests
from celery import shared_task
from django.db import transaction
from django.conf import settings
from invoices.models import Invoice, GSTRFiling, GSTRFilingItem

logger = logging.getLogger(__name__)

@shared_task(
    bind=True,
    max_retries=3,
    time_limit=300,
    retry_backoff=True
)
def file_gstr3b_task(self, filing_id: int):
    """
    Task 2: GSTN API Integration for GSTR-3B Filing
    """
    try:
        filing = GSTRFiling.objects.select_related('user').get(id=filing_id)

        # Prepare payload from invoices
        invoices = filing.invoices.select_related('client').all()
        payload = _prepare_gstr3b_payload(invoices, filing.user)

        # Mock API call (replace with actual GSTN API integration)
        logger.info(f"Preparing to file GSTR-3B for {filing.filing_period}")

        # Simulate API call
        response = _mock_gstn_api_call(payload)

        if response.get('status') == 'success':
            filing.status = 'filed'
            filing.gstn_acknowledgement_ref = response.get('acknowledgement_ref')
            filing.gstn_filing_date = timezone.now()
            filing.save()

            # Update all invoices
            invoices.update(gst_filing_status='filed', filed_at=timezone.now())

            logger.info(f"GSTR-3B filed successfully for {filing.filing_period}")
            return {'status': 'success', 'filing_id': filing_id}
        else:
            raise Exception(f"GSTN API error: {response.get('error')}")

    except Exception as e:
        logger.error(f"GSTR-3B filing failed: {str(e)}")
        filing.status = 'failed'
        filing.error_message = str(e)
        filing.retry_count += 1
        filing.save()
        raise self.retry(exc=e)

@shared_task
def sync_tally_ledger_task(self, user_id: int, transaction_data: list):
    """
    Task 3: Tally ERP 9 Synchronization
    """
    try:
        # Prepare Tally XML payload
        xml_payload = _prepare_tally_xml(transaction_data)

        # Mock Tally API call
        response = _mock_tally_api_call(xml_payload)

        if response.get('status') == 'success':
            logger.info(f"Tally sync completed for user {user_id}")
            return {'status': 'success', 'synced_count': len(transaction_data)}
        else:
            raise Exception(f"Tally sync error: {response.get('error')}")

    except Exception as e:
        logger.error(f"Tally sync failed: {str(e)}")
        raise self.retry(exc=e)

@shared_task
def generate_xbrl_report_task(self, user_id: int, period: str):
    """
    Task 8: XBRL Report Generation
    """
    try:
        # Generate XBRL content
        xbrl_content = _generate_xbrl_content(user_id, period)

        # Store report
        report_data = {
            'content': xbrl_content,
            'generated_at': timezone.now().isoformat(),
            'period': period
        }

        logger.info(f"XBRL report generated for user {user_id}, period {period}")
        return {'status': 'success', 'report': report_data}

    except Exception as e:
        logger.error(f"XBRL report generation failed: {str(e)}")
        raise self.retry(exc=e)

# Helper functions for compliance tasks
def _prepare_gstr3b_payload(invoices, user):
    """Prepare GSTR-3B payload from invoices"""
    payload = {
        "gstin": user.profile.company_gstin,
        "fp": invoices[0].extracted_data.get('date', '')[:7],  # YYYY-MM
        "b2b": [],
        "b2cs": [],
        "cdnr": [],
        "cdnur": []
    }

    for invoice in invoices:
        invoice_data = {
            "ctin": invoice.extracted_data.get('gstin', ''),
            "cfs": "Y" if invoice.fraud_risk_level == 'low' else "N",
            "chksum": _generate_checksum(invoice),
            "inv": [{
                "val": float(invoice.extracted_data.get('amount', 0)),
                "pos": "01",  # Assume intra-state
                "rt": 18.0,   # Assume 18% GST
                "txval": float(invoice.extracted_data.get('amount', 0)) / 1.18,
                "iamt": 0,
                "camt": 0,
                "samt": 0,
                "csamt": 0
            }]
        }
        payload["b2b"].append(invoice_data)

    return payload

def _mock_gstn_api_call(payload):
    """Mock GSTN API call (replace with actual implementation)"""
    # Simulate API delay
    import time
    time.sleep(2)

    # Simulate success response
    return {
        "status": "success",
        "acknowledgement_ref": f"ACK{int(time.time())}",
        "filing_date": timezone.now().isoformat()
    }

def _prepare_tally_xml(transaction_data):
    """Prepare Tally XML payload"""
    # Simplified XML generation
    xml_template = """<?xml version="1.0"?>
    <ENVELOPE>
        <HEADER>
            <TALLYREQUEST>Import Data</TALLYREQUEST>
        </HEADER>
        <BODY>
            <IMPORTDATA>
                <REQUESTDESC>
                    <REPORTNAME>Vouchers</REPORTNAME>
                </REQUESTDESC>
                <REQUESTDATA>
                    {vouchers}
                </REQUESTDATA>
            </IMPORTDATA>
        </BODY>
    </ENVELOPE>"""

    vouchers = ""
    for transaction in transaction_data:
        vouchers += f"""
        <TALLYMESSAGE>
            <VOUCHER>
                <DATE>{transaction['date']}</DATE>
                <NARRATION>{transaction['description']}</NARRATION>
                <AMOUNT>{transaction['amount']}</AMOUNT>
            </VOUCHER>
        </TALLYMESSAGE>"""

    return xml_template.format(vouchers=vouchers)

def _mock_tally_api_call(xml_payload):
    """Mock Tally API call"""
    return {"status": "success", "processed": xml_payload.count('VOUCHER')}

def _generate_xbrl_content(user_id, period):
    """Generate XBRL report content"""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance">
    <xbrli:context>
        <xbrli:entity>
            <xbrli:identifier scheme="http://www.gst.gov.in/">GSTIN</xbrli:identifier>
        </xbrli:entity>
        <xbrli:period>
            <xbrli:startDate>{period}-01</xbrli:startDate>
            <xbrli:endDate>{period}-31</xbrli:endDate>
        </xbrli:period>
    </xbrli:context>
    <in-gaap:TotalAssets contextRef="current">1000000</in-gaap:TotalAssets>
</xbrli:xbrl>"""

def _generate_checksum(invoice):
    """Generate checksum for GSTN validation"""
    import hashlib
    data = f"{invoice.extracted_data.get('gstin','')}{invoice.extracted_data.get('amount',0)}{invoice.extracted_data.get('date','')}"
    return hashlib.md5(data.encode()).hexdigest()
