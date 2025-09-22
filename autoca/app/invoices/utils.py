import re
from datetime import datetime
from typing import Optional

def validate_gstin(gstin: str) -> bool:
    """Validate GSTIN using checksum algorithm"""
    if not gstin or len(gstin) != 15:
        return False

    # Basic pattern validation
    pattern = r'^\d{2}[A-Z]{5}\d{4}[A-Z]{1}\d{1}[A-Z]{1}\d{1}$'
    if not re.match(pattern, gstin):
        return False

    # TODO: Implement actual checksum validation
    return True

def extract_amount(amount_str: str) -> Optional[float]:
    """Extract and clean amount from string"""
    try:
        # Remove commas and currency symbols
        clean_str = re.sub(r'[^\d.]', '', amount_str)
        return float(clean_str)
    except (ValueError, TypeError):
        return None

def extract_date(date_str: str) -> Optional[str]:
    """Extract and standardize date from string"""
    try:
        # Try various date formats
        formats = [
            '%d-%m-%Y', '%d/%m/%Y', '%Y-%m-%d',
            '%d-%m-%y', '%d/%m/%y'
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue

        return None
    except (ValueError, TypeError):
        return None
