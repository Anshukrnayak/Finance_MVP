import json
from rest_framework import serializers
from django.core.validators import MinValueValidator
from django.utils import timezone
from .models import Invoice, InvoiceBatch, Client, GSTRFiling, GSTRFilingItem
from core.models import BaseUser

class ClientSerializer(serializers.ModelSerializer):
    """Client serializer with validation"""
    recent_invoice_count = serializers.SerializerMethodField()
    total_business_volume = serializers.SerializerMethodField()

    class Meta:
        model = Client
        fields = (
            'id', 'client_id', 'name', 'business_name', 'email', 'phone',
            'gstin', 'pan', 'business_type', 'billing_address', 'shipping_address',
            'tags', 'notes', 'is_active', 'recent_invoice_count', 'total_business_volume',
            'created_at', 'updated_at'
        )
        read_only_fields = ('id', 'client_id', 'created_at', 'updated_at')

    def get_recent_invoice_count(self, obj):
        return obj.invoices.filter(is_deleted=False).count()

    def get_total_business_volume(self, obj):
        from django.db.models import Sum
        result = obj.invoices.filter(
            is_deleted=False,
            processing_status='processed',
            extracted_data__has_key='amount'
        ).aggregate(total=Sum('extracted_data__amount'))
        return result['total'] or 0

    def validate_gstin(self, value):
        if value and len(value) != 15:
            raise serializers.ValidationError("GSTIN must be 15 characters long")
        return value

    def validate_pan(self, value):
        if value and len(value) != 10:
            raise serializers.ValidationError("PAN must be 10 characters long")
        return value

class ExtractedDataField(serializers.DictField):
    """Custom field for extracted invoice data"""
    def to_representation(self, value):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return {}
        return value or {}

    def to_internal_value(self, data):
        if isinstance(data, str):
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                raise serializers.ValidationError("Invalid JSON format")
        return data or {}

class ConfidenceScoresField(serializers.DictField):
    """Custom field for confidence scores"""
    def to_representation(self, value):
        if isinstance(value, str):
            try:
                scores = json.loads(value)
                # Convert numeric scores to percentages for display
                return {k: f"{v*100:.1f}%" if isinstance(v, (int, float)) else v
                        for k, v in scores.items()}
            except json.JSONDecodeError:
                return {}
        return value or {}

class MLResultsField(serializers.DictField):
    """Custom field for ML analysis results"""
    def to_representation(self, value):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return {}
        return value or {}

class InvoiceSerializer(serializers.ModelSerializer):
    """Invoice serializer with ML integration"""
    extracted_data = ExtractedDataField(required=False)
    confidence_scores = ConfidenceScoresField(required=False, read_only=True)
    advanced_analytics = MLResultsField(required=False, read_only=True)
    client_name = serializers.CharField(source='client.name', read_only=True)
    client_gstin = serializers.CharField(source='client.gstin', read_only=True)
    processing_duration = serializers.SerializerMethodField()
    fraud_risk_display = serializers.SerializerMethodField()

    # ML Analysis flags
    requires_ml_analysis = serializers.SerializerMethodField()
    ml_analysis_status = serializers.SerializerMethodField()

    class Meta:
        model = Invoice
        fields = (
            'id', 'file_id', 'client', 'client_name', 'client_gstin',
            'original_filename', 'file_type', 'file_size',
            'processing_status', 'processing_attempts', 'last_processing_error',
            'extracted_data', 'confidence_scores', 'is_gstin_valid',
            'is_amount_valid', 'is_date_valid', 'fraud_score', 'fraud_risk_level',
            'fraud_risk_display', 'advanced_analytics', 'gst_filing_status',
            'gstr_transaction_id', 'uploaded_at', 'processed_at', 'verified_at',
            'filed_at', 'processing_duration', 'requires_ml_analysis',
            'ml_analysis_status'
        )
        read_only_fields = (
            'id', 'file_id', 'file_size', 'processing_status', 'processing_attempts',
            'last_processing_error', 'confidence_scores', 'fraud_score',
            'fraud_risk_level', 'advanced_analytics', 'uploaded_at', 'processed_at',
            'verified_at', 'filed_at'
        )

    def get_processing_duration(self, obj):
        if obj.processed_at and obj.uploaded_at:
            duration = obj.processed_at - obj.uploaded_at
            return str(duration)
        return None

    def get_fraud_risk_display(self, obj):
        risk_levels = {
            'low': 'Low Risk',
            'medium': 'Medium Risk',
            'high': 'High Risk',
            'critical': 'Critical Risk'
        }
        return risk_levels.get(obj.fraud_risk_level, 'Unknown')

    def get_requires_ml_analysis(self, obj):
        return (obj.processing_status == 'processed' and
                obj.extracted_data and
                'amount' in obj.extracted_data)

    def get_ml_analysis_status(self, obj):
        if obj.fraud_score is not None:
            return 'completed'
        elif self.get_requires_ml_analysis(obj):
            return 'pending'
        return 'not_required'

    def validate(self, attrs):
        # Validate that client belongs to the same user
        client = attrs.get('client')
        if client and client.user != self.context['request'].user:
            raise serializers.ValidationError({"client": "Invalid client"})

        return attrs

class InvoiceBatchSerializer(serializers.ModelSerializer):
    """Invoice batch serializer"""
    success_rate = serializers.SerializerMethodField()
    processing_time = serializers.SerializerMethodField()
    file_types = serializers.SerializerMethodField()

    class Meta:
        model = InvoiceBatch
        fields = (
            'id', 'batch_id', 'name', 'description', 'status',
            'total_invoices', 'successful_invoices', 'failed_invoices',
            'success_rate', 'processing_time', 'file_types', 'metadata',
            'created_at', 'updated_at'
        )
        read_only_fields = ('id', 'batch_id', 'created_at', 'updated_at')

    def get_success_rate(self, obj):
        if obj.total_invoices > 0:
            return (obj.successful_invoices / obj.total_invoices) * 100
        return 0

    def get_processing_time(self, obj):
        if obj.updated_at and obj.created_at:
            return str(obj.updated_at - obj.created_at)
        return None

    def get_file_types(self, obj):
        from django.db.models import Count
        file_types = Invoice.objects.filter(
            batch=obj,
            is_deleted=False
        ).values('file_type').annotate(count=Count('id'))
        return {item['file_type']: item['count'] for item in file_types}

class GSTRFilingItemSerializer(serializers.ModelSerializer):
    """GSTR filing item serializer"""
    invoice_data = serializers.SerializerMethodField()

    class Meta:
        model = GSTRFilingItem
        fields = (
            'id', 'invoice', 'invoice_data', 'taxable_value', 'tax_amount',
            'created_at'
        )
        read_only_fields = ('id', 'created_at')

    def get_invoice_data(self, obj):
        return {
            'file_id': obj.invoice.file_id,
            'original_filename': obj.invoice.original_filename,
            'amount': obj.invoice.extracted_data.get('amount'),
            'date': obj.invoice.extracted_data.get('date'),
            'gstin': obj.invoice.extracted_data.get('gstin')
        }

class GSTRFilingSerializer(serializers.ModelSerializer):
    """GSTR filing serializer with items"""
    items = GSTRFilingItemSerializer(many=True, read_only=True)
    status_display = serializers.SerializerMethodField()
    filing_type_display = serializers.SerializerMethodField()
    can_file = serializers.SerializerMethodField()

    class Meta:
        model = GSTRFiling
        fields = (
            'id', 'filing_period', 'filing_type', 'filing_type_display',
            'status', 'status_display', 'gstn_acknowledgement_ref',
            'gstn_filing_date', 'total_taxable_value', 'total_tax_amount',
            'total_liability', 'error_message', 'retry_count', 'items',
            'can_file', 'created_at', 'updated_at'
        )
        read_only_fields = ('id', 'created_at', 'updated_at')

    def get_status_display(self, obj):
        status_map = {
            'draft': 'Draft',
            'ready': 'Ready to File',
            'processing': 'Processing',
            'filed': 'Filed Successfully',
            'acknowledged': 'Acknowledged by GSTN',
            'rejected': 'Rejected by GSTN',
            'failed': 'Filing Failed'
        }
        return status_map.get(obj.status, obj.status)

    def get_filing_type_display(self, obj):
        filing_map = {
            'gstr1': 'GSTR-1',
            'gstr3b': 'GSTR-3B',
            'gstr9': 'GSTR-9'
        }
        return filing_map.get(obj.filing_type, obj.filing_type)

    def get_can_file(self, obj):
        return obj.status == 'ready' and obj.items.count() > 0

    def validate_filing_period(self, value):
        # Validate filing period format (YYYY-MM)
        try:
            year, month = map(int, value.split('-'))
            if not (2020 <= year <= 2030 and 1 <= month <= 12):
                raise ValueError
        except (ValueError, AttributeError):
            raise serializers.ValidationError("Filing period must be in YYYY-MM format")
        return value

    