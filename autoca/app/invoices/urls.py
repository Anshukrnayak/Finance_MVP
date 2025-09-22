from django.urls import path
from .views import (
    InvoiceUploadView,
    BatchInvoiceUploadView,
    InvoiceDetailView,
    InvoiceListView,
    ClientManagementView
)

urlpatterns = [
    path('upload/', InvoiceUploadView.as_view(), name='invoice-upload'),
    path('upload/batch/', BatchInvoiceUploadView.as_view(), name='batch-upload'),
    path('invoices/', InvoiceListView.as_view(), name='invoice-list'),
    path('invoices/<int:invoice_id>/', InvoiceDetailView.as_view(), name='invoice-detail'),
    path('clients/', ClientManagementView.as_view(), name='client-management'),
]

