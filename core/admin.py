from django.contrib import admin
from .models import VerificationCase


@admin.register(VerificationCase)
class VerificationCaseAdmin(admin.ModelAdmin):
    list_display  = ['case_id', 'full_name', 'status', 'risk_total',
                     'doc_score', 'biometric_score', 'aml_score', 'created_at']
    list_filter   = ['status']
    search_fields = ['full_name', 'extracted_pan_number', 'case_id']
    readonly_fields = ['case_id', 'created_at', 'updated_at']

    fieldsets = (
        ('Identity', {
            'fields': ('case_id', 'full_name', 'pan_card_image', 'selfie_image', 'created_at', 'updated_at')
        }),
        ('Module 1 — Document Verification', {
            'fields': ('doc_ocr_valid', 'doc_font_consistency', 'doc_edge_consistency',
                       'doc_color_histogram', 'doc_structural_align', 'doc_metadata_clean',
                       'doc_ssim_score', 'doc_score',
                       'extracted_pan_number', 'extracted_name', 'extracted_dob')
        }),
        ('Module 2 — Biometric Authentication', {
            'fields': ('face_match_score', 'liveness_score', 'behavioral_score', 'biometric_score')
        }),
        ('Module 3 — AML Monitoring', {
            'fields': ('rule_flags', 'isolation_forest_score', 'lstm_anomaly_score', 'aml_score')
        }),
        ('Module 4 — Risk Engine', {
            'fields': ('risk_total', 'penalty_flags', 'status', 'decision_reason')
        }),
    )