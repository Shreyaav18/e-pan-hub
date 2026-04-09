from django.db import models
import uuid


class VerificationCase(models.Model):
    STATUS_CHOICES = [
        ('pending',    'Pending'),
        ('processing', 'Processing'),
        ('approved',   'Auto Approved'),
        ('review',     'Manual Review'),
        ('rejected',   'Auto Rejected'),
    ]

    case_id        = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    full_name      = models.CharField(max_length=255, blank=True)
    created_at     = models.DateTimeField(auto_now_add=True)
    updated_at     = models.DateTimeField(auto_now=True)

    # Uploaded files
    pan_card_image = models.ImageField(upload_to='uploads/pan_cards/', null=True, blank=True)
    selfie_image   = models.ImageField(upload_to='uploads/selfies/',   null=True, blank=True)

    # Module 1 - Document Verification
    doc_ocr_valid        = models.FloatField(null=True, blank=True)
    doc_font_consistency = models.FloatField(null=True, blank=True)
    doc_edge_consistency = models.FloatField(null=True, blank=True)
    doc_color_histogram  = models.FloatField(null=True, blank=True)
    doc_structural_align = models.FloatField(null=True, blank=True)
    doc_metadata_clean   = models.FloatField(null=True, blank=True)
    doc_ssim_score       = models.FloatField(null=True, blank=True)
    doc_score            = models.FloatField(null=True, blank=True)
    extracted_pan_number = models.CharField(max_length=20,  blank=True)
    extracted_name       = models.CharField(max_length=255, blank=True)
    extracted_dob        = models.CharField(max_length=20,  blank=True)

    # Module 2 - Biometric Authentication
    face_match_score   = models.FloatField(null=True, blank=True)
    liveness_score     = models.FloatField(null=True, blank=True)
    behavioral_score   = models.FloatField(null=True, blank=True)
    biometric_score    = models.FloatField(null=True, blank=True)

    # Module 3 - AML Transaction Monitoring
    rule_flags             = models.JSONField(default=list)
    isolation_forest_score = models.FloatField(null=True, blank=True)
    lstm_anomaly_score     = models.FloatField(null=True, blank=True)
    aml_score              = models.FloatField(null=True, blank=True)

    # Module 4 - Risk Scoring Engine
    risk_total      = models.FloatField(null=True, blank=True)
    penalty_flags   = models.JSONField(default=list)
    status          = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    decision_reason = models.TextField(blank=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Verification Case'

    def __str__(self):
        return f"Case {self.case_id} | {self.full_name or 'Unknown'} | {self.status}"

    def get_all_scores(self):
        return {
            'document':   self.doc_score,
            'liveness':   self.liveness_score,
            'behavioral': self.behavioral_score,
            'video_kyc':  None,
            'aml':        self.aml_score,
        }
