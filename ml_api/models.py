"""
Models for SMS Phishing Detection System
"""
import uuid
from django.db import models


class DetectionLog(models.Model):
    """Stores every SMS scan result."""
    LABEL_CHOICES = [('spam', 'Spam'), ('legitimate', 'Legitimate')]
    RISK_CHOICES  = [('Low', 'Low'), ('Medium', 'Medium'), ('High', 'High')]
    MODE_CHOICES  = [('ml', 'ML Model'), ('rule_based', 'Rule Based')]

    detection_id = models.CharField(max_length=64, unique=True, default=uuid.uuid4)
    message_hash = models.CharField(max_length=64, blank=True, help_text="SHA-256 of original message for dedup")
    label        = models.CharField(max_length=12, choices=LABEL_CHOICES)
    confidence   = models.FloatField()
    risk_level   = models.CharField(max_length=8, choices=RISK_CHOICES)
    language     = models.CharField(max_length=5, default='en')
    mode         = models.CharField(max_length=12, choices=MODE_CHOICES, default='ml')
    indicators   = models.JSONField(default=list, blank=True)
    spam_probability  = models.FloatField(default=0)
    legit_probability = models.FloatField(default=0)
    timestamp    = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.detection_id} — {self.label} ({self.confidence}%)"


class ReportedNumber(models.Model):
    """Community-reported scam phone numbers."""
    number          = models.CharField(max_length=20, db_index=True)
    report_count    = models.PositiveIntegerField(default=1)
    language        = models.CharField(max_length=5, default='en')
    predicted_label = models.CharField(max_length=12, blank=True, default='')
    sample_message  = models.TextField(blank=True, default='')
    first_reported  = models.DateTimeField(auto_now_add=True)
    last_reported   = models.DateTimeField(auto_now=True)
    auto_flagged    = models.BooleanField(default=False)
    flagged_at      = models.DateTimeField(null=True, blank=True)

    AUTO_FLAG_THRESHOLD = 20

    class Meta:
        ordering = ['-report_count']

    def __str__(self):
        return f"{self.number} ({self.report_count} reports)"


class Feedback(models.Model):
    """User feedback when the ML model gets it wrong."""
    detection_id    = models.CharField(max_length=64)
    original_label  = models.CharField(max_length=12)
    corrected_label = models.CharField(max_length=12)
    language        = models.CharField(max_length=5, default='en')
    timestamp       = models.DateTimeField(auto_now_add=True)
    processed       = models.BooleanField(default=False)

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.detection_id}: {self.original_label} → {self.corrected_label}"
# In your models.py add:

class CachedAudio(models.Model):
    cache_key = models.CharField(max_length=64, unique=True, db_index=True)
    audio_data = models.BinaryField()
    language = models.CharField(max_length=10)
    voice = models.CharField(max_length=50)
    tts_text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['cache_key']),
            models.Index(fields=['created_at']),
        ]