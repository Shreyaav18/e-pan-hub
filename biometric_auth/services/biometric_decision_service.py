"""
Biometric Decision Service
---------------------------
Orchestrates all three biometric sub-services and computes
the final biometric_score saved to VerificationCase.
"""

import cv2
import os
import uuid
from django.conf import settings
from core.utils import load_image_from_upload
from .face_match_service import run_face_match, extract_pan_photo_region
from .liveness_service import run_liveness_detection
from .behavioral_service import run_behavioral_analysis


# Weights for final biometric score
BIOMETRIC_WEIGHTS = {
    'face_match': 0.50,   # Most critical — is this the same person?
    'liveness':   0.35,   # Anti-spoof
    'behavioral': 0.15,   # Supporting signal
}


def run_biometric_verification(pan_file, selfie_file,
                                keystroke_data=None,
                                mouse_data=None,
                                session_data=None):
    """
    Master biometric pipeline.

    Parameters
    ----------
    pan_file       : Django InMemoryUploadedFile (PAN card)
    selfie_file    : Django InMemoryUploadedFile (live selfie)
    keystroke_data : list (from JS) or None
    mouse_data     : list (from JS) or None
    session_data   : dict (from JS) or None

    Returns
    -------
    dict with all fields for VerificationCase + display context
    """

    # ── Load images ──────────────────────────────────────────────────────────
    pan_bgr    = load_image_from_upload(pan_file)
    selfie_bgr = load_image_from_upload(selfie_file)

    if pan_bgr is None:
        return {'error': 'Could not decode PAN card image.'}
    if selfie_bgr is None:
        return {'error': 'Could not decode selfie image.'}

    # ── Face Matching (Upgrade 3) ─────────────────────────────────────────────
    face_result = run_face_match(pan_bgr, selfie_bgr)

    # ── Liveness Detection (Upgrade 4) ───────────────────────────────────────
    # Run liveness on the selfie face crop
    selfie_face, _ = _get_selfie_face(selfie_bgr)
    liveness_result = run_liveness_detection(selfie_face)

    # ── Behavioral Biometrics ────────────────────────────────────────────────
    behavioral_result = run_behavioral_analysis(
        keystroke_data=keystroke_data,
        mouse_data=mouse_data,
        session_data=session_data,
    )

    # ── Weighted biometric score ─────────────────────────────────────────────
    face_score = face_result['face_match_score']
    live_score = liveness_result['liveness_score']
    beh_score  = behavioral_result['behavioral_score']

    biometric_score = (
        BIOMETRIC_WEIGHTS['face_match'] * face_score +
        BIOMETRIC_WEIGHTS['liveness']   * live_score +
        BIOMETRIC_WEIGHTS['behavioral'] * beh_score
    )
    biometric_score = round(float(min(1.0, max(0.0, biometric_score))), 4)

    # ── Annotate selfie with face bbox ───────────────────────────────────────
    annotated_path = _save_annotated_selfie(
        selfie_bgr, face_result.get('pan_face_bbox')
    )

    # ── Penalty flag: liveness failed = hard block ───────────────────────────
    penalty_flags = []
    if not liveness_result['is_live']:
        penalty_flags.append('LIVENESS_FAILED')
    if face_result['match_verdict'] == 'NO MATCH':
        penalty_flags.append('FACE_MISMATCH')

    return {
        # DB fields
        'face_match_score':   face_score,
        'liveness_score':     live_score,
        'behavioral_score':   beh_score,
        'biometric_score':    biometric_score,
        'penalty_flags':      penalty_flags,

        # Display context
        'face_verdict':        face_result['match_verdict'],
        'liveness_verdict':    liveness_result['liveness_verdict'],
        'behavioral_verdict':  behavioral_result['behavioral_verdict'],
        'embedding_method':    face_result['embedding_method'],
        'liveness_methods':    liveness_result['method_scores'],
        'annotated_path':      annotated_path,
        'pan_face_detected':   face_result['pan_face_detected'],
        'selfie_face_detected':face_result['selfie_face_detected'],
    }


def _get_selfie_face(selfie_bgr):
    """Extract face from selfie for liveness check."""
    from .face_match_service import detect_face
    face, bbox = detect_face(selfie_bgr)
    if face is None:
        return selfie_bgr, None
    return face, bbox


def _save_annotated_selfie(selfie_bgr, bbox):
    """Draw face bounding box on selfie and save."""
    try:
        annotated = selfie_bgr.copy()
        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(annotated, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)
            cv2.putText(annotated, 'FACE DETECTED', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        filename = f"bio_annotated_{uuid.uuid4().hex[:8]}.jpg"
        out_dir  = os.path.join(settings.MEDIA_ROOT, 'outputs')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, filename)
        cv2.imwrite(out_path, annotated)
        return f"outputs/{filename}"
    except Exception:
        return None