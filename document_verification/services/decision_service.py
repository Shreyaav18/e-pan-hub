"""
Decision Service — Document Verification final output
------------------------------------------------------
Combines OCR results + Sdoc into a clean structured output
that gets saved to the VerificationCase model.
"""

import cv2
import os
import uuid
from django.conf import settings
from .ocr_service import run_ocr_validation
from .authenticity_service import compute_sdoc
from core.utils import load_image_from_upload, draw_diff_regions


def run_document_verification(uploaded_file, reference_file=None):
    """
    Master pipeline for Module 1.

    Parameters
    ----------
    uploaded_file  : Django InMemoryUploadedFile (the PAN card)
    reference_file : optional reference PAN image for SSIM comparison

    Returns
    -------
    dict with all fields needed to populate VerificationCase
    """

    # ── Step 1: Load images ─────────────────────────────────────────────────
    img_bgr = load_image_from_upload(uploaded_file)
    if img_bgr is None:
        return {'error': 'Could not decode uploaded image.'}

    ref_gray = None
    if reference_file:
        ref_bgr  = load_image_from_upload(reference_file)
        if ref_bgr is not None:
            ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)

    # Save temp file for EXIF metadata check
    temp_path = _save_temp(uploaded_file)

    # ── Step 2: OCR extraction + validation (Upgrade 1) ────────────────────
    ocr_result = run_ocr_validation(img_bgr)

    # ── Step 3: Multi-feature authenticity scoring (Upgrade 2) ─────────────
    auth_result = compute_sdoc(
        img_bgr,
        ocr_score    = ocr_result['ocr_score'],
        image_path   = temp_path,
        reference_gray = ref_gray,
    )

    # ── Step 4: Annotate image with suspicious regions ──────────────────────
    annotated_path = _save_annotated(img_bgr, auth_result['diff_contours'])

    # ── Step 5: Cleanup temp file ────────────────────────────────────────────
    if temp_path and os.path.exists(temp_path):
        os.remove(temp_path)

    # ── Step 6: Build structured output (Upgrade 5 starts here) ────────────
    sdoc = auth_result['sdoc']
    feature_scores = auth_result['feature_scores']

    return {
        # OCR extracted fields
        'extracted_pan_number': ocr_result['pan_number'],
        'extracted_name':       ocr_result['name'],
        'extracted_dob':        ocr_result['dob'],
        'raw_text':             ocr_result['raw_text'],

        # Individual feature scores (map to DB columns)
        'doc_ocr_valid':        feature_scores['ocr_valid'],
        'doc_font_consistency': feature_scores['font_consistency'],
        'doc_edge_consistency': feature_scores['edge_consistency'],
        'doc_color_histogram':  feature_scores['color_histogram'],
        'doc_structural_align': feature_scores['structural_align'],
        'doc_metadata_clean':   feature_scores['metadata_clean'],
        'doc_ssim_score':       feature_scores['ssim'],

        # Final module score
        'doc_score':            sdoc,

        # For UI rendering
        'annotated_image_path': annotated_path,
        'verdict':              _doc_verdict(sdoc),
        'confidence':           _confidence_label(sdoc),
    }


def _doc_verdict(sdoc):
    """Human-readable document verdict."""
    if sdoc >= 0.75:
        return 'AUTHENTIC'
    elif sdoc >= 0.50:
        return 'SUSPICIOUS'
    else:
        return 'LIKELY FORGED'


def _confidence_label(sdoc):
    if sdoc >= 0.85:
        return 'High'
    elif sdoc >= 0.60:
        return 'Medium'
    else:
        return 'Low'


def _save_temp(uploaded_file):
    """Save uploaded file to a temp path for EXIF reading."""
    try:
        import tempfile
        uploaded_file.seek(0)
        suffix = os.path.splitext(uploaded_file.name)[-1] or '.jpg'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            uploaded_file.seek(0)
            return tmp.name
    except Exception:
        return None


def _save_annotated(img_bgr, contours):
    """Draw bounding boxes on suspicious regions and save output image."""
    try:
        annotated = draw_diff_regions(img_bgr, contours)
        filename  = f"doc_annotated_{uuid.uuid4().hex[:8]}.jpg"
        out_dir   = os.path.join(settings.MEDIA_ROOT, 'outputs')
        os.makedirs(out_dir, exist_ok=True)
        out_path  = os.path.join(out_dir, filename)
        cv2.imwrite(out_path, annotated)
        return f"outputs/{filename}"
    except Exception:
        return None