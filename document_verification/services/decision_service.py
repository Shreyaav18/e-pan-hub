import cv2
import os
import uuid
from django.conf import settings
from .ocr_service import run_ocr_validation
from .authenticity_service import compute_sdoc
from .field_detector import detect_fields
from core.utils import load_image_from_upload, draw_diff_regions


def run_document_verification(uploaded_file, reference_file=None):
    img_bgr = load_image_from_upload(uploaded_file)
    if img_bgr is None:
        return {'error': 'Could not decode uploaded image.'}

    ref_gray = None
    if reference_file:
        ref_bgr = load_image_from_upload(reference_file)
        if ref_bgr is not None:
            ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)

    temp_path = _save_temp(uploaded_file)

    fields = detect_fields(img_bgr)

    ocr_result = run_ocr_validation(img_bgr, fields=fields)

    auth_result = compute_sdoc(
        img_bgr,
        ocr_score      = ocr_result['ocr_score'],
        image_path     = temp_path,
        reference_gray = ref_gray,
        raw_ocr_text   = ocr_result['raw_text'],
    )

    annotated_path = _save_annotated(img_bgr, auth_result['diff_contours'])

    if temp_path and os.path.exists(temp_path):
        os.remove(temp_path)

    sdoc           = auth_result['sdoc']
    feature_scores = auth_result['feature_scores']
    security_scores = auth_result['security_feature_scores']

    return {
        'extracted_pan_number': ocr_result['pan_number'],
        'extracted_name':       ocr_result['name'],
        'extracted_dob':        ocr_result['dob'],
        'extracted_father':     ocr_result['father'],
        'raw_text':             ocr_result['raw_text'],

        'doc_ocr_valid':          feature_scores['ocr_valid'],
        'doc_security_features':  feature_scores['security_features'],
        'doc_font_consistency':   feature_scores['font_consistency'],
        'doc_edge_consistency':   feature_scores['edge_consistency'],
        'doc_color_histogram':    feature_scores['color_histogram'],
        'doc_structural_align':   feature_scores['structural_align'],
        'doc_metadata_clean':     feature_scores['metadata_clean'],
        'doc_ssim_score':         feature_scores['ssim'],

        'sec_itd_emblem':         security_scores['itd_emblem'],
        'sec_qr_code':            security_scores['qr_code'],
        'sec_face_present':       security_scores['face_present'],
        'sec_signature':          security_scores['signature'],
        'sec_govt_text':          security_scores['govt_text'],
        'sec_background_pattern': security_scores['background_pattern'],

        'doc_score':            sdoc,
        'annotated_image_path': annotated_path,
        'verdict':              _doc_verdict(sdoc),
        'confidence':           _confidence_label(sdoc),

        'field_detections': {
            k: {'bbox': v['bbox'], 'conf': round(v['conf'], 4)}
            for k, v in fields.items()
        },
    }


def _doc_verdict(sdoc):
    if sdoc >= 0.75:
        return 'AUTHENTIC'
    elif sdoc >= 0.50:
        return 'SUSPICIOUS'
    return 'LIKELY FORGED'


def _confidence_label(sdoc):
    if sdoc >= 0.85:
        return 'High'
    elif sdoc >= 0.60:
        return 'Medium'
    return 'Low'


def _save_temp(uploaded_file):
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