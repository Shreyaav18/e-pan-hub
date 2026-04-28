from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from core.models import VerificationCase
from .services.decision_service import run_document_verification
from django.views.decorators.csrf import csrf_exempt

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

@csrf_exempt
def api_verify_view(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)

    pan_file  = request.FILES.get('pan_card')
    ref_file  = request.FILES.get('reference_card')
    full_name = request.POST.get('full_name', '').strip()

    if not pan_file:
        return JsonResponse({'error': 'No PAN card uploaded'}, status=400)

    result = run_document_verification(pan_file, ref_file)

    if 'error' in result:
        return JsonResponse(result, status=400)

    result.pop('diff_contours', None)
    result['case_id'] = f"CASE-{result.get('extracted_pan_number', 'UNKNOWN')}"
    result['status']  = 'approved' if result['doc_score'] >= 0.75 else ('review' if result['doc_score'] >= 0.50 else 'rejected')

    return JsonResponse(result)

@csrf_exempt
def upload_view(request):
    if request.method == 'POST':
        pan_file  = request.FILES.get('pan_card')
        ref_file  = request.FILES.get('reference_card')
        full_name = request.POST.get('full_name', '').strip()

        if not pan_file:
            messages.error(request, 'Please upload a PAN card image.')
            return redirect('document_verification:upload')

        result = run_document_verification(pan_file, ref_file)

        if 'error' in result:
            messages.error(request, result['error'])
            return redirect('document_verification:upload')

        pan_file.seek(0)
        case = VerificationCase(full_name=full_name or result['extracted_name'])
        case.pan_card_image = pan_file

        case.extracted_pan_number = result['extracted_pan_number']
        case.extracted_name       = result['extracted_name']
        case.extracted_dob        = result['extracted_dob']
        case.extracted_father     = result['extracted_father']

        case.doc_ocr_valid          = result['doc_ocr_valid']
        case.doc_security_features  = result['doc_security_features']
        case.doc_font_consistency   = result['doc_font_consistency']
        case.doc_edge_consistency   = result['doc_edge_consistency']
        case.doc_color_histogram    = result['doc_color_histogram']
        case.doc_structural_align   = result['doc_structural_align']
        case.doc_metadata_clean     = result['doc_metadata_clean']
        case.doc_ssim_score         = result['doc_ssim_score']

        case.sec_itd_emblem         = result['sec_itd_emblem']
        case.sec_qr_code            = result['sec_qr_code']
        case.sec_face_present       = result['sec_face_present']
        case.sec_signature          = result['sec_signature']
        case.sec_govt_text          = result['sec_govt_text']
        case.sec_background_pattern = result['sec_background_pattern']

        case.doc_score = result['doc_score']
        case.status    = 'processing'
        case.save()

        return redirect('document_verification:result', pk=case.pk)

    return render(request, 'document_verification/upload.html')


def result_view(request, pk):
    case = get_object_or_404(VerificationCase, pk=pk)

    feature_scores = {
        'OCR Validation':     case.doc_ocr_valid,
        'Security Features':  case.doc_security_features,
        'Font Consistency':   case.doc_font_consistency,
        'Edge Consistency':   case.doc_edge_consistency,
        'Color Histogram':    case.doc_color_histogram,
        'Structural Align':   case.doc_structural_align,
        'Metadata Clean':     case.doc_metadata_clean,
        'SSIM Score':         case.doc_ssim_score,
    }

    feature_scores_display = {
        label: (score * 100 if score is not None else None)
        for label, score in feature_scores.items()
    }

    security_scores = {
        'ITD Emblem':         case.sec_itd_emblem,
        'QR Code':            case.sec_qr_code,
        'Face Present':       case.sec_face_present,
        'Signature':          case.sec_signature,
        'Govt Text':          case.sec_govt_text,
        'Background Pattern': case.sec_background_pattern,
    }

    security_scores_display = {
        label: (score * 100 if score is not None else None)
        for label, score in security_scores.items()
    }

    verdict = _doc_verdict(case.doc_score)
    context = {
        'case':                  case,
        'feature_scores':        feature_scores_display,
        'security_scores':       security_scores_display,
        'verdict':               verdict,
        'sdoc_percent':          round((case.doc_score or 0) * 100, 1),
    }
    return render(request, 'document_verification/result.html', context)


def _doc_verdict(sdoc):
    if not sdoc:
        return ('UNKNOWN', 'secondary')
    if sdoc >= 0.75:
        return ('AUTHENTIC', 'success')
    elif sdoc >= 0.50:
        return ('SUSPICIOUS', 'warning')
    return ('LIKELY FORGED', 'danger')