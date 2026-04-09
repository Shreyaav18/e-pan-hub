from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from core.models import VerificationCase
from .services.decision_service import run_document_verification


def upload_view(request):
    """
    GET  — show upload form
    POST — run Module 1 pipeline and save results to VerificationCase
    """
    if request.method == 'POST':
        pan_file  = request.FILES.get('pan_card')
        ref_file  = request.FILES.get('reference_card')  # optional
        full_name = request.POST.get('full_name', '').strip()

        if not pan_file:
            messages.error(request, 'Please upload a PAN card image.')
            return redirect('document_verification:upload')

        # Run the full Module 1 pipeline
        result = run_document_verification(pan_file, ref_file)

        if 'error' in result:
            messages.error(request, result['error'])
            return redirect('document_verification:upload')

        # Save or update VerificationCase
        pan_file.seek(0)
        case = VerificationCase(full_name=full_name or result['extracted_name'])
        case.pan_card_image = pan_file

        # Write all Module 1 scores to the DB
        case.extracted_pan_number = result['extracted_pan_number']
        case.extracted_name       = result['extracted_name']
        case.extracted_dob        = result['extracted_dob']
        case.doc_ocr_valid        = result['doc_ocr_valid']
        case.doc_font_consistency = result['doc_font_consistency']
        case.doc_edge_consistency = result['doc_edge_consistency']
        case.doc_color_histogram  = result['doc_color_histogram']
        case.doc_structural_align = result['doc_structural_align']
        case.doc_metadata_clean   = result['doc_metadata_clean']
        case.doc_ssim_score       = result['doc_ssim_score']
        case.doc_score            = result['doc_score']
        case.status               = 'processing'
        case.save()

        return redirect('document_verification:result', pk=case.pk)

    return render(request, 'document_verification/upload.html')


def result_view(request, pk):
    """Show Module 1 results for a given VerificationCase."""
    case = get_object_or_404(VerificationCase, pk=pk)

    feature_scores = {
        'OCR Validation':      case.doc_ocr_valid,
        'Font Consistency':    case.doc_font_consistency,
        'Edge Consistency':    case.doc_edge_consistency,
        'Color Histogram':     case.doc_color_histogram,
        'Structural Alignment':case.doc_structural_align,
        'Metadata Clean':      case.doc_metadata_clean,
        'SSIM Score':          case.doc_ssim_score,
    }

    verdict = _doc_verdict(case.doc_score)
    context = {
        'case':           case,
        'feature_scores': feature_scores,
        'verdict':        verdict,
        'sdoc_percent':   round((case.doc_score or 0) * 100, 1),
    }
    return render(request, 'document_verification/result.html', context)


def _doc_verdict(sdoc):
    if not sdoc:
        return ('UNKNOWN', 'secondary')
    if sdoc >= 0.75:
        return ('AUTHENTIC', 'success')
    elif sdoc >= 0.50:
        return ('SUSPICIOUS', 'warning')
    else:
        return ('LIKELY FORGED', 'danger')