"""
OCR Service — Upgrade 1: Content-Aware Verification
----------------------------------------------------
Replaces pure SSIM comparison with actual content understanding.
Extracts text fields from the PAN card and validates them
structurally — no reference image needed.

Fields targeted on a standard Indian PAN card:
  - PAN Number  : 5 letters + 4 digits + 1 letter  e.g. ABCDE1234F
  - Full Name   : All-caps text line
  - Date of Birth: DD/MM/YYYY or DD-MM-YYYY
  - Father Name : All-caps text line
"""

import re
import cv2
import numpy as np

# --------------------------------------------------------------------------- #
#  PAN field regex patterns
# --------------------------------------------------------------------------- #
PAN_REGEX   = re.compile(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b')
DOB_REGEX   = re.compile(r'\b(\d{2}[\/\-]\d{2}[\/\-]\d{4})\b')
NAME_REGEX  = re.compile(r'\b([A-Z][A-Z\s]{3,40})\b')   # Consecutive caps words


def _preprocess_for_ocr(img):
    """
    Aggressive preprocessing to maximise OCR accuracy.
    Returns a cleaned grayscale image.
    """
    # Upscale small images — OCR accuracy drops on low-res
    h, w = img.shape[:2]
    if w < 800:
        scale = 800 / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_CUBIC)

    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # CLAHE — improves contrast for faded / poorly scanned cards
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # Otsu binarisation
    _, binary = cv2.threshold(
        enhanced, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary


def _try_easyocr(img):
    """Attempt OCR with EasyOCR. Returns raw string or None on import error."""
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        results = reader.readtext(img, detail=0, paragraph=False)
        return ' '.join(results)
    except Exception:
        return None


def _try_tesseract(img):
    """Fallback OCR using pytesseract if available."""
    try:
        import pytesseract
        text = pytesseract.image_to_string(img, config='--psm 6')
        return text
    except Exception:
        return None


def extract_text(img_bgr):
    """
    Run OCR on a BGR numpy image.
    Returns raw extracted text string.
    Tries EasyOCR first, falls back to Tesseract, then returns empty string.
    """
    processed = _preprocess_for_ocr(img_bgr)

    text = _try_easyocr(processed)
    if not text:
        text = _try_tesseract(processed)
    if not text:
        text = ''

    return text.upper().strip()


# --------------------------------------------------------------------------- #
#  Field extraction
# --------------------------------------------------------------------------- #

def extract_pan_number(text):
    """Extract PAN number matching official format: AAAAA9999A"""
    match = PAN_REGEX.search(text)
    return match.group(0) if match else ''


def extract_dob(text):
    """Extract date of birth in DD/MM/YYYY or DD-MM-YYYY format."""
    match = DOB_REGEX.search(text)
    return match.group(1) if match else ''


def extract_name(text, pan_number=''):
    """
    Extract the card holder name.
    Heuristic: longest all-caps word sequence that is NOT the PAN number.
    """
    # Remove PAN number from text before searching
    cleaned = text.replace(pan_number, '') if pan_number else text

    candidates = NAME_REGEX.findall(cleaned)
    if not candidates:
        return ''

    # Filter out known non-name strings
    noise = {'INCOME TAX DEPARTMENT', 'GOVT OF INDIA', 'PERMANENT ACCOUNT',
             'NUMBER', 'DATE OF BIRTH', 'FATHER'}
    valid = [c.strip() for c in candidates
             if c.strip() not in noise and len(c.strip()) > 3]

    if not valid:
        return ''
    return max(valid, key=len)   # longest match is usually the full name


# --------------------------------------------------------------------------- #
#  Validation scoring
# --------------------------------------------------------------------------- #

def validate_pan_format(pan_number):
    """
    Returns a score 0–1 based on PAN format correctness.
    1.0 = perfectly valid format
    0.5 = partial match (right length, wrong pattern)
    0.0 = empty or completely wrong
    """
    if not pan_number:
        return 0.0
    if PAN_REGEX.fullmatch(pan_number):
        return 1.0
    # Partial credit: correct length but minor issue
    if len(pan_number) == 10:
        return 0.5
    return 0.0


def validate_dob_format(dob):
    """Returns 1.0 if DOB is a plausible date, 0.0 otherwise."""
    if not dob:
        return 0.0
    try:
        parts = re.split(r'[\/\-]', dob)
        if len(parts) != 3:
            return 0.0
        day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
        if 1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= 2025:
            return 1.0
        return 0.0
    except ValueError:
        return 0.0


def validate_name(name):
    """Returns 1.0 if name looks like a real name (letters and spaces only)."""
    if not name or len(name.strip()) < 3:
        return 0.0
    if re.fullmatch(r'[A-Z][A-Z\s]+', name.strip()):
        return 1.0
    return 0.5


def run_ocr_validation(img_bgr):
    """
    Master function called by the document verification view.

    Returns
    -------
    dict with keys:
        raw_text        : full OCR output
        pan_number      : extracted PAN string
        name            : extracted name
        dob             : extracted DOB
        ocr_score       : float 0–1  (used as doc_ocr_valid in the DB)
        field_scores    : per-field breakdown for transparency
    """
    raw_text   = extract_text(img_bgr)
    pan_number = extract_pan_number(raw_text)
    dob        = extract_dob(raw_text)
    name       = extract_name(raw_text, pan_number)

    # Individual field validation scores
    pan_score  = validate_pan_format(pan_number)
    dob_score  = validate_dob_format(dob)
    name_score = validate_name(name)

    # Weighted OCR score
    # PAN number is the most critical field (weight 0.5)
    # Name and DOB are supporting (0.25 each)
    ocr_score = (pan_score * 0.50) + (dob_score * 0.25) + (name_score * 0.25)

    return {
        'raw_text':     raw_text,
        'pan_number':   pan_number,
        'name':         name,
        'dob':          dob,
        'ocr_score':    round(ocr_score, 4),
        'field_scores': {
            'pan':  pan_score,
            'dob':  dob_score,
            'name': name_score,
        }
    }