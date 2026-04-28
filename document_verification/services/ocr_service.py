import re
import cv2
import numpy as np

PAN_REGEX  = re.compile(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b')
DOB_REGEX  = re.compile(r'\b(\d{2}[\/\-]\d{2}[\/\-]\d{4})\b')
NAME_REGEX = re.compile(r'\b([A-Z][A-Z\s]{3,40})\b')


def _preprocess_for_ocr(img):
    h, w = img.shape[:2]
    if w < 800:
        scale = 800 / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_CUBIC)

    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    _, binary = cv2.threshold(
        enhanced, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary


def _try_easyocr(img):
    try:
        import easyocr
        reader  = easyocr.Reader(['en'], gpu=False, verbose=False)
        results = reader.readtext(img, detail=0, paragraph=False)
        return ' '.join(results)
    except Exception:
        return None


def _try_tesseract(img):
    try:
        import pytesseract
        return pytesseract.image_to_string(img, config='--psm 6')
    except Exception:
        return None


def _ocr_image(img_bgr):
    processed = _preprocess_for_ocr(img_bgr)
    text = _try_easyocr(processed)
    if not text:
        text = _try_tesseract(processed)
    return (text or '').upper().strip()


def extract_text(img_bgr):
    return _ocr_image(img_bgr)


def extract_pan_number(text):
    match = PAN_REGEX.search(text)
    return match.group(0) if match else ''


def extract_dob(text):
    match = DOB_REGEX.search(text)
    return match.group(1) if match else ''


def extract_name(text, pan_number=''):
    cleaned    = text.replace(pan_number, '') if pan_number else text
    candidates = NAME_REGEX.findall(cleaned)
    if not candidates:
        return ''

    noise = {'INCOME TAX DEPARTMENT', 'GOVT OF INDIA', 'PERMANENT ACCOUNT',
             'NUMBER', 'DATE OF BIRTH', 'FATHER'}
    valid = [c.strip() for c in candidates
             if c.strip() not in noise and len(c.strip()) > 3]

    return max(valid, key=len) if valid else ''


def validate_pan_format(pan_number):
    if not pan_number:
        return 0.0
    if PAN_REGEX.fullmatch(pan_number):
        return 1.0
    if len(pan_number) == 10:
        return 0.5
    return 0.0


def validate_dob_format(dob):
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
    if not name or len(name.strip()) < 3:
        return 0.0
    if re.fullmatch(r'[A-Z][A-Z\s]+', name.strip()):
        return 1.0
    return 0.5


def _ocr_with_field_crops(fields):
    pan_text    = _ocr_image(fields['pan']['crop'])   if 'pan'    in fields else ''
    name_text   = _ocr_image(fields['name']['crop'])  if 'name'   in fields else ''
    dob_text    = _ocr_image(fields['dob']['crop'])   if 'dob'    in fields else ''
    father_text = _ocr_image(fields['father']['crop']) if 'father' in fields else ''

    pan_number = extract_pan_number(pan_text)
    dob        = extract_dob(dob_text)
    name       = name_text.strip() if name_text else ''
    father     = father_text.strip() if father_text else ''

    raw_combined = ' '.join([pan_text, name_text, dob_text, father_text])
    return pan_number, name, dob, father, raw_combined


def _ocr_full_image(img_bgr):
    raw_text   = _ocr_image(img_bgr)
    pan_number = extract_pan_number(raw_text)
    dob        = extract_dob(raw_text)
    name       = extract_name(raw_text, pan_number)
    father     = ''
    return pan_number, name, dob, father, raw_text


def run_ocr_validation(img_bgr, fields=None):
    if fields and any(k in fields for k in ('pan', 'name', 'dob')):
        pan_number, name, dob, father, raw_text = _ocr_with_field_crops(fields)
    else:
        pan_number, name, dob, father, raw_text = _ocr_full_image(img_bgr)

    pan_score  = validate_pan_format(pan_number)
    dob_score  = validate_dob_format(dob)
    name_score = validate_name(name)

    ocr_score = (pan_score * 0.50) + (dob_score * 0.25) + (name_score * 0.25)

    return {
        'raw_text':     raw_text,
        'pan_number':   pan_number,
        'name':         name,
        'father':       father,
        'dob':          dob,
        'ocr_score':    round(ocr_score, 4),
        'field_scores': {
            'pan':  pan_score,
            'dob':  dob_score,
            'name': name_score,
        }
    }