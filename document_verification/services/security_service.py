import cv2
import numpy as np


def check_itd_emblem(gray_img, reference_gray=None):
    h, w = gray_img.shape
    roi = gray_img[0:int(h * 0.35), 0:int(w * 0.25)]

    if reference_gray is not None:
        ref_roi = reference_gray[
            0:int(reference_gray.shape[0] * 0.35),
            0:int(reference_gray.shape[1] * 0.25)
        ]
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(ref_roi, None)
        kp2, des2 = orb.detectAndCompute(roi, None)

        if des1 is None or des2 is None or len(kp1) < 5 or len(kp2) < 5:
            return 0.4

        bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        good    = [m for m in matches if m.distance < 50]
        score   = min(1.0, len(good) / max(len(kp1), 1) * 2)
        return round(float(score), 4)

    edges   = cv2.Canny(roi, 50, 150)
    density = np.sum(edges > 0) / max(edges.size, 1)
    score   = min(1.0, density * 10)
    return round(float(score), 4)


def check_qr_code(img_bgr):
    try:
        from pyzbar.pyzbar import decode
        decoded = decode(img_bgr)
        if decoded:
            return 1.0
        detector = cv2.QRCodeDetector()
        data, _, _ = detector.detectAndDecode(img_bgr)
        if data:
            return 1.0
        return 0.3
    except Exception:
        try:
            detector = cv2.QRCodeDetector()
            data, _, _ = detector.detectAndDecode(img_bgr)
            return 1.0 if data else 0.3
        except Exception:
            return 0.5


def check_face_present(img_bgr):
    h, w = img_bgr.shape[:2]
    photo_roi  = img_bgr[int(h * 0.05):int(h * 0.75), int(w * 0.65):w]
    photo_gray = cv2.cvtColor(photo_roi, cv2.COLOR_BGR2GRAY)

    cascade_path  = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade  = cv2.CascadeClassifier(cascade_path)
    faces         = face_cascade.detectMultiScale(
        photo_gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20)
    )

    if len(faces) > 0:
        return 1.0

    _, binary  = cv2.threshold(photo_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink_ratio  = np.sum(binary > 0) / max(binary.size, 1)
    return 0.6 if ink_ratio > 0.05 else 0.1


def check_signature_zone(gray_img):
    h, w = gray_img.shape
    sig_roi = gray_img[int(h * 0.75):int(h * 0.95), int(w * 0.25):int(w * 0.75)]

    if sig_roi.size == 0:
        return 0.5

    _, binary  = cv2.threshold(sig_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink_ratio  = np.sum(binary > 0) / max(binary.size, 1)

    if 0.01 < ink_ratio < 0.30:
        return 1.0
    if ink_ratio <= 0.01:
        return 0.1
    return 0.5


def check_govt_text(raw_ocr_text):
    required = ['INCOME TAX', 'GOVT OF INDIA', 'PERMANENT ACCOUNT']
    text     = raw_ocr_text.upper()
    found    = sum(1 for term in required if term in text)
    return round(found / len(required), 4)


def check_background_pattern(gray_img, reference_gray=None):
    h, w = gray_img.shape

    bg_regions = [
        gray_img[int(h * 0.30):int(h * 0.70), int(w * 0.05):int(w * 0.30)],
        gray_img[int(h * 0.30):int(h * 0.70), int(w * 0.35):int(w * 0.60)],
    ]

    if reference_gray is not None:
        from skimage.metrics import structural_similarity as ssim
        scores = []
        ref_regions = [
            reference_gray[int(h * 0.30):int(h * 0.70), int(w * 0.05):int(w * 0.30)],
            reference_gray[int(h * 0.30):int(h * 0.70), int(w * 0.35):int(w * 0.60)],
        ]
        for reg, ref in zip(bg_regions, ref_regions):
            if reg.size == 0 or ref.size == 0:
                continue
            ref_r = cv2.resize(ref, (reg.shape[1], reg.shape[0]))
            s, _  = ssim(reg, ref_r, full=True)
            scores.append(float(s))
        return round(float(np.mean(scores)), 4) if scores else 0.5

    textures = []
    for reg in bg_regions:
        if reg.size == 0:
            continue
        lap_var = cv2.Laplacian(reg, cv2.CV_64F).var()
        textures.append(lap_var)

    if not textures:
        return 0.5

    cv_val = np.std(textures) / (np.mean(textures) + 1e-6)
    return round(float(max(0.0, 1.0 - cv_val)), 4)


SECURITY_WEIGHTS = {
    'govt_text':          0.25,
    'face_present':       0.25,
    'itd_emblem':         0.20,
    'signature':          0.15,
    'background_pattern': 0.10,
    'qr_code':            0.05,
}


def compute_security_score(img_bgr, raw_ocr_text='', reference_gray=None):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    scores = {
        'itd_emblem':         check_itd_emblem(gray, reference_gray),
        'qr_code':            check_qr_code(img_bgr),
        'face_present':       check_face_present(img_bgr),
        'signature':          check_signature_zone(gray),
        'govt_text':          check_govt_text(raw_ocr_text),
        'background_pattern': check_background_pattern(gray, reference_gray),
    }

    total = sum(SECURITY_WEIGHTS[k] * v for k, v in scores.items())
    total = round(float(max(0.0, min(1.0, total))), 4)

    return {
        'security_score':          total,
        'security_feature_scores': scores,
    }