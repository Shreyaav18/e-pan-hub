import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from .security_service import compute_security_score


def check_font_consistency(gray_img):
    h, w = gray_img.shape
    strip_height = max(1, h // 10)
    variances = []

    for i in range(0, h - strip_height, strip_height):
        strip   = gray_img[i:i + strip_height, :]
        lap_var = cv2.Laplacian(strip, cv2.CV_64F).var()
        variances.append(lap_var)

    if not variances or max(variances) == 0:
        return 0.5

    mean_var = np.mean(variances)
    std_var  = np.std(variances)
    cv       = std_var / (mean_var + 1e-6)
    score    = max(0.0, 1.0 - (cv / 2.0))
    return round(float(score), 4)


def check_edge_consistency(gray_img):
    edges = cv2.Canny(gray_img, threshold1=50, threshold2=150)
    h, w  = edges.shape
    block_size = max(1, h // 8)

    edge_densities = []
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block   = edges[y:y + block_size, x:x + block_size]
            density = np.sum(block > 0) / (block_size * block_size)
            edge_densities.append(density)

    if not edge_densities:
        return 0.5

    variance = np.var(edge_densities)
    score    = max(0.0, 1.0 - (variance / 0.05))
    return round(float(score), 4)


def check_color_histogram(bgr_img):
    h, w       = bgr_img.shape[:2]
    mid_h, mid_w = h // 2, w // 2

    quadrants = [
        bgr_img[0:mid_h,  0:mid_w],
        bgr_img[0:mid_h,  mid_w:w],
        bgr_img[mid_h:h,  0:mid_w],
        bgr_img[mid_h:h,  mid_w:w],
    ]

    histograms = []
    for quad in quadrants:
        hist = cv2.calcHist([quad], [0, 1, 2], None,
                            [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        histograms.append(hist)

    distances = []
    for i in range(len(histograms)):
        for j in range(i + 1, len(histograms)):
            d = cv2.compareHist(histograms[i], histograms[j],
                                cv2.HISTCMP_BHATTACHARYYA)
            distances.append(d)

    avg_distance = np.mean(distances) if distances else 1.0
    score        = max(0.0, 1.0 - avg_distance)
    return round(float(score), 4)


def check_structural_alignment(gray_img):
    h, w = gray_img.shape

    corners   = cv2.cornerHarris(np.float32(gray_img), blockSize=2, ksize=3, k=0.04)
    corners   = cv2.dilate(corners, None)
    threshold = 0.01 * corners.max()
    corner_mask = corners > threshold

    if not corner_mask.any():
        return 0.5

    grid_h, grid_w  = h // 3, w // 3
    occupied_cells  = 0

    for row in range(3):
        for col in range(3):
            cell = corner_mask[
                row * grid_h:(row + 1) * grid_h,
                col * grid_w:(col + 1) * grid_w
            ]
            if cell.any():
                occupied_cells += 1

    return round(float(occupied_cells / 9.0), 4)


def check_metadata(image_path):
    if not image_path:
        return 0.7

    try:
        from PIL import Image as PILImage
        from PIL.ExifTags import TAGS

        img       = PILImage.open(image_path)
        exif_data = img._getexif()

        if not exif_data:
            return 0.8

        decoded = {TAGS.get(k, k): v for k, v in exif_data.items()}

        edit_software = ['photoshop', 'gimp', 'paint', 'pixlr',
                         'snapseed', 'lightroom', 'affinity']
        software = str(decoded.get('Software', '')).lower()
        for flag in edit_software:
            if flag in software:
                return 0.1

        orig_time = decoded.get('DateTimeOriginal', '')
        mod_time  = decoded.get('DateTime', '')
        if orig_time and mod_time and orig_time != mod_time:
            return 0.4

        return 1.0

    except Exception:
        return 0.7


def check_ssim(gray_img, reference_gray=None):
    if reference_gray is not None:
        ref_resized = cv2.resize(reference_gray,
                                 (gray_img.shape[1], gray_img.shape[0]))
        score, _    = ssim(gray_img, ref_resized, full=True)
        return round(float(max(0.0, score)), 4)

    h, w   = gray_img.shape
    left   = gray_img[:, :w // 2]
    right  = gray_img[:, w // 2:w // 2 * 2]
    right_r = cv2.resize(right, (left.shape[1], left.shape[0]))
    score, _ = ssim(left, right_r, full=True)
    normalized = (float(score) + 1.0) / 2.0
    return round(normalized, 4)


FEATURE_WEIGHTS = {
    'ocr_valid':          0.25,
    'security_features':  0.20,
    'edge_consistency':   0.15,
    'font_consistency':   0.12,
    'color_histogram':    0.12,
    'structural_align':   0.08,
    'metadata_clean':     0.05,
    'ssim':               0.03,
}


def compute_sdoc(img_bgr, ocr_score, image_path=None, reference_gray=None, raw_ocr_text=''):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    security_result = compute_security_score(img_bgr, raw_ocr_text, reference_gray)

    feature_scores = {
        'ocr_valid':         ocr_score,
        'security_features': security_result['security_score'],
        'font_consistency':  check_font_consistency(gray),
        'edge_consistency':  check_edge_consistency(gray),
        'color_histogram':   check_color_histogram(img_bgr),
        'structural_align':  check_structural_alignment(gray),
        'metadata_clean':    check_metadata(image_path),
        'ssim':              check_ssim(gray, reference_gray),
    }

    sdoc = sum(
        FEATURE_WEIGHTS[feat] * score
        for feat, score in feature_scores.items()
    )
    sdoc = round(float(max(0.0, min(1.0, sdoc))), 4)

    diff_contours = _get_diff_contours(gray)

    return {
        'sdoc':                    sdoc,
        'feature_scores':          feature_scores,
        'security_feature_scores': security_result['security_feature_scores'],
        'diff_contours':           diff_contours,
    }


def _get_diff_contours(gray_img):
    edges  = cv2.Canny(gray_img, 80, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return [c for c in contours if cv2.contourArea(c) > 200]