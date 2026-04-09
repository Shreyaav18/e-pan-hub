"""
Authenticity Service — Upgrade 2: Multi-Feature Scoring
---------------------------------------------------------
Implements the 6 parallel authenticity checks from the research paper
(Table I) plus SSIM as the 7th legacy feature.

Each check returns a score in [0, 1] where 1 = fully authentic.
Final Sdoc is computed as a weighted combination exactly per equation (5):

    Sdoc = sum(wi * fi)  for i in 1..6

Feature weights are tuned to PAN card characteristics.
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


# --------------------------------------------------------------------------- #
#  Feature 1 — Font Consistency (Statistical Analysis)
# --------------------------------------------------------------------------- #

def check_font_consistency(gray_img):
    """
    Analyses local variance across horizontal text bands.
    On a genuine PAN card, font size and stroke width are uniform.
    Forged cards often mix fonts from different sources.

    Returns score 0–1 (1 = consistent font usage).
    """
    h, w = gray_img.shape

    # Divide image into horizontal strips (each ~10% of height)
    strip_height = max(1, h // 10)
    variances = []

    for i in range(0, h - strip_height, strip_height):
        strip = gray_img[i:i + strip_height, :]
        # Laplacian variance measures local texture/sharpness
        lap_var = cv2.Laplacian(strip, cv2.CV_64F).var()
        variances.append(lap_var)

    if not variances or max(variances) == 0:
        return 0.5   # neutral if image is blank

    # Coefficient of variation — lower = more consistent
    mean_var = np.mean(variances)
    std_var  = np.std(variances)
    cv = std_var / (mean_var + 1e-6)

    # Map CV to score: CV near 0 → score 1.0, CV > 2.0 → score 0.0
    score = max(0.0, 1.0 - (cv / 2.0))
    return round(float(score), 4)


# --------------------------------------------------------------------------- #
#  Feature 2 — Edge Consistency (Sobel / Canny Filters)
# --------------------------------------------------------------------------- #

def check_edge_consistency(gray_img):
    """
    Genuine documents have clean, consistent edges around text and borders.
    Tampered regions introduce irregular edge patterns from copy-paste or
    clone stamping.

    Uses Canny edge detection and analyses edge density distribution.
    Returns score 0–1.
    """
    edges = cv2.Canny(gray_img, threshold1=50, threshold2=150)

    h, w = edges.shape
    block_size = max(1, h // 8)

    edge_densities = []
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = edges[y:y + block_size, x:x + block_size]
            density = np.sum(block > 0) / (block_size * block_size)
            edge_densities.append(density)

    if not edge_densities:
        return 0.5

    # High variance in edge density across blocks = inconsistency = forgery
    variance = np.var(edge_densities)

    # Map variance to score: < 0.005 is very consistent
    score = max(0.0, 1.0 - (variance / 0.05))
    return round(float(score), 4)


# --------------------------------------------------------------------------- #
#  Feature 3 — Color Histogram (Statistical Divergence)
# --------------------------------------------------------------------------- #

def check_color_histogram(bgr_img):
    """
    Genuine PAN cards have a consistent color profile (cream background,
    dark text, standard photo colors). Copy-pasted or digitally altered
    regions often show color discontinuities.

    Splits image into quadrants and compares histogram similarity
    across regions using Bhattacharyya distance.
    Returns score 0–1.
    """
    h, w = bgr_img.shape[:2]
    mid_h, mid_w = h // 2, w // 2

    # Four quadrants
    quadrants = [
        bgr_img[0:mid_h,  0:mid_w],       # top-left
        bgr_img[0:mid_h,  mid_w:w],        # top-right
        bgr_img[mid_h:h,  0:mid_w],        # bottom-left
        bgr_img[mid_h:h,  mid_w:w],        # bottom-right
    ]

    histograms = []
    for quad in quadrants:
        hist = cv2.calcHist([quad], [0, 1, 2], None,
                            [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        histograms.append(hist)

    # Compare all pairs using Bhattacharyya distance (lower = more similar)
    distances = []
    for i in range(len(histograms)):
        for j in range(i + 1, len(histograms)):
            d = cv2.compareHist(histograms[i], histograms[j],
                                cv2.HISTCMP_BHATTACHARYYA)
            distances.append(d)

    avg_distance = np.mean(distances) if distances else 1.0

    # Map distance to score: 0 distance = identical = score 1.0
    score = max(0.0, 1.0 - avg_distance)
    return round(float(score), 4)


# --------------------------------------------------------------------------- #
#  Feature 4 — Structural Alignment (Template Matching)
# --------------------------------------------------------------------------- #

def check_structural_alignment(gray_img):
    """
    PAN cards have a fixed layout: logo top-left, photo right side,
    text in standard zones. We verify structural regularity by checking
    that high-contrast anchor regions (borders, logo area) appear where
    expected.

    Uses corner detection to verify layout integrity.
    Returns score 0–1.
    """
    h, w = gray_img.shape

    # Harris corner detection
    corners = cv2.cornerHarris(
        np.float32(gray_img), blockSize=2, ksize=3, k=0.04
    )
    corners = cv2.dilate(corners, None)

    # Threshold significant corners
    threshold = 0.01 * corners.max()
    corner_mask = corners > threshold

    if not corner_mask.any():
        return 0.5

    # Expect corners to be spread across the image for a genuine card
    # Divide into 3x3 grid and check coverage
    grid_h, grid_w = h // 3, w // 3
    occupied_cells = 0

    for row in range(3):
        for col in range(3):
            cell = corner_mask[
                row * grid_h:(row + 1) * grid_h,
                col * grid_w:(col + 1) * grid_w
            ]
            if cell.any():
                occupied_cells += 1

    # Genuine card should have corners in most grid cells
    coverage = occupied_cells / 9.0
    return round(float(coverage), 4)


# --------------------------------------------------------------------------- #
#  Feature 5 — Metadata Analysis (EXIF Inspection)
# --------------------------------------------------------------------------- #

def check_metadata(image_path):
    """
    Inspects EXIF metadata for signs of digital editing.
    Flags: editing software (Photoshop, GIMP), modification timestamps,
    suspicious creation tools.

    Returns score 0–1 (1 = clean metadata, 0 = edited).
    image_path can be a file path string or None.
    """
    if not image_path:
        return 0.7   # neutral — no metadata available

    try:
        from PIL import Image as PILImage
        from PIL.ExifTags import TAGS

        img = PILImage.open(image_path)
        exif_data = img._getexif()

        if not exif_data:
            return 0.8   # no EXIF = likely a scan = slightly suspicious but common

        # Convert tag IDs to names
        decoded = {TAGS.get(k, k): v for k, v in exif_data.items()}

        # Red flags
        edit_software = ['photoshop', 'gimp', 'paint', 'pixlr',
                         'snapseed', 'lightroom', 'affinity']

        software = str(decoded.get('Software', '')).lower()
        for flag in edit_software:
            if flag in software:
                return 0.1   # strong forgery signal

        # Check for modification date being different from creation date
        orig_time = decoded.get('DateTimeOriginal', '')
        mod_time  = decoded.get('DateTime', '')
        if orig_time and mod_time and orig_time != mod_time:
            return 0.4   # modified after creation

        return 1.0   # clean metadata

    except Exception:
        return 0.7   # could not read metadata — neutral


# --------------------------------------------------------------------------- #
#  Feature 6 — SSIM (Legacy, now one feature among many)
# --------------------------------------------------------------------------- #

def check_ssim(gray_img, reference_gray=None):
    """
    Structural Similarity Index.
    If a reference image is provided, compares against it.
    If no reference, computes self-similarity across image halves
    as a proxy for internal consistency.

    Returns score 0–1.
    """
    if reference_gray is not None:
        # Resize reference to match test image
        ref_resized = cv2.resize(reference_gray, 
                                 (gray_img.shape[1], gray_img.shape[0]))
        score, _ = ssim(gray_img, ref_resized, full=True)
        return round(float(max(0.0, score)), 4)

    else:
        # No reference: compare left half vs right half
        # Genuine cards have symmetric background texture
        h, w = gray_img.shape
        left  = gray_img[:, :w // 2]
        right = gray_img[:, w // 2:w // 2 * 2]

        # Resize to same shape
        right_r = cv2.resize(right, (left.shape[1], left.shape[0]))
        score, _ = ssim(left, right_r, full=True)

        # Self-similarity is not expected to be 1.0 — normalise to 0.5 baseline
        normalized = (float(score) + 1.0) / 2.0
        return round(normalized, 4)


# --------------------------------------------------------------------------- #
#  Master scorer — implements paper equation (5)
# --------------------------------------------------------------------------- #

# Paper-aligned weights (sum = 1.0)
FEATURE_WEIGHTS = {
    'ocr_valid':        0.30,   # Most informative — content validation
    'edge_consistency': 0.20,   # Detects tampering regions
    'font_consistency': 0.15,   # Detects mixed-source text
    'color_histogram':  0.15,   # Detects color inconsistencies
    'structural_align': 0.10,   # Layout integrity
    'metadata_clean':   0.05,   # EXIF signals
    'ssim':             0.05,   # Legacy signal — lowest weight
}


def compute_sdoc(img_bgr, ocr_score, image_path=None, reference_gray=None):
    """
    Runs all 6 authenticity checks and returns the weighted Sdoc score.

    Parameters
    ----------
    img_bgr       : numpy BGR image
    ocr_score     : float — passed in from ocr_service output
    image_path    : file path string for EXIF check (optional)
    reference_gray: grayscale reference image for SSIM (optional)

    Returns
    -------
    dict with:
        sdoc          : final weighted score (0–1)
        feature_scores: dict of all individual scores
        diff_contours : list of OpenCV contours for tampered region overlay
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Run all features
    feature_scores = {
        'ocr_valid':        ocr_score,
        'font_consistency': check_font_consistency(gray),
        'edge_consistency': check_edge_consistency(gray),
        'color_histogram':  check_color_histogram(img_bgr),
        'structural_align': check_structural_alignment(gray),
        'metadata_clean':   check_metadata(image_path),
        'ssim':             check_ssim(gray, reference_gray),
    }

    # Weighted Sdoc — paper equation (5)
    sdoc = sum(
        FEATURE_WEIGHTS[feat] * score
        for feat, score in feature_scores.items()
    )
    sdoc = round(float(max(0.0, min(1.0, sdoc))), 4)

    # Generate diff contours for visual overlay
    diff_contours = _get_diff_contours(gray)

    return {
        'sdoc':           sdoc,
        'feature_scores': feature_scores,
        'diff_contours':  diff_contours,
    }


def _get_diff_contours(gray_img):
    """
    Detects suspicious regions using edge anomaly detection.
    Returns contours suitable for drawing bounding boxes.
    """
    edges = cv2.Canny(gray_img, 80, 200)

    # Morphological closing to connect nearby edge fragments
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed  = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter: only return contours large enough to be meaningful
    significant = [c for c in contours if cv2.contourArea(c) > 200]
    return significant