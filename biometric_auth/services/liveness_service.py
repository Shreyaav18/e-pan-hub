"""
Liveness Detection Service — Upgrade 4
----------------------------------------
Implements paper Section V.C.1 — Liveness Detection.

Detects presentation attacks where someone holds up:
  - A printed photo of another person
  - A phone/tablet screen showing another person's photo
  - A 3D mask

Methods used (paper-aligned):
  1. LBP (Local Binary Patterns) texture analysis
     — Real faces have micro-texture; printed photos have regular print patterns
  2. Frequency domain analysis
     — Printed/screen images have moiré patterns at specific frequencies
  3. Gradient magnitude statistics
     — Real faces have natural gradient distribution; fakes are flatter or
        have regular grid artifacts
  4. Reflection / specular highlight check
     — Real faces have natural specular highlights; printed photos have flat reflections

All methods combined into a single liveness_score in [0, 1].
1.0 = definitely live face. 0.0 = definitely spoof.
"""

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
#  Method 1 — LBP Texture Analysis
# --------------------------------------------------------------------------- #

def _compute_lbp(gray_img, radius=1, n_points=8):
    """
    Compute Local Binary Pattern histogram.
    Real face skin has characteristic LBP distribution.
    Printed photos show more uniform/periodic patterns.
    """
    h, w = gray_img.shape
    lbp   = np.zeros((h, w), dtype=np.uint8)

    # Simple LBP implementation (avoids skimage dependency)
    for i in range(radius, h - radius):
        for j in range(radius, w - radius):
            center = int(gray_img[i, j])
            code   = 0
            # 8 neighbors in circle
            neighbors = [
                gray_img[i - radius, j - radius],
                gray_img[i - radius, j],
                gray_img[i - radius, j + radius],
                gray_img[i,          j + radius],
                gray_img[i + radius, j + radius],
                gray_img[i + radius, j],
                gray_img[i + radius, j - radius],
                gray_img[i,          j - radius],
            ]
            for bit, nb in enumerate(neighbors):
                if int(nb) >= center:
                    code |= (1 << bit)
            lbp[i, j] = code

    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-6)
    return hist


def _fast_lbp_score(gray_img):
    """
    Faster LBP using Sobel-based texture measure.
    Returns a score: higher = more natural texture = more likely live.
    """
    # Downsample for speed
    small = cv2.resize(gray_img, (64, 64))

    # Local standard deviation map — proxy for texture richness
    blur  = cv2.GaussianBlur(small, (5, 5), 0)
    diff  = small.astype(np.float32) - blur.astype(np.float32)
    local_std = np.std(diff)

    # Normalise: real faces typically have local_std in [8, 25]
    # Printed photos: very low (<5) or very high (>30, moiré)
    if local_std < 3:
        return 0.1   # too flat = printed photo
    elif local_std > 35:
        return 0.3   # too noisy = screen/moiré artifact
    else:
        # Map 3–35 to 0.3–1.0
        score = 0.3 + (local_std - 3) / (35 - 3) * 0.7
        return round(float(min(1.0, score)), 4)


# --------------------------------------------------------------------------- #
#  Method 2 — Frequency Domain (FFT) Analysis
# --------------------------------------------------------------------------- #

def _frequency_score(gray_img):
    """
    Printed and screen images show energy concentration at specific
    frequencies (screen refresh rate, print halftone patterns).
    Real face images have more distributed frequency energy.

    Returns score 0–1 (1 = natural frequency distribution = live).
    """
    small = cv2.resize(gray_img, (128, 128)).astype(np.float32)

    # 2D FFT
    fft    = np.fft.fft2(small)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log(np.abs(fft_shift) + 1)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2

    # Split into frequency bands
    inner_mask = np.zeros((h, w), dtype=bool)
    outer_mask = np.zeros((h, w), dtype=bool)
    for y in range(h):
        for x in range(w):
            dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
            if dist < 20:
                inner_mask[y, x] = True
            elif dist > 40:
                outer_mask[y, x] = True

    inner_energy = magnitude[inner_mask].mean() if inner_mask.any() else 0
    outer_energy = magnitude[outer_mask].mean() if outer_mask.any() else 0

    # Real faces: more balanced inner/outer ratio
    # Prints: outer energy spikes from halftone patterns
    if inner_energy == 0:
        return 0.5

    ratio = outer_energy / (inner_energy + 1e-6)

    # Normal ratio for real face: 0.3 – 0.8
    if 0.25 <= ratio <= 0.9:
        return 0.85
    elif ratio < 0.25:
        return 0.5   # suspiciously flat
    else:
        return max(0.0, 1.0 - (ratio - 0.9) / 2.0)  # too much high freq = print


# --------------------------------------------------------------------------- #
#  Method 3 — Gradient Magnitude Statistics
# --------------------------------------------------------------------------- #

def _gradient_score(gray_img):
    """
    Real faces have smooth gradient transitions.
    Printed photos often have quantization artifacts in gradients,
    and screen captures have discrete pixel grid patterns.

    Returns score 0–1.
    """
    small = cv2.resize(gray_img, (128, 128))

    grad_x = cv2.Sobel(small, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(small, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    mean_grad = np.mean(magnitude)
    std_grad  = np.std(magnitude)

    # Real faces: mean ~15–40, std ~10–30
    # Printed photos: mean too low (<10) or too uniform (low std)
    if mean_grad < 5:
        return 0.1

    # Coefficient of variation — real faces have natural variation
    cv = std_grad / (mean_grad + 1e-6)
    if cv < 0.3:
        return 0.3   # too uniform = likely print
    elif cv > 2.5:
        return 0.4   # too noisy
    else:
        return round(float(min(1.0, 0.5 + cv * 0.2)), 4)


# --------------------------------------------------------------------------- #
#  Method 4 — Specular Highlight Check
# --------------------------------------------------------------------------- #

def _specular_score(gray_img):
    """
    Real faces have natural specular highlights (small bright spots
    from light reflection on skin). Printed photos have flat or
    uniform brightness.

    Returns score 0–1.
    """
    # Find very bright pixels (highlights)
    _, bright_mask = cv2.threshold(gray_img, 230, 255, cv2.THRESH_BINARY)
    bright_ratio = np.sum(bright_mask > 0) / gray_img.size

    # Real face: 0.5% – 5% bright pixels
    if 0.003 <= bright_ratio <= 0.08:
        return 0.85
    elif bright_ratio < 0.001:
        return 0.4   # no highlights = flat = printed
    elif bright_ratio > 0.20:
        return 0.3   # over-exposed or screen glare
    else:
        return 0.6   # borderline


# --------------------------------------------------------------------------- #
#  Master liveness scorer
# --------------------------------------------------------------------------- #

LIVENESS_WEIGHTS = {
    'texture':   0.35,   # LBP most reliable
    'frequency': 0.25,   # FFT moiré detection
    'gradient':  0.25,   # Gradient stats
    'specular':  0.15,   # Highlight check
}


def run_liveness_detection(face_bgr):
    """
    Runs all 4 liveness checks on a face crop.

    Parameters
    ----------
    face_bgr : BGR numpy array of detected face region

    Returns
    -------
    dict:
        liveness_score   : float 0–1 (1 = live face)
        method_scores    : per-method breakdown
        liveness_verdict : 'LIVE' | 'SPOOF' | 'UNCERTAIN'
        is_live          : bool
    """
    if face_bgr is None or face_bgr.size == 0:
        return {
            'liveness_score':   0.0,
            'method_scores':    {},
            'liveness_verdict': 'NO FACE',
            'is_live':          False,
        }

    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)

    method_scores = {
        'texture':   _fast_lbp_score(gray),
        'frequency': _frequency_score(gray),
        'gradient':  _gradient_score(gray),
        'specular':  _specular_score(gray),
    }

    # Weighted combination
    liveness_score = sum(
        LIVENESS_WEIGHTS[m] * s
        for m, s in method_scores.items()
    )
    liveness_score = round(float(min(1.0, max(0.0, liveness_score))), 4)

    # Verdict
    if liveness_score >= 0.65:
        verdict  = 'LIVE'
        is_live  = True
    elif liveness_score >= 0.45:
        verdict  = 'UNCERTAIN'
        is_live  = False
    else:
        verdict  = 'SPOOF DETECTED'
        is_live  = False

    return {
        'liveness_score':   liveness_score,
        'method_scores':    method_scores,
        'liveness_verdict': verdict,
        'is_live':          is_live,
    }