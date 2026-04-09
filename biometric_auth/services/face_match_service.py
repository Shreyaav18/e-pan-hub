"""
Face Match Service — Upgrade 3: Face Matching
----------------------------------------------
Implements paper Section V.C.1 — Facial Recognition.

Pipeline:
  1. Detect and crop face from PAN card photo region
  2. Detect and crop face from live selfie
  3. Extract d-dimensional embeddings using a FaceNet-inspired CNN
  4. Compute Euclidean distance between embeddings
  5. Return similarity score and match verdict

Paper equation (6):
    Distance(Fcapture, Ftemplate) = ||phi(Fcapture) - phi(Ftemplate)||_2
    Authentication succeeds if Distance < tau_face
"""

import cv2
import numpy as np
import os


# Distance threshold from paper (tau_face)
# Below this = same person
FACE_DISTANCE_THRESHOLD = 0.6


# --------------------------------------------------------------------------- #
#  Face Detection
# --------------------------------------------------------------------------- #

def detect_face(img_bgr):
    """
    Detect and return the largest face crop from an image.
    Uses OpenCV's Haar Cascade — no heavy dependencies.
    Returns cropped face as BGR numpy array, or None if no face found.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Load Haar cascade
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) == 0:
        return None, None

    # Take the largest detected face
    largest = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = largest

    # Add 10% padding around face for better embedding quality
    pad = int(0.10 * min(w, h))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(img_bgr.shape[1], x + w + pad)
    y2 = min(img_bgr.shape[0], y + h + pad)

    face_crop = img_bgr[y1:y2, x1:x2]
    bbox = (x1, y1, x2 - x1, y2 - y1)
    return face_crop, bbox


def extract_pan_photo_region(img_bgr):
    """
    On a standard PAN card the photo sits in the top-right quadrant.
    Try face detection on the full image first, then fall back to
    cropping the known photo region if no face is detected.
    """
    # First try full image detection
    face, bbox = detect_face(img_bgr)
    if face is not None:
        return face, bbox

    # Fallback: crop the standard PAN photo region
    # PAN card photo is roughly top-right: x=65%-95%, y=10%-55%
    h, w = img_bgr.shape[:2]
    x1 = int(w * 0.65)
    y1 = int(h * 0.08)
    x2 = int(w * 0.95)
    y2 = int(h * 0.58)
    region = img_bgr[y1:y2, x1:x2]

    # Try face detection on this region
    face, bbox = detect_face(region)
    if face is not None:
        return face, bbox

    # Return the raw region crop as last resort
    return region, (x1, y1, x2 - x1, y2 - y1)


# --------------------------------------------------------------------------- #
#  Face Embedding
# --------------------------------------------------------------------------- #

def _get_embedding_deepface(face_bgr):
    """
    Extract face embedding using DeepFace (FaceNet backend).
    Returns a numpy array or None on failure.
    """
    try:
        from deepface import DeepFace
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        result = DeepFace.represent(
            img_path      = rgb,
            model_name    = 'Facenet',
            enforce_detection = False,
            detector_backend  = 'skip'
        )
        if result and isinstance(result, list):
            return np.array(result[0]['embedding'])
        return None
    except Exception:
        return None


def _get_embedding_opencv(face_bgr):
    """
    Fallback embedding using raw pixel features when DeepFace unavailable.
    Resizes face to 64x64, flattens and normalises.
    Less accurate but always available.
    """
    resized = cv2.resize(face_bgr, (64, 64))
    gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    flat    = gray.flatten().astype(np.float32)
    norm    = flat / (np.linalg.norm(flat) + 1e-6)
    return norm


def get_face_embedding(face_bgr):
    """
    Try DeepFace first, fall back to OpenCV pixel embedding.
    Returns (embedding_array, method_used).
    """
    emb = _get_embedding_deepface(face_bgr)
    if emb is not None:
        return emb, 'deepface_facenet'

    emb = _get_embedding_opencv(face_bgr)
    return emb, 'opencv_pixel'


# --------------------------------------------------------------------------- #
#  Distance → Similarity Score
# --------------------------------------------------------------------------- #

def euclidean_distance(emb1, emb2):
    """
    Paper equation (6):
    Distance = ||phi(Fcapture) - phi(Ftemplate)||_2
    """
    if emb1 is None or emb2 is None:
        return float('inf')

    # Align lengths if using different backends
    min_len = min(len(emb1), len(emb2))
    e1 = emb1[:min_len]
    e2 = emb2[:min_len]

    return float(np.linalg.norm(e1 - e2))


def distance_to_similarity(distance, method='deepface_facenet'):
    """
    Convert Euclidean distance to a similarity score in [0, 1].
    Thresholds differ by embedding method.
    """
    if method == 'deepface_facenet':
        # FaceNet: typical same-person distance < 10, different > 15
        max_dist = 20.0
    else:
        # OpenCV pixel: distances are in [0, sqrt(4096)] ≈ 64
        max_dist = 64.0

    if distance == float('inf'):
        return 0.0

    similarity = max(0.0, 1.0 - (distance / max_dist))
    return round(similarity, 4)


# --------------------------------------------------------------------------- #
#  Master face match function
# --------------------------------------------------------------------------- #

def run_face_match(pan_img_bgr, selfie_img_bgr):
    """
    Full face matching pipeline.

    Parameters
    ----------
    pan_img_bgr    : BGR numpy array of PAN card
    selfie_img_bgr : BGR numpy array of selfie

    Returns
    -------
    dict:
        face_match_score   : float 0–1 (1 = same person)
        distance           : raw Euclidean distance
        match_verdict      : 'MATCH' | 'NO MATCH' | 'UNCERTAIN'
        pan_face_detected  : bool
        selfie_face_detected: bool
        embedding_method   : str
        pan_face_bbox      : bounding box tuple or None
    """

    # Step 1: Extract faces
    pan_face,    pan_bbox    = extract_pan_photo_region(pan_img_bgr)
    selfie_face, selfie_bbox = detect_face(selfie_img_bgr)

    pan_detected    = pan_face is not None and pan_face.size > 0
    selfie_detected = selfie_face is not None and selfie_face.size > 0

    if not pan_detected or not selfie_detected:
        return {
            'face_match_score':    0.0,
            'distance':            float('inf'),
            'match_verdict':       'FACE NOT DETECTED',
            'pan_face_detected':   pan_detected,
            'selfie_face_detected':selfie_detected,
            'embedding_method':    'none',
            'pan_face_bbox':       pan_bbox,
        }

    # Step 2: Get embeddings
    emb_pan,    method = get_face_embedding(pan_face)
    emb_selfie, _      = get_face_embedding(selfie_face)

    # Step 3: Compute distance (paper equation 6)
    distance   = euclidean_distance(emb_pan, emb_selfie)
    similarity = distance_to_similarity(distance, method)

    # Step 4: Verdict
    if method == 'deepface_facenet':
        threshold = FACE_DISTANCE_THRESHOLD * 20  # scale to FaceNet space
    else:
        threshold = FACE_DISTANCE_THRESHOLD * 64  # scale to pixel space

    if distance < threshold * 0.8:
        verdict = 'MATCH'
    elif distance < threshold * 1.2:
        verdict = 'UNCERTAIN'
    else:
        verdict = 'NO MATCH'

    return {
        'face_match_score':     similarity,
        'distance':             round(distance, 4),
        'match_verdict':        verdict,
        'pan_face_detected':    pan_detected,
        'selfie_face_detected': selfie_detected,
        'embedding_method':     method,
        'pan_face_bbox':        pan_bbox,
    }