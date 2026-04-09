import cv2
import numpy as np
from PIL import Image
import io


def load_image_from_field(image_field):
    """Load a Django ImageField into a numpy array (BGR, OpenCV format)."""
    image_field.open()
    img_bytes = image_field.read()
    image_field.close()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img


def load_image_from_upload(uploaded_file):
    """Load an InMemoryUploadedFile into a numpy array."""
    img_bytes = uploaded_file.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img


def preprocess_image(img, target_size=(400, 250)):
    """
    Canonical preprocessing for all document images.
    Resize → grayscale → Gaussian blur → adaptive threshold.
    Returns both grayscale and the original resized color image.
    """
    resized   = cv2.resize(img, target_size)
    gray      = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred   = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh    = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return resized, gray, blurred, thresh


def normalize_score(value, min_val=0.0, max_val=1.0):
    """Clamp any float to [0, 1]."""
    return max(min_val, min(max_val, float(value)))


def weighted_score(feature_scores: dict, weights: dict) -> float:
    """
    Compute weighted average of feature scores.
    Implements paper equation: Sdoc = sum(wi * fi)
    Both dicts must have matching keys.
    """
    total = 0.0
    weight_sum = 0.0
    for key, weight in weights.items():
        score = feature_scores.get(key)
        if score is not None:
            total      += weight * score
            weight_sum += weight
    if weight_sum == 0:
        return 0.0
    return normalize_score(total / weight_sum)


def draw_diff_regions(original_img, contours, color=(0, 0, 255), thickness=2):
    """Draw bounding boxes around suspicious regions on a copy of the image."""
    annotated = original_img.copy()
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # filter noise
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
    return annotated


def image_to_bytes(img, ext='.jpg'):
    """Convert numpy image back to bytes for saving."""
    success, buffer = cv2.imencode(ext, img)
    if success:
        return buffer.tobytes()
    return None