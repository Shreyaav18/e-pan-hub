"""
Behavioral Biometrics Service
-------------------------------
Implements paper Section V.C.2 — Behavioral Biometrics.

Captures interaction signals during onboarding:
  - Keystroke timing dynamics
  - Mouse movement velocity and trajectory
  - Session interaction patterns

Uses Isolation Forest to detect anomalous behavior profiles.
Returns Sbehavior in [0, 1] where 1 = normal behavior.
"""

import numpy as np
import json
from sklearn.ensemble import IsolationForest


# --------------------------------------------------------------------------- #
#  Feature extraction from raw behavioral signals
# --------------------------------------------------------------------------- #

def extract_keystroke_features(keystroke_data):
    """
    keystroke_data: list of dicts with keys:
        key, press_time (ms), release_time (ms)

    Returns feature vector:
        - mean dwell time (key held duration)
        - std dwell time
        - mean flight time (gap between keys)
        - std flight time
        - typing speed (keys per second)
    """
    if not keystroke_data or len(keystroke_data) < 3:
        return [0.0] * 5

    dwell_times  = []
    flight_times = []

    for i, k in enumerate(keystroke_data):
        press   = k.get('press_time', 0)
        release = k.get('release_time', press)
        dwell_times.append(release - press)

        if i > 0:
            prev_release = keystroke_data[i - 1].get('release_time', 0)
            flight_times.append(press - prev_release)

    total_time = (keystroke_data[-1].get('release_time', 0) -
                  keystroke_data[0].get('press_time', 0))
    speed = len(keystroke_data) / (total_time / 1000.0 + 1e-6)  # keys/sec

    return [
        np.mean(dwell_times),
        np.std(dwell_times),
        np.mean(flight_times) if flight_times else 0.0,
        np.std(flight_times)  if flight_times else 0.0,
        speed,
    ]


def extract_mouse_features(mouse_data):
    """
    mouse_data: list of dicts with keys:
        x, y, timestamp (ms)

    Returns feature vector:
        - mean velocity
        - std velocity
        - mean acceleration
        - direction change count (curviness)
        - total distance
    """
    if not mouse_data or len(mouse_data) < 3:
        return [0.0] * 5

    velocities    = []
    accelerations = []
    direction_changes = 0
    total_distance = 0.0
    prev_vx, prev_vy = 0.0, 0.0

    for i in range(1, len(mouse_data)):
        dx = mouse_data[i]['x'] - mouse_data[i - 1]['x']
        dy = mouse_data[i]['y'] - mouse_data[i - 1]['y']
        dt = (mouse_data[i]['timestamp'] - mouse_data[i - 1]['timestamp']) / 1000.0 + 1e-6

        dist     = np.sqrt(dx ** 2 + dy ** 2)
        velocity = dist / dt
        velocities.append(velocity)
        total_distance += dist

        if i > 1:
            vx, vy = dx / dt, dy / dt
            dot = prev_vx * vx + prev_vy * vy
            if dot < 0:
                direction_changes += 1
            accel = abs(velocity - velocities[-2]) / dt if len(velocities) > 1 else 0
            accelerations.append(accel)
            prev_vx, prev_vy = vx, vy

    return [
        np.mean(velocities),
        np.std(velocities),
        np.mean(accelerations) if accelerations else 0.0,
        float(direction_changes),
        total_distance,
    ]


def extract_session_features(session_data):
    """
    session_data: dict with keys:
        total_time_ms, scroll_events, click_count,
        focus_changes, idle_periods

    Returns feature vector of 5 values.
    """
    if not session_data:
        return [0.0] * 5

    return [
        session_data.get('total_time_ms', 0) / 1000.0,
        session_data.get('scroll_events', 0),
        session_data.get('click_count', 0),
        session_data.get('focus_changes', 0),
        session_data.get('idle_periods', 0),
    ]


# --------------------------------------------------------------------------- #
#  Isolation Forest anomaly scoring
# --------------------------------------------------------------------------- #

def _build_normal_population():
    """
    Generate synthetic 'normal' behavior population for Isolation Forest
    training in absence of historical real data.

    In production this would be replaced by real historical user data.
    """
    np.random.seed(42)
    n_samples = 300

    # Normal keystroke features: dwell 80–150ms, flight 50–120ms, speed 3–6 k/s
    keystroke_pop = np.column_stack([
        np.random.normal(110, 20, n_samples),   # mean dwell
        np.random.normal(25,  10, n_samples),   # std dwell
        np.random.normal(80,  20, n_samples),   # mean flight
        np.random.normal(20,  8,  n_samples),   # std flight
        np.random.normal(4.5, 0.8, n_samples),  # speed
    ])

    # Normal mouse features
    mouse_pop = np.column_stack([
        np.random.normal(300, 80,  n_samples),  # mean velocity px/s
        np.random.normal(120, 40,  n_samples),  # std velocity
        np.random.normal(50,  20,  n_samples),  # mean acceleration
        np.random.normal(8,   3,   n_samples),  # direction changes
        np.random.normal(800, 200, n_samples),  # total distance
    ])

    # Normal session features
    session_pop = np.column_stack([
        np.random.normal(45,  15, n_samples),   # time seconds
        np.random.normal(5,   3,  n_samples),   # scroll events
        np.random.normal(8,   3,  n_samples),   # clicks
        np.random.normal(2,   1,  n_samples),   # focus changes
        np.random.normal(1,   1,  n_samples),   # idle periods
    ])

    return np.hstack([keystroke_pop, mouse_pop, session_pop])


def run_behavioral_analysis(keystroke_data=None, mouse_data=None, session_data=None):
    """
    Full behavioral biometrics pipeline.

    Parameters
    ----------
    keystroke_data : list of keystroke dicts (from JS frontend)
    mouse_data     : list of mouse event dicts (from JS frontend)
    session_data   : dict of session-level metrics

    Returns
    -------
    dict:
        behavioral_score    : float 0–1 (1 = normal behavior)
        anomaly_score_raw   : raw Isolation Forest score
        feature_vector      : the 15-dim feature vector used
        behavioral_verdict  : 'NORMAL' | 'SUSPICIOUS' | 'ANOMALOUS'
    """

    # Extract feature vectors
    kf = extract_keystroke_features(keystroke_data or [])
    mf = extract_mouse_features(mouse_data or [])
    sf = extract_session_features(session_data or {})

    feature_vector = kf + mf + sf   # 15-dimensional

    # If no behavioral data provided, return neutral score
    if all(v == 0.0 for v in feature_vector):
        return {
            'behavioral_score':   0.7,   # neutral — no data
            'anomaly_score_raw':  0.0,
            'feature_vector':     feature_vector,
            'behavioral_verdict': 'NO DATA',
        }

    # Train Isolation Forest on synthetic normal population
    population  = _build_normal_population()
    clf = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42
    )
    clf.fit(population)

    # Score the new sample
    sample      = np.array(feature_vector).reshape(1, -1)
    raw_score   = clf.decision_function(sample)[0]  # negative = anomaly
    # decision_function returns: more negative = more anomalous
    # Typically in range [-0.5, 0.5]

    # Convert to [0, 1] where 1 = normal
    # raw_score: -0.5 → 0.0, 0.0 → 0.5, 0.5 → 1.0
    behavioral_score = float(min(1.0, max(0.0, raw_score + 0.5)))
    behavioral_score = round(behavioral_score, 4)

    # Verdict
    if behavioral_score >= 0.6:
        verdict = 'NORMAL'
    elif behavioral_score >= 0.4:
        verdict = 'SUSPICIOUS'
    else:
        verdict = 'ANOMALOUS'

    return {
        'behavioral_score':   behavioral_score,
        'anomaly_score_raw':  round(float(raw_score), 4),
        'feature_vector':     feature_vector,
        'behavioral_verdict': verdict,
    }   