# PAN Card Verification & AML Monitoring System
## Comprehensive Project Documentation

---

## TABLE OF CONTENTS
1. Project Overview
2. System Architecture
3. Module 1: Document Verification
4. Module 2: Biometric Authentication
5. Module 3: AML Transaction Monitoring
6. Database Schema
7. Frontend Integration Guide

---

## 1. PROJECT OVERVIEW

### What is This Project?
This is an **end-to-end KYC (Know Your Customer) and AML (Anti-Money Laundering) verification system** designed for Indian users using **PAN (Permanent Account Number) cards**. It's built with Django backend and integrates multiple AI/ML algorithms for document verification, biometric authentication, and transaction monitoring.

### Key Features
- **Automated PAN Document Verification** with forgery detection
- **Face Recognition** between PAN card photo and live selfie
- **Liveness Detection** to prevent spoof attacks (photos/videos/masks)
- **Behavioral Biometrics** to detect unusual interaction patterns
- **Real-time Transaction Monitoring** for money laundering detection
- **Multi-stage Risk Scoring** with automatic approval/rejection

### Tech Stack
```
Backend:    Django 4.2+, Django REST Framework
CV/ML:      OpenCV, scikit-learn, DeepFace, Keras
OCR:        EasyOCR (with Tesseract fallback)
Database:   Django ORM (SQLite/PostgreSQL)
```

### Project Structure
```
├── core/
│   ├── models.py           (VerificationCase storage)
│   ├── utils.py            (image loading, preprocessing)
│   └── admin.py
├── document_verification/
│   ├── services/
│   │   ├── ocr_service.py           (text extraction & validation)
│   │   ├── authenticity_service.py  (6 feature checks for forgery)
│   │   └── decision_service.py      (orchestration)
│   └── views.py            (upload endpoint)
├── biometric_auth/
│   ├── services/
│   │   ├── face_match_service.py    (face recognition)
│   │   ├── liveness_service.py      (spoof detection)
│   │   ├── behavioral_service.py    (keystroke/mouse analysis)
│   │   └── biometric_decision_service.py (orchestration)
├── aml_monitoring/
│   ├── services/
│   │   ├── rule_engine.py        (10 FATF/RBI compliance rules)
│   │   ├── isolation_forest.py   (statistical anomaly detection)
│   │   ├── lstm_monitor.py       (sequential pattern detection)
│   │   └── aml_decision_service.py (orchestration)
└── requirements.txt
```

---

## 2. SYSTEM ARCHITECTURE

### Workflow Overview

```
User Upload
    ↓
Module 1: Document Verification
    ├─ OCR Text Extraction
    ├─ 6-Feature Authenticity Check
    └─ Doc Score (0-1)
    ↓ (if passed)
Module 2: Biometric Verification
    ├─ Face Matching (PAN photo vs Selfie)
    ├─ Liveness Detection
    ├─ Behavioral Biometrics
    └─ Biometric Score (0-1)
    ↓ (if passed)
Module 3: AML Monitoring (Optional)
    ├─ Rule Engine (10 FATF/RBI rules)
    ├─ Isolation Forest (anomaly detection)
    ├─ LSTM Sequential Monitoring
    └─ AML Score (0-1)
    ↓
Final Decision + Risk Scoring
    └─ APPROVED / MANUAL REVIEW / REJECTED
```

### Database Model (VerificationCase)

All results stored in a single `VerificationCase` record:

```python
case_id                    # UUID primary key
full_name                  # User's name
pan_card_image            # Uploaded PAN file
selfie_image              # Uploaded selfie file
created_at / updated_at   # Timestamps

# Module 1 - Document Verification Scores
doc_ocr_valid             # OCR validation (0-1)
doc_font_consistency      # Font analysis (0-1)
doc_edge_consistency      # Edge pattern check (0-1)
doc_color_histogram       # Color profile analysis (0-1)
doc_structural_align      # Layout integrity (0-1)
doc_metadata_clean        # EXIF inspection (0-1)
doc_ssim_score            # Structural similarity (0-1)
doc_score                 # Final Document Score (0-1)
extracted_pan_number      # OCR result: PAN
extracted_name            # OCR result: Name
extracted_dob             # OCR result: DOB

# Module 2 - Biometric Scores
face_match_score          # Face matching (0-1)
liveness_score            # Liveness detection (0-1)
behavioral_score          # Behavior analysis (0-1)
biometric_score           # Final Biometric Score (0-1)

# Module 3 - AML Scores
rule_flags                # List of triggered rules
isolation_forest_score    # Anomaly score (0-1)
lstm_anomaly_score        # Sequential anomaly (0-1)
aml_score                 # Final AML Score (0-1)

# Final Decision
risk_total                # Composite risk score
penalty_flags             # Hard failures
status                    # pending/processing/approved/review/rejected
decision_reason           # Why approved/rejected
```

---

## 3. MODULE 1: DOCUMENT VERIFICATION

### Purpose
Verify that the uploaded PAN card image is:
1. **Authentic** (not forged/manipulated)
2. **Readable** (OCR extraction works)
3. **Valid** (fields match expected formats)

### 3.1 OCR Text Extraction (ocr_service.py)

#### Step 1: Image Preprocessing
```
Input Image (BGR)
    ↓
Upscale to 800px (if smaller)
    ↓
Denoise (fastNlMeansDenoising)
    ↓
CLAHE (Contrast enhancement for faded cards)
    ↓
Otsu Binarization (convert to B&W)
    ↓
Output: High-contrast binary image
```

#### Step 2: OCR Extraction
- Tries **EasyOCR** (primary - highest accuracy)
- Falls back to **Tesseract** if EasyOCR unavailable
- Converts output to uppercase for standardization

#### Step 3: Field Extraction

**PAN Number** (AAAAA9999A)
- Regex: `[A-Z]{5}[0-9]{4}[A-Z]`
- Score: 1.0 (perfect), 0.5 (right length, wrong pattern), 0.0 (missing)

**Name** (ALL CAPS)
- Heuristic: Longest all-caps sequence not matching noise words
- Noise filter: "INCOME TAX DEPARTMENT", "GOVT OF INDIA", etc.
- Score: 1.0 (valid), 0.5 (partial), 0.0 (missing)

**Date of Birth** (DD/MM/YYYY or DD-MM-YYYY)
- Regex: `\d{2}[/-]\d{2}[/-]\d{4}`
- Validates day (1-31), month (1-12), year (1900-2025)
- Score: 1.0 (plausible), 0.0 (impossible date)

#### Step 4: OCR Score Calculation

```
OCR Score = (PAN_score × 0.50) + (DOB_score × 0.25) + (Name_score × 0.25)

Example:
- PAN valid (1.0) × 0.50 = 0.50
- DOB valid (1.0) × 0.25 = 0.25
- Name valid (1.0) × 0.25 = 0.25
────────────────────────────────
OCR Score = 1.00 (perfect)
```

### 3.2 Authenticity Scoring (authenticity_service.py)

Six parallel image analysis checks to detect forgery:

#### Feature 1: Font Consistency (15% weight)
**What it detects:** Mixed fonts from different sources (common forgery)

**Method:**
1. Divide image into 10 horizontal strips
2. Calculate Laplacian variance for each strip (measures texture/sharpness)
3. Compute coefficient of variation (CV) across strips
4. CV=0 → all strips identical → score 1.0
5. CV>2.0 → high variation → score 0.0

**Why it works:** Real PAN cards have uniform typography. Forged cards cobble together text from different sources = inconsistent texture.

#### Feature 2: Edge Consistency (20% weight) ⭐ MOST SENSITIVE
**What it detects:** Copy-paste, clone stamping, or edited regions

**Method:**
1. Apply Canny edge detection (find all sharp transitions)
2. Divide into 8×8 blocks
3. Calculate edge density in each block
4. Measure variance: High variance = inconsistent patterns
5. Normal face: low variance → score 1.0
6. Tampered: high variance → score 0.0

**Why it works:** Real documents have uniform edge patterns. Pasted regions create edge anomalies.

#### Feature 3: Color Histogram (15% weight)
**What it detects:** Color inconsistencies from pasting different source materials

**Method:**
1. Split image into 4 quadrants (top-left, top-right, bottom-left, bottom-right)
2. Calculate 8×8×8 RGB histogram for each
3. Compare all pairwise combinations using **Bhattacharyya distance**
4. Low distance = all regions same color = score 1.0
5. High distance = color mismatch = score 0.0

**Why it works:** PAN cards have consistent cream background and standard colors. Pasted regions often have different color casts.

#### Feature 4: Structural Alignment (10% weight)
**What it detects:** Layout inconsistencies (logo position, text zones)

**Method:**
1. Apply Harris corner detection (finds structural keypoints)
2. Divide image into 3×3 grid
3. Count cells containing corners
4. Score = corners_count / 9
5. Genuine card: corners in 8-9 cells → score ~0.8-1.0
6. Tampered: corners missing in some areas → score <0.8

**Why it works:** Real PAN cards have fixed layout. Tampering disrupts structural integrity.

#### Feature 5: Metadata Analysis (5% weight)
**What it detects:** Digital editing software traces (Photoshop, GIMP, etc.)

**Method:**
1. Extract EXIF metadata from JPEG
2. Check Software field for editing tools: Photoshop, GIMP, Lightroom, etc.
3. Check for modification timestamp ≠ creation timestamp
4. Score: 1.0 (clean), 0.4 (modified after creation), 0.1 (editing software found)

**Why it works:** Scanned originals don't have EXIF. Digital edits leave software signatures.

#### Feature 6: SSIM - Structural Similarity (5% weight)
**What it detects:** Internal structural consistency

**Method:**
```
If reference image provided:
    SSIM = compare_images(test_image, reference_image)

Else (no reference, use self-similarity):
    left_half = split image at middle
    right_half = split image at middle
    SSIM = compare(left_half, right_half)
    # Genuine cards have symmetric background texture
    # Scale: -1 to 1 → normalize to 0 to 1
```

**Why it works:** Genuine documents have internal symmetry/consistency.

#### Final Document Score Calculation

```
FEATURE_WEIGHTS = {
    'ocr_valid':        0.30,   (most critical—content validation)
    'edge_consistency': 0.20,   (tamper detection)
    'font_consistency': 0.15,   (font mixing detection)
    'color_histogram':  0.15,   (color consistency)
    'structural_align': 0.10,   (layout integrity)
    'metadata_clean':   0.05,   (EXIF analysis)
    'ssim':             0.05,   (structural similarity)
}

Sdoc = Σ (weight_i × feature_i)

Example:
  OCR (0.30) × 0.95 = 0.285
  Edge (0.20) × 0.88 = 0.176
  Font (0.15) × 0.92 = 0.138
  Color (0.15) × 0.85 = 0.127
  Struct (0.10) × 0.90 = 0.090
  Metadata (0.05) × 1.0 = 0.050
  SSIM (0.05) × 0.78 = 0.039
  ───────────────────────────
  Sdoc = 0.905 (Very Authentic)
```

#### Verdict Logic (from Module 1):
```
Sdoc >= 0.75  → AUTHENTIC (green)
Sdoc 0.50-0.74 → SUSPICIOUS (yellow) - Manual Review
Sdoc < 0.50  → LIKELY FORGED (red) - Auto-Reject
```

---

## 4. MODULE 2: BIOMETRIC AUTHENTICATION

### Purpose
Verify that the person presenting the selfie is the same person in the PAN card photo AND is alive (not a spoofed image).

### 4.1 Face Matching (face_match_service.py)

#### Step 1: Extract Faces

**From PAN Card:**
1. Try full image face detection (Haar Cascade)
2. If fails, crop standard PAN photo region (top-right: 65%-95% width, 8%-58% height)
3. Detect face in cropped region
4. Add 10% padding around detected face

**From Selfie:**
1. Detect largest face in live selfie
2. Take that face region

#### Step 2: Generate Face Embeddings

**Method 1: DeepFace with FaceNet** (preferred)
- Uses state-of-the-art deep CNN trained on millions of faces
- Returns 128-dimensional embedding vector
- Distance < 10 = same person, Distance > 15 = different person

**Method 2: OpenCV Fallback** (if DeepFace unavailable)
- Resize face to 64×64 pixels
- Convert to grayscale
- Flatten and normalize
- Returns 4096-dimensional vector
- Less accurate but always available

#### Step 3: Face Matching

```
Paper Equation (6):
Distance = ||φ(Fcapture) - φ(Ftemplate)||₂

Where:
  Fcapture = embedding from selfie
  Ftemplate = embedding from PAN card photo
  φ() = embedding function (FaceNet or OpenCV)
  ||·||₂ = Euclidean L2 norm distance
```

#### Step 4: Convert Distance to Similarity Score

```
Method 1 (DeepFace):
  max_distance = 20.0
  similarity = max(0, 1 - distance/20)

Method 2 (OpenCV):
  max_distance = 64.0
  similarity = max(0, 1 - distance/64)
```

#### Verdict Logic:

```
THRESHOLD = 0.6 (internal distance in normalized space)

if distance < threshold × 0.8:
    verdict = "MATCH" (0.9+ similarity)
elif distance < threshold × 1.2:
    verdict = "UNCERTAIN" (0.5-0.9 similarity)
else:
    verdict = "NO MATCH" (< 0.5 similarity)

If faces not detected:
    verdict = "FACE_NOT_DETECTED" (auto-fail)
```

### 4.2 Liveness Detection (liveness_service.py)

**Purpose:** Detect if the selfie is a real live face or a spoof (printed photo, phone screen, 3D mask).

**Four Independent Checks (combined with weights):**

#### Method 1: LBP Texture Analysis (35% weight)

**What detects:** Living skin vs. flat printed media

**How it works:**
1. Local Binary Patterns (LBP) = compare each pixel with 8 neighbors
2. Real skin has natural micro-texture variation
3. Printed photos have regular, periodic print patterns
4. Screen displays have regular pixel grid artifacts

**Scoring:**
```
local_std = standard deviation of pixel differences

IF local_std < 3:
    score = 0.1      (too flat = printed photo)
ELSE IF local_std > 35:
    score = 0.3      (too noisy = screen/moiré)
ELSE:
    score = 0.3 + (local_std - 3) / (35 - 3) × 0.7
    # Maps 3-35 range to 0.3-1.0
```

#### Method 2: Frequency Domain (FFT) Analysis (25% weight)

**What detects:** Moiré patterns and periodic artifacts

**How it works:**
1. Apply 2D FFT (Fast Fourier Transform) to convert spatial image to frequency domain
2. Real faces have **distributed** frequency energy
3. Printed photos have **concentrated** energy at specific frequencies (halftone patterns)
4. Phone screens have energy spikes at refresh frequencies (50/60 Hz)

**Scoring:**
```
inner_energy = low frequencies (faces, broad features)
outer_energy = high frequencies (details, patterns)

ratio = outer_energy / inner_energy

IF 0.25 <= ratio <= 0.9:
    score = 0.85     (natural balance)
ELSE IF ratio < 0.25:
    score = 0.5      (suspiciously flat)
ELSE:
    score = max(0, 1 - (ratio - 0.9) / 2)  (too much print pattern)
```

#### Method 3: Gradient Statistics (25% weight)

**What detects:** Quantization artifacts and grid patterns

**How it works:**
1. Calculate gradient magnitude (Sobel filters) = rate of pixel value change
2. Real skin = smooth gradients
3. Printed/screen = quantized/discrete step changes

**Scoring:**
```
mean_grad = average gradient magnitude
std_grad = standard deviation of gradients
cv = std_grad / mean_grad  (coefficient of variation)

IF mean_grad < 5:
    score = 0.1      (completely flat)
ELSE IF cv < 0.3:
    score = 0.3      (too uniform = likely print)
ELSE IF cv > 2.5:
    score = 0.4      (too noisy)
ELSE:
    score = min(1.0, 0.5 + cv × 0.2)
```

#### Method 4: Specular Highlights (15% weight)

**What detects:** Lack of natural light reflections

**How it works:**
1. Find bright pixels (intensity > 230/255)
2. Real faces have natural specular highlights (eye shine, skin reflection)
3. Printed photos/screens are mostly flat

**Scoring:**
```
bright_ratio = % of very bright pixels

IF 0.3% <= bright_ratio <= 8%:
    score = 0.85     (natural amount of highlights)
ELSE IF bright_ratio < 0.1%:
    score = 0.4      (no highlights = flat = printed)
ELSE IF bright_ratio > 20%:
    score = 0.3      (over-exposed or screen glare)
ELSE:
    score = 0.6      (borderline)
```

#### Final Liveness Score

```
LIVENESS_WEIGHTS = {
    'texture':   0.35,
    'frequency': 0.25,
    'gradient':  0.25,
    'specular':  0.15,
}

liveness = Σ (weight_i × method_i)

Example:
  Texture (0.35) × 0.88 = 0.308
  Freq (0.25) × 0.92 = 0.230
  Grad (0.25) × 0.85 = 0.212
  Specular (0.15) × 0.90 = 0.135
  ─────────────────────────
  liveness = 0.885 (Definitely Live)

Verdict:
  liveness >= 0.65 → LIVE (accept)
  liveness 0.45-0.64 → UNCERTAIN (review)
  liveness < 0.45 → SPOOF (reject)
```

### 4.3 Behavioral Biometrics (behavioral_service.py)

**Purpose:** Detect if the user is the enrolled person by tracking interaction patterns.

**Captures 3 Types of Signals:**

#### 1. Keystroke Dynamics
```
Extracted Features:
  - Mean dwell time (how long key held): ~80-150ms
  - Std dwell time: ~10-25ms
  - Mean flight time (gap between keys): ~50-120ms
  - Std flight time: ~10-20ms
  - Typing speed: ~3-6 keys/second

Normal Range:
  If features within expected distribution → score high
  Else (too fast, too slow, irregular) → score low
```

#### 2. Mouse Movement Dynamics
```
Extracted Features:
  - Mean velocity: ~300 px/s
  - Std velocity: ~120 px/s
  - Mean acceleration
  - Direction changes (curviness)
  - Total distance traveled

Normal Behavior:
  Smooth, curved trajectories = legitimate user
  Jittery, jerky movement = bot/spoofed
```

#### 3. Session-Level Metrics
```
Metrics:
  - Total session time
  - Scroll events count
  - Click count
  - Focus change events
  - Idle periods count

Normal:
  Natural interaction pattern (scrolling, clicking)
  Suspicious:
  Too fast (automated), too many idles (unnatural)
```

#### Anomaly Detection: Isolation Forest

```
Training:
  Build synthetic "normal" user population (300 samples)
  Train Isolation Forest (100 trees, 5% contamination)

Scoring:
  User's 15-dim feature vector → decision_function → [-0.5, 0.5]

  Negative score = anomalous (unusual behavior)
  Positive score = normal behavior

Conversion:
  behavioral_score = min(1.0, max(0.0, raw_score + 0.5))

  Example:
    raw_score = -0.3 → behavioral_score = 0.2 (suspicious)
    raw_score = 0.1 → behavioral_score = 0.6 (somewhat normal)

Verdict:
  behavioral_score >= 0.6 → NORMAL
  behavioral_score 0.4-0.59 → SUSPICIOUS
  behavioral_score < 0.4 → ANOMALOUS
```

### 4.4 Final Biometric Score

```
BIOMETRIC_WEIGHTS = {
    'face_match': 0.50,   (is it the same person?)
    'liveness':   0.35,   (is the face alive?)
    'behavioral': 0.15,   (is interaction natural?)
}

Sbiometric = (0.50 × face_score) + (0.35 × liveness) + (0.15 × behavior)

HARD RULES:
  IF liveness=SPOOF → Sbiometric capped at 0.3
  IF face_match=NO_MATCH → Auto-reject
  IF face_not_detected → Auto-reject
```

---

## 5. MODULE 3: AML TRANSACTION MONITORING

### Purpose
Monitor ongoing transactions for money laundering patterns per FATF (Financial Action Task Force) and RBI (Reserve Bank of India) guidelines.

### 5.1 Rule-Based Engine (rule_engine.py)

**10 Compliance Rules implemented:**

#### Rule 1: Large Transaction (R1)
```
THRESHOLD = INR 50,000 (RBI CTR - Currency Transaction Report)

IF transaction_amount >= 50,000:
    FLAG = "R1_LARGE_TRANSACTION" (HIGH severity)
    Detail: Amount exceeds CTR threshold
```

#### Rule 2: Structuring / Smurfing (R2)
```
PATTERN: Multiple just-below-threshold transactions = "structuring"
Common ML technique to evade reporting

IF within 24 hours:
    count_of_transactions_in_range(40,000-49,000) >= 3
    AND total >= 50,000:

    FLAG = "R2_STRUCTURING" (HIGH severity)
    Detail: {N} transactions totalling {amount} in 24h
```

#### Rule 3: High Frequency (R3)
```
IF transactions_in_last_1_hour >= 10:
    FLAG = "R3_HIGH_FREQUENCY" (MEDIUM severity)
    Detail: {N} txns in last hour (unusual velocity)
```

#### Rule 4: High-Risk Jurisdiction (R4)
```
HIGH_RISK_COUNTRIES = {
    'IR' (Iran), 'KP' (N. Korea), 'MM' (Myanmar),
    'SY' (Syria), 'YE' (Yemen), 'PK' (Pakistan),
    'AF' (Afghanistan), ... (FATF grey/blacklist)
}

IF counterparty_country in HIGH_RISK:
    FLAG = "R4_HIGH_RISK_JURISDICTION" (HIGH severity)
```

#### Rule 5: Unusual Time (R5)
```
ODD_HOURS = 11pm to 5am

IF transaction_hour in ODD_HOURS:
    FLAG = "R5_UNUSUAL_TIME" (LOW severity)
    Detail: Transaction at {hour}:{minute}

Note: Sometimes legitimate (medical, emergency)
      but statistical flag
```

#### Rule 6: Round Number Amounts (R6)
```
SUSPICIOUS ROUNDS = 1000, 5000, 10000, 25000, 50000, 100000

IF abs(amount - round_base) / round_base < 1%:
    FLAG = "R6_ROUND_NUMBER" (LOW severity)
    Detail: Amount {amount} suspiciously round

Note: Money laundering often uses round numbers
      for easy tracking
```

#### Rule 7: Rapid Fund Movement (R7)
```
PATTERN: Funds in → quickly out = "layering"

IF this_txn is DEBIT and
   within last 2 hours there exists CREDIT of
   >= 80% of current amount:

    FLAG = "R7_RAPID_MOVEMENT" (HIGH severity)
    Detail: Funds credited then debited within 2h
```

#### Rule 8: New Account High Value (R8)
```
IF account_age <= 30 days AND
   transaction_amount >= 25,000:

    FLAG = "R8_NEW_ACCOUNT_HIGH_VALUE" (MEDIUM severity)
    Detail: Account {N} days old, high activity detected
```

#### Rule 9: Multiple Counterparties (R9)
```
IF unique_counterparties_in_24h >= 8:
    FLAG = "R9_MULTIPLE_COUNTERPARTIES" (MEDIUM severity)
    Detail: {N} unique counterparties in 24h window

Note: Indicates potential customer acquisition for laundering
```

#### Rule 10: Dormant Account Reactivation (R10)
```
IF account_dormant_for >= 180 days AND
   current_transaction >= 10,000:

    FLAG = "R10_DORMANT_ACCOUNT" (MEDIUM severity)
    Detail: Dormant {N} days, now showing activity
```

#### Rule Score Calculation

```
severity_counts = {
    'HIGH': count of HIGH severity flags,
    'MEDIUM': count of MEDIUM,
    'LOW': count of LOW
}

penalty = (HIGH × 0.30) + (MEDIUM × 0.15) + (LOW × 0.05)

rule_score = max(0.0, 1.0 - penalty)

Example:
  1 HIGH, 0 MEDIUM, 0 LOW:
    penalty = 0.30
    rule_score = 0.70 (MONITOR) ⚠️

  2 HIGH, 1 MEDIUM, 2 LOW:
    penalty = 0.60 + 0.15 + 0.10 = 0.85
    rule_score = 0.15 (SUSPICIOUS) 🚨
```

### 5.2 Isolation Forest (isolation_forest.py)

**Purpose:** Detect statistical anomalies that don't match hardcoded rules.

#### 15-Dimensional Feature Vector

```
f1   amount              (raw INR amount)
f2   log_amount          (log-scaled for skew)
f3   hour_of_day         (0-23)
f4   day_of_week         (0=Monday, 6=Sunday)
f5   is_weekend          (binary)
f6   is_odd_hour         (binary: 11pm-5am)
f7   txn_type_encoded    (0=credit, 1=debit, 2=transfer)
f8   counterparty_risk    (0=domestic, 1=high-risk country)
f9   txn_count_24h       (# transactions in last 24h)
f10  total_amount_24h    (rolling sum last 24h)
f11  avg_amount_7d       (baseline average over 7 days)
f12  amount_vs_avg       (current / 7d_average deviation)
f13  unique_counterparties_24h
f14  max_single_24h      (largest single txn in 24h)
f15  velocity_change     (ratio of 24h activity vs 7d baseline)
```

#### Paper Equation (8): Isolation Forest Anomaly Scoring

```
s(x, n) = 2^(-E(h(x)) / c(n))

Where:
  x = sample (transaction feature vector)
  n = training set size
  h(x) = average path length to isolate x
  c(n) = normalization factor

Score near 1.0 = definitely anomalous (isolated quickly)
Score near 0.5 = normal (requires many splits)
```

#### Scoring Process

```
1. Train Isolation Forest on synthetic "normal" population
   - 500 samples of normal transactions
   - 200 trees, 5% contamination assumption

2. For new transaction:
   - decision_function(x) → typically [-0.5, 0.5]
   - negative = anomalous, positive = normal

3. Convert to consistent 0-1 scale:
   anomaly_score = max(0, min(1, 0.5 - raw_score))

4. For UI: clean_score = 1.0 - anomaly_score
   (so 1.0 = safe, 0.0 = anomalous)

Example:
  raw_score = -0.3 → anomaly = 0.8 (very suspicious)
  raw_score = 0.1 → anomaly = 0.4 (somewhat normal)
```

### 5.3 LSTM Sequential Pattern Monitor (lstm_monitor.py)

**Purpose:** Detect AML patterns across transaction sequences (layering, smurfing).

#### Paper Equation (9): Sequential Anomaly

```
AnomalyScore_LSTM = ||x_actual - x_predicted||₂

Where:
  x_predicted = expected features based on history
  x_actual = observed transaction features
  ||·||₂ = Euclidean L2 distance
```

#### Implementation: Exponential Weighted Moving Average (EWMA)

```
Since full LSTM requires DL infrastructure, we use EWMA
as a lightweight statistical equivalent.

Algorithm:
1. Extract last 10 transactions (ordered by time)
2. For each: compute [log_amount, hour, txn_type, is_weekend, cp_risk]
3. Apply exponential weights (recent txns weight more)
4. Compute weighted average = predicted_features
5. Calculate L2 distance to current transaction
6. Map distance to score:
     distance < 0.5 → score < 0.3 (expected)
     distance 0.5-1.5 → score 0.3-0.6 (deviation)
     distance > 2.5 → score > 0.8 (break)
```

#### Pattern Detection

**Layering Pattern:**
```
Signature: Large credit → multiple debits to different parties
           within 48 hours

Example:
  T1: Deposit INR 100,000 ✓
  T2: Transfer to Party A: 30,000 ✓
  T3: Transfer to Party B: 40,000 ✓
  T4: Transfer to Party C: 25,000 🚨 LAYERING
```

**Smurfing Pattern:**
```
Signature: Many small transactions aggregate above
           threshold within 48 hours

Example:
  T1: Deposit 8,000 ✓
  T2: Deposit 9,000 ✓
  T3: Deposit 8,500 ✓
  T4: Deposit 9,200 ✓
  T5: Deposit 8,100 ✓
  Total: 42,800 in 48h 🚨 SMURFING
```

#### Sequential Score Calculation

```
lstm_anomaly = (
    (sequence_deviation × 0.50) +
    (layering_score × 0.30) +
    (smurfing_score × 0.20)
)

sequence_score = 1.0 - lstm_anomaly

Verdict:
  sequence_score >= 0.75 → CLEAN
  sequence_score 0.45-0.74 → MONITOR
  sequence_score < 0.45 → SUSPICIOUS_PATTERN
```

### 5.4 Final AML Score (aml_decision_service.py)

```
WEIGHTS = {
    'rule':    0.40,      (regulatory baseline)
    'iforest': 0.35,      (statistical anomaly)
    'lstm':    0.25,      (sequential patterns)
}

SAML = (0.40 × rule_score) +
       (0.35 × iforest_clean) +
       (0.25 × lstm_score)

HARD OVERRIDE:
  IF any HIGH severity rule triggered:
      SAML = min(SAML, 0.30)  (force review even if other scores high)

Verdict:
  SAML >= 0.70 → CLEAN (accept) ✓
  SAML 0.40-0.69 → MONITOR (flag for review) ⚠️
  SAML < 0.40 → SUSPICIOUS (likely reject) ✗
```

---

## 6. DATABASE SCHEMA (VerificationCase)

### Field Mappings

| DB Field | Type | Module | Description |
|----------|------|--------|-------------|
| case_id | UUID | Core | Unique case identifier |
| full_name | CharField | Input | User's full name |
| created_at | DateTime | Core | Case creation timestamp |
| updated_at | DateTime | Core | Last update timestamp |
| pan_card_image | ImageField | Module 1 | Uploaded PAN image |
| selfie_image | ImageField | Module 2 | Uploaded selfie |
| status | CharField | Core | pending/processing/approved/review/rejected |
| **Document Verification** | | | |
| doc_ocr_valid | Float | Module 1 | OCR field validation (0-1) |
| doc_font_consistency | Float | Module 1 | Font analysis score |
| doc_edge_consistency | Float | Module 1 | Edge anomaly detection |
| doc_color_histogram | Float | Module 1 | Color consistency |
| doc_structural_align | Float | Module 1 | Layout integrity |
| doc_metadata_clean | Float | Module 1 | EXIF analysis |
| doc_ssim_score | Float | Module 1 | Structural similarity |
| doc_score | Float | Module 1 | **Final Document Score (0-1)** |
| extracted_pan_number | CharField | Module 1 | OCR result: PAN |
| extracted_name | CharField | Module 1 | OCR result: Name |
| extracted_dob | CharField | Module 1 | OCR result: DOB |
| **Biometric Authentication** | | | |
| face_match_score | Float | Module 2 | Face matching (0-1) |
| liveness_score | Float | Module 2 | Liveness detection (0-1) |
| behavioral_score | Float | Module 2 | Behavior analysis (0-1) |
| biometric_score | Float | Module 2 | **Final Biometric Score (0-1)** |
| **AML Monitoring** | | | |
| rule_flags | JSONField | Module 3 | List of triggered rule codes |
| isolation_forest_score | Float | Module 3 | Statistical anomaly (0-1) |
| lstm_anomaly_score | Float | Module 3 | Sequential anomaly (0-1) |
| aml_score | Float | Module 3 | **Final AML Score (0-1)** |
| **Risk & Decision** | | | |
| risk_total | Float | Core | Composite risk score |
| penalty_flags | JSONField | Core | Hard failure flags |
| decision_reason | TextField | Core | Explanation for decision |

### Status Workflow

```
PENDING
  ↓ (user uploads PAN)
PROCESSING
  ├─ Run Module 1 (Document Verification)
  └─ If passed, run Module 2 (Biometric Auth)
    └─ If passed, run Module 3 (AML Monitoring)
  ↓
Choose Outcome:
├─ APPROVED (all modules >= 0.75)
├─ MANUAL_REVIEW (conflicting signals, needs human review)
└─ REJECTED (hard failures: spoof, forgery, rules violations)
```

---

## 7. FRONTEND INTEGRATION GUIDE

### 7.1 File Upload Endpoint

**Endpoint:** `POST /verify/upload/`

**Request:**
```json
{
    "full_name": "JOHN DOE",
    "pan_card": <binary file>,
    "reference_card": <binary file (optional)>,
    "selfie_image": <binary file>
}
```

**Response (on success):**
```json
{
    "case_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "processing",
    "doc_score": 0.88,
    "doc_verdict": "AUTHENTIC",
    "biometric_score": 0.92,
    "aml_score": 0.76,
    "final_status": "APPROVED"
}
```

### 7.2 Behavioral Data Capture (JavaScript)

Collect keystroke, mouse, and session metrics:

```javascript
// Keystroke tracking
const keystrokeData = [];
document.addEventListener('keydown', (e) => {
    keystrokeData.push({
        key: e.key,
        press_time: Date.now(),
        release_time: null
    });
});

document.addEventListener('keyup', (e) => {
    if (keystrokeData.length > 0) {
        keystrokeData[keystrokeData.length - 1].release_time = Date.now();
    }
});

// Mouse tracking
const mouseData = [];
document.addEventListener('mousemove', (e) => {
    mouseData.push({
        x: e.clientX,
        y: e.clientY,
        timestamp: Date.now()
    });
});

// Session metrics
const sessionData = {
    total_time_ms: Date.now() - startTime,
    scroll_events: scrollCount,
    click_count: clickCount,
    focus_changes: focusChangeCount,
    idle_periods: idleCount
};

// Send with form
form.addEventListener('submit', (e) => {
    document.getElementById('keystroke_data').value = JSON.stringify(keystrokeData);
    document.getElementById('mouse_data').value = JSON.stringify(mouseData);
    document.getElementById('session_data').value = JSON.stringify(sessionData);
});
```

### 7.3 AML Transaction Monitoring Endpoint

**Endpoint:** `POST /aml/monitor/`

**Request:**
```json
{
    "transaction": {
        "amount": 75000,
        "type": "transfer_out",
        "timestamp": "2025-04-07T14:30:00Z",
        "counterparty_id": "CUST_12345",
        "counterparty_country": "IN"
    },
    "history": [
        // previous transactions
    ],
    "account_age_days": 45,
    "days_since_last_txn": 2
}
```

**Response:**
```json
{
    "aml_score": 0.45,
    "verdict": "MONITOR",
    "rule_flags": ["R1_LARGE_TRANSACTION", "R5_UNUSUAL_TIME"],
    "isolation_forest_score": 0.38,
    "lstm_anomaly_score": 0.52,
    "recommended_action": "MANUAL_REVIEW"
}
```

### 7.4 Frontend Display Components

**Document Verification Results:**
```
┌─────────────────────────────────┐
│ Document Verification Results   │
├─────────────────────────────────┤
│ Overall Score: 88% AUTHENTIC ✓  │
├─────────────────────────────────┤
│ Feature Breakdown:              │
│  OCR Validation:    95% ✓       │
│  Font Consistency:  92% ✓       │
│  Edge Consistency:  88% ✓       │
│  Color Histogram:   85% ✓       │
│  Structural Align:  90% ✓       │
│  Metadata Clean:    100% ✓      │
│  SSIM Score:        78% ✓       │
└─────────────────────────────────┘
```

**Biometric Verification Results:**
```
┌─────────────────────────────────┐
│ Biometric Verification Results  │
├─────────────────────────────────┤
│ Overall Score: 92% MATCHED ✓    │
├─────────────────────────────────┤
│ Face Matching:                  │
│  Status: MATCH (similarity: 94%)│
│  Distance: 2.5 (threshold: 12) │
│                                 │
│ Liveness Detection:             │
│  Status: LIVE (score: 0.88)    │
│  Texture: 0.88 ✓               │
│  Frequency: 0.92 ✓             │
│  Gradient: 0.85 ✓              │
│  Specular: 0.90 ✓              │
│                                 │
│ Behavioral Biometrics:          │
│  Status: NORMAL (score: 0.72)  │
└─────────────────────────────────┘
```

**AML Monitoring Results:**
```
┌─────────────────────────────────┐
│ AML Transaction Monitoring      │
├─────────────────────────────────┤
│ Overall Score: 65% MONITOR ⚠️    │
├─────────────────────────────────┤
│ Triggered Rules:                │
│  🚨 R1_LARGE_TRANSACTION       │
│      Amount (75,000) > Threshold│
│      Severity: HIGH             │
│                                 │
│  🔔 R5_UNUSUAL_TIME             │
│      Transaction at 02:30 AM    │
│      Severity: LOW              │
│                                 │
│ Anomaly Detection:              │
│  Isolation Forest: 0.38 (normal)│
│  LSTM Pattern: 0.52 (deviation) │
│                                 │
│ Recommendation: MANUAL_REVIEW   │
└─────────────────────────────────┘
```

---

## 8. TECHNICAL SPECIFICATIONS

### Image Processing Requirements

**Supported Formats:** JPG, PNG, BMP, TIFF

**Minimum Dimensions:**
- PAN Card: 400×250 pixels
- Selfie: 480×480 pixels (for face detection)

**Preprocessing Pipeline:**
```
Input Image
    ↓
Resize to canonical dimensions (400×250 for PAN)
    ↓
Convert BGR (OpenCV) format
    ↓
Denoise (bilateral filter or NL-Means)
    ↓
Enhance contrast (CLAHE)
    ↓
Create multiple versions:
    ├─ Grayscale (for OCR, edge detection)
    ├─ Color (for histogram analysis)
    └─ Binary (adaptive threshold)
```

### Scoring Summary

| Module | Sub-Component | Weight | Threshold | Range |
|--------|---------------|--------|-----------|-------|
| **Module 1: Document** |
| | OCR Valid | 30% | - | 0-1 |
| | Font Consistency | 15% | - | 0-1 |
| | Edge Consistency | 20% | - | 0-1 |
| | Color Histogram | 15% | - | 0-1 |
| | Structural Alignment | 10% | - | 0-1 |
| | Metadata Analysis | 5% | - | 0-1 |
| | SSIM | 5% | - | 0-1 |
| | **DOC_SCORE** | **100%** | **0.75** | **0-1** |
| **Module 2: Biometric** |
| | Face Matching | 50% | Match | 0-1 |
| | Liveness Detection | 35% | 0.65 | 0-1 |
| | Behavioral | 15% | 0.60 | 0-1 |
| | **BIOMETRIC_SCORE** | **100%** | **0.75** | **0-1** |
| **Module 3: AML** |
| | Rule Engine | 40% | - | 0-1 |
| | Isolation Forest | 35% | - | 0-1 |
| | LSTM Sequential | 25% | - | 0-1 |
| | **AML_SCORE** | **100%** | **0.70** | **0-1** |

---

## 9. DEPLOYMENT CHECKLIST

- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Set up Django database: `python manage.py migrate`
- [ ] Configure MEDIA_ROOT and MEDIA_URL in settings.py
- [ ] Create superuser: `python manage.py createsuperuser`
- [ ] Run server: `python manage.py runserver`
- [ ] Test document upload endpoint
- [ ] Test biometric endpoints with sample images
- [ ] Set up AML monitoring with test transactions
- [ ] Configure admin panel for case review
- [ ] Set up logging for audit trail
- [ ] Configure email alerts for manual review cases
- [ ] Performance test with load scenarios

---

## 10. FUTURE ENHANCEMENTS

### Planned Upgrades:
1. **Full LSTM with PyTorch** - Replace EWMA with trained LSTM
2. **3D Liveness Detection** - Detect 3D masks
3. **Multi-language OCR** - Support regional scripts
4. **Real-time Dashboard** - Case monitoring & analytics
5. **Blockchain Audit Trail** - Immutable verification records
6. **API Rate Limiting** - Prevent abuse
7. **Mobile App** - Native iOS/Android integration
8. **Video KYC** - Live agent verification
9. **Periodic Re-verification** - Annual compliance checks
10. **Federation** - Multi-institution data sharing (secure)

---

**Document Version:** 1.0
**Last Updated:** 2025-04-07
**Author:** AI Development Team

