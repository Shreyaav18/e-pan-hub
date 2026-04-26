from ultralyticsplus import YOLO

_model = None

FIELD_ALIASES = {
    'name': 'name',
    'father': 'father',
    'father name': 'father',
    'dob': 'dob',
    'date of birth': 'dob',
    'pan': 'pan',
    'pan number': 'pan',
}

def _get_model():
    global _model
    if _model is None:
        _model = YOLO('foduucom/pan-card-detection')
        _model.overrides['conf'] = 0.25
        _model.overrides['iou'] = 0.45
        _model.overrides['agnostic_nms'] = False
        _model.overrides['max_det'] = 1000
    return _model


def detect_fields(img_bgr):
    model = _get_model()
    results = model.predict(img_bgr)

    class_names = results[0].names
    boxes = results[0].boxes
    fields = {}

    h, w = img_bgr.shape[:2]

    for box in boxes:
        cls_id = int(box.cls[0])
        raw_name = class_names[cls_id].lower().strip()
        cls_name = FIELD_ALIASES.get(raw_name, raw_name)

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        conf = float(box.conf[0])

        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        if cls_name not in fields or conf > fields[cls_name]['conf']:
            fields[cls_name] = {
                'bbox': (x1, y1, x2, y2),
                'conf': conf,
                'crop': crop,
            }

    return fields