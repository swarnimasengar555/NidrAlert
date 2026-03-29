import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import math
import time
import numpy as np
import urllib.request
import os
from datetime import datetime

# ─────────────────────────────────────────────
# MONGODB  –  pip install pymongo
# ─────────────────────────────────────────────
try:
    from pymongo import MongoClient
    MONGO_URI = "mongodb://localhost:27017/"   # ← change if needed
    DB_NAME   = "nidralert"
    _client   = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    _client.server_info()                      # will raise if not reachable
    db        = _client[DB_NAME]
    sessions  = db["sessions"]
    MONGO_OK  = True
    print("✓ MongoDB connected.")
except Exception as e:
    MONGO_OK  = False
    print(f"⚠  MongoDB not available ({e}). Session will still run; summary saved to session_log.txt instead.")

def save_session(doc: dict):
    """Persist session summary. Falls back to a text file if Mongo is down."""
    if MONGO_OK:
        try:
            result = sessions.insert_one(doc)
            print(f"✓ Session saved to MongoDB  (_id: {result.inserted_id})")
            return
        except Exception as e:
            print(f"⚠  MongoDB write failed: {e}")
    # Fallback
    with open("session_log.txt", "a", encoding="utf-8") as f:
        f.write(str(doc) + "\n")
    print("✓ Session saved to session_log.txt (fallback).")

# ─────────────────────────────────────────────
# ALARM
# ─────────────────────────────────────────────
try:
    import winsound
    def beep_alarm(): winsound.Beep(1500, 250)
except ImportError:
    def beep_alarm(): print("\a")

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────
DROWSY_PERCENTAGE        = 0.78
MAR_THRESHOLD            = 0.58
CONSECUTIVE_FRAMES_EYE   = 15
CONSECUTIVE_FRAMES_MOUTH = 25
NOD_PITCH_THRESHOLD      = 0.20   # head-down trigger
NOD_RECOVER_THRESHOLD    = 0.08   # must recover this far before next nod counts
NOD_DEBOUNCE_S           = 1.5    # min seconds between logged nods
YAW_DISTRACTION_THR      = 0.35
DISTRACTION_FRAMES       = 40
CALIBRATION_DURATION     = 20

# ─────────────────────────────────────────────
# 2. COLOURS / FONT
# ─────────────────────────────────────────────
C_RED    = (45,  45,  225)
C_AMBER  = (30,  180, 230)
C_WHITE  = (240, 240, 240)
C_DARK   = (20,  20,  20)
C_ACCENT = (0,   180, 120)
C_DIM    = (80,  80,  80)
C_LABEL  = (160, 160, 160)
FONT     = cv2.FONT_HERSHEY_DUPLEX

# ─────────────────────────────────────────────
# 3. SESSION STATE
# ─────────────────────────────────────────────
session_running    = True
calibration_samples = []
baseline_ear       = None
is_calibrated      = False

total_nods         = 0
total_yawns        = 0
total_distractions = 0
yawn_flag          = False
nod_head_down      = False        # STATE MACHINE: True while head is down
last_nod_time      = 0.0

f_eye = f_mouth = f_distract = 0

# ─────────────────────────────────────────────
# 4. CORE MATH
# ─────────────────────────────────────────────
def gdist(a, b):
    return math.dist((a.x, a.y), (b.x, b.y))

def calculate_ear(lm):
    ear_l = (gdist(lm[385], lm[373]) + gdist(lm[387], lm[380])) / (2.0 * gdist(lm[362], lm[263]) or 1)
    ear_r = (gdist(lm[160], lm[144]) + gdist(lm[158], lm[153])) / (2.0 * gdist(lm[33],  lm[133]) or 1)
    return (ear_l + ear_r) / 2.0

def calculate_mar(lm):
    hz = gdist(lm[61], lm[291])
    return gdist(lm[13], lm[14]) / hz if hz else 0.0

def estimate_head_pose(lm):
    # Pitch: nose tip vs forehead-chin midpoint, normalised by face height
    fh = abs(lm[152].y - lm[10].y) or 1e-6
    pitch = (lm[1].y - (lm[10].y + lm[152].y) / 2.0) / fh
    # Yaw: asymmetry of nose-to-face-edges distances
    dl = gdist(lm[1], lm[234])
    dr = gdist(lm[1], lm[454])
    yaw = (dl - dr) / ((dl + dr) or 1e-6)
    return float(pitch), float(yaw)

# ─────────────────────────────────────────────
# 5. UI HELPERS
# ─────────────────────────────────────────────
def alpha_rect(img, x1, y1, x2, y2, color, alpha):
    """Filled rectangle blended at `alpha` opacity."""
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return
    block = np.full(roi.shape, color, dtype=np.uint8)
    cv2.addWeighted(block, alpha, roi, 1 - alpha, 0, roi)
    img[y1:y2, x1:x2] = roi

def draw_header(img, w, elapsed):
    """Dark translucent header bar."""
    alpha_rect(img, 0, 0, w, 62, (0, 0, 0), 0.65)
    cv2.line(img, (0, 62), (w, 62), C_ACCENT, 1)
    # Logo: NIDRALERT  (single word, two colours)
    cv2.putText(img, "NIDRA", (22, 44), FONT, 1.05, C_WHITE,  2, cv2.LINE_AA)
    cv2.putText(img, "ALERT", (148, 44), FONT, 1.05, C_ACCENT, 2, cv2.LINE_AA)
    # Session timer (right-aligned)
    timer = f"SESSION  {int(elapsed//60):02d}:{int(elapsed%60):02d}"
    tw = cv2.getTextSize(timer, FONT, 0.55, 1)[0][0]
    cv2.putText(img, timer, (w - tw - 18, 40), FONT, 0.55, C_LABEL, 1, cv2.LINE_AA)

def draw_calibration_bar(img, w, h, cal_rem):
    """Bottom progress bar during calibration."""
    pct   = 1.0 - cal_rem / CALIBRATION_DURATION
    bar_w = int((w - 40) * pct)
    alpha_rect(img, 0, h - 38, w, h, (0, 0, 0), 0.55)
    cv2.rectangle(img, (20, h-24), (w-20, h-12), C_DIM, -1)
    if bar_w > 4:
        cv2.rectangle(img, (20, h-24), (20+bar_w, h-12), C_ACCENT, -1)
    label = f"CALIBRATING SENSORS — {int(cal_rem)}s remaining"
    cv2.putText(img, label, (22, h-30), FONT, 0.42, (0, 200, 255), 1, cv2.LINE_AA)

def draw_alert_banner(img, w, h, text, color):
    """
    Translucent pill-shaped alert banner centred on screen.
    Professional, non-opaque look.
    """
    scale  = 0.85
    thick  = 2
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
    pad_x, pad_y = 32, 18
    bw = tw + pad_x * 2
    bh = th + pad_y * 2
    bx = (w - bw) // 2
    by = h // 2 - bh // 2

    # Semi-transparent dark fill
    alpha_rect(img, bx, by, bx+bw, by+bh, (10, 10, 20), 0.72)
    # Coloured border
    cv2.rectangle(img, (bx, by), (bx+bw, by+bh), color, thick, cv2.LINE_AA)
    # Subtle inner glow line
    cv2.rectangle(img, (bx+3, by+3), (bx+bw-3, by+bh-3), color, 1, cv2.LINE_AA)
    # Text
    tx = bx + pad_x
    ty = by + pad_y + th - 4
    cv2.putText(img, text, (tx, ty), FONT, scale, C_WHITE, thick, cv2.LINE_AA)
    # Small coloured dot prefix
    dot_r = 6
    cv2.circle(img, (bx + pad_x - 16, by + bh//2), dot_r, color, -1)

def draw_end_button(img, w, h):
    """Returns btn_rect (x1,y1,x2,y2)."""
    bx1, by1, bx2, by2 = w-172, h-56, w-16, h-18
    alpha_rect(img, bx1, by1, bx2, by2, (20, 10, 40), 0.80)
    cv2.rectangle(img, (bx1, by1), (bx2, by2), C_RED, 1, cv2.LINE_AA)
    label = "END SESSION"
    lw = cv2.getTextSize(label, FONT, 0.50, 1)[0][0]
    cv2.putText(img, label, (bx1 + (bx2-bx1-lw)//2, by1+26), FONT, 0.50, C_WHITE, 1, cv2.LINE_AA)
    return (bx1, by1, bx2, by2)

def draw_metric_strip(img, w, h, ear, mar, pitch, yaw, baseline, calibrated):
    """Small stats strip at top-right corner (below header)."""
    if not calibrated:
        return
    lines = [
        (f"EAR  {ear:.2f} / {baseline*DROWSY_PERCENTAGE:.2f}", C_WHITE),
        (f"MAR  {mar:.2f}",                                     C_WHITE),
        (f"PITCH {pitch:+.2f}  YAW {yaw:+.2f}",                C_LABEL),
    ]
    sx = w - 210
    for i, (txt, col) in enumerate(lines):
        cv2.putText(img, txt, (sx, 88 + i * 20), FONT, 0.35, col, 1, cv2.LINE_AA)

# ─────────────────────────────────────────────
# 6. MOUSE CALLBACK
# ─────────────────────────────────────────────
_btn_rect_cache = (0, 0, 1, 1)

def on_click(event, x, y, flags, param):
    global session_running
    if event == cv2.EVENT_LBUTTONDOWN:
        bx1, by1, bx2, by2 = _btn_rect_cache
        if bx1 <= x <= bx2 and by1 <= y <= by2:
            session_running = False

# ─────────────────────────────────────────────
# 7. MODEL DOWNLOAD
# ─────────────────────────────────────────────
MODEL_PATH = "face_landmarker.task"
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
if not os.path.exists(MODEL_PATH):
    print("Downloading face landmarker model (~30 MB)…")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")

# ─────────────────────────────────────────────
# 8. MEDIAPIPE SETUP
# ─────────────────────────────────────────────
_face_options = mp_vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp_vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ─────────────────────────────────────────────
# 9. CAMERA + WINDOW
# ─────────────────────────────────────────────
video = cv2.VideoCapture(0)
if not video.isOpened():
    raise RuntimeError("Cannot open webcam.")

start_time = time.time()
WIN = "NIDRALERT"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(WIN, on_click)

# ─────────────────────────────────────────────
# 10. MAIN LOOP
# ─────────────────────────────────────────────
cur_ear = cur_mar = cur_pitch = cur_yaw = 0.0

with mp_vision.FaceLandmarker.create_from_options(_face_options) as detector:

    while session_running:
        ret, frame = video.read()
        if not ret:
            break

        frame   = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        elapsed = time.time() - start_time
        ts_ms   = int(elapsed * 1000)

        # ── MediaPipe detection ──────────────────────────────────────────
        try:
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = detector.detect_for_video(mp_img, ts_ms)
        except Exception:
            results = None          # swallow sporadic MediaPipe errors

        alert_text  = None
        alert_color = C_RED

        if results and results.face_landmarks:
            lm = results.face_landmarks[0]

            try:
                cur_ear   = calculate_ear(lm)
                cur_mar   = calculate_mar(lm)
                cur_pitch, cur_yaw = estimate_head_pose(lm)
            except Exception:
                pass   # landmark index out of range guard

            # ── Calibration ──────────────────────────────────────────────
            if not is_calibrated:
                calibration_samples.append(cur_ear)
                if elapsed >= CALIBRATION_DURATION:
                    baseline_ear  = float(np.mean(calibration_samples))
                    is_calibrated = True
                    print(f"✓ Calibrated. Baseline EAR = {baseline_ear:.4f}")

            # ── Detection (only after calibration) ───────────────────────
            else:
                dyn_thr = baseline_ear * DROWSY_PERCENTAGE

                # 1. Drowsiness
                if cur_ear < dyn_thr:
                    f_eye += 1
                    if f_eye > CONSECUTIVE_FRAMES_EYE:
                        alert_text  = "CRITICAL: DROWSINESS"
                        alert_color = C_RED
                else:
                    f_eye = 0

                # 2. Yawning
                if cur_mar > MAR_THRESHOLD:
                    f_mouth += 1
                    if f_mouth > CONSECUTIVE_FRAMES_MOUTH:
                        alert_text  = "WARNING: YAWNING"
                        alert_color = C_AMBER
                        if not yawn_flag:
                            total_yawns += 1
                            yawn_flag = True
                else:
                    f_mouth   = 0
                    yawn_flag = False

                # 3. Distraction (yaw)
                if abs(cur_yaw) > YAW_DISTRACTION_THR:
                    f_distract += 1
                    if f_distract > DISTRACTION_FRAMES:
                        alert_text  = "ALERT: FOCUS ON ROAD"
                        alert_color = C_AMBER
                        if f_distract == DISTRACTION_FRAMES + 1:
                            total_distractions += 1
                else:
                    f_distract = 0

                # 4. Head Nod — state machine
                #    Phase A: head goes DOWN  → set nod_head_down flag
                #    Phase B: head comes UP   → log nod, clear flag
                #    This prevents the old crash-after-2-nods bug because
                #    we NEVER stop the loop; we only increment a counter.
                now = time.time()
                if not nod_head_down:
                    if cur_pitch > NOD_PITCH_THRESHOLD:
                        nod_head_down = True
                else:
                    if cur_pitch < NOD_RECOVER_THRESHOLD:
                        # Completed a full nod cycle
                        nod_head_down = False
                        if now - last_nod_time > NOD_DEBOUNCE_S:
                            total_nods   += 1
                            last_nod_time = now
                            alert_text    = "ALERT: HEAD DROP"
                            alert_color   = C_RED

                if alert_text:
                    beep_alarm()

        # ── Draw UI ──────────────────────────────────────────────────────
        draw_header(img=frame, w=w, elapsed=elapsed)

        if is_calibrated:
            draw_metric_strip(frame, w, h, cur_ear, cur_mar, cur_pitch,
                              cur_yaw, baseline_ear, is_calibrated)
        else:
            draw_calibration_bar(frame, w, h, max(0.0, CALIBRATION_DURATION - elapsed))

        if alert_text:
            draw_alert_banner(frame, w, h, alert_text, alert_color)

        btn = draw_end_button(frame, w, h)
        _btn_rect_cache = btn                  # update for mouse callback

        cv2.imshow(WIN, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

# ─────────────────────────────────────────────
# 11. SESSION SUMMARY + SAVE
# ─────────────────────────────────────────────
duration = time.time() - start_time

summary = {
    "timestamp":        datetime.utcnow().isoformat() + "Z",
    "duration_seconds": round(duration, 1),
    "duration_fmt":     f"{int(duration//60)}m {int(duration%60)}s",
    "baseline_ear":     round(baseline_ear, 4) if baseline_ear else None,
    "drowsy_nods":      total_nods,
    "yawn_count":       total_yawns,
    "distraction_acts": total_distractions,
    "calibrated":       is_calibrated,
}

print(f"\n{'='*38}")
print(f"  NIDRALERT SESSION SUMMARY")
print(f"{'='*38}")
print(f"  Timestamp       : {summary['timestamp']}")
print(f"  Duration        : {summary['duration_fmt']}")
print(f"  Baseline EAR    : {summary['baseline_ear']}")
print(f"  Drowsy Nods     : {summary['drowsy_nods']}")
print(f"  Yawn Count      : {summary['yawn_count']}")
print(f"  Distraction Acts: {summary['distraction_acts']}")
print(f"{'='*38}\n")

save_session(summary)

video.release()
cv2.destroyAllWindows()