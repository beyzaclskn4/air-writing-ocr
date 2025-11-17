# air_write_final_fixed2.py
# Havada parmak ucu ile yaz -> pinch ile bitir -> Tesseract OCR single-char tahmini gösterir

import cv2
import numpy as np
import mediapipe as mp
import pytesseract
import time
from collections import deque

# ---------- CONFIG ----------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

CAM_ID = 0
FRAME_SIZE = (640, 480)
CANVAS_BG = 255
DRAW_COLOR = (0, 0, 0)
LINE_THICK = 18                  # Daha kalın çizgi -> OCR için
SMOOTHING = 0.5
PINCH_THRESHOLD = 0.05
MIN_POINTS_FOR_OCR = 5
OCR_COOLDOWN = 1.0
MINI_SIZE = 200                  # Sol üst mini preview boyutu

# ---------- SETUP ----------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(CAM_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])

canvas = np.ones((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8) * CANVAS_BG
pts = deque(maxlen=1024)
prev_x, prev_y = None, None

def lm_to_px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)

def norm_dist(lm1, lm2):
    dx = lm1.x - lm2.x
    dy = lm1.y - lm2.y
    return (dx*dx + dy*dy)**0.5

def preprocess_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray < 250)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = max(xs.min()-10,0), min(xs.max()+10, img.shape[1])
    y1, y2 = max(ys.min()-10,0), min(ys.max()+10, img.shape[0])
    crop = gray[y1:y2, x1:x2]
    h, w = crop.shape
    size = max(h, w)
    square = 255 * np.ones((size, size), dtype=np.uint8)
    y_off = (size - h)//2
    x_off = (size - w)//2
    square[y_off:y_off+h, x_off:x_off+w] = crop
    square = cv2.GaussianBlur(square, (3,3), 0)
    square = cv2.resize(square, (256, 256), interpolation=cv2.INTER_LINEAR)
    _, thr = cv2.threshold(square, 128, 255, cv2.THRESH_BINARY_INV)
    return thr

def ocr_single_char(img_gray):
    config = "--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    try:
        text = pytesseract.image_to_string(img_gray, config=config)
    except Exception as e:
        print("Tesseract error:", e)
        return ""
    return text.strip()

# ---------- MAIN ----------
with mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1) as hands:
    last_ocr_time = 0
    predicted = ""
    show_pred_until = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        drawing = frame.copy()
        pinch = False

        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            idx_tip = hand.landmark[8]  # işaret parmağı ucu
            idx_pip = hand.landmark[6]  # pip eklemi
            thumb_tip = hand.landmark[4]

            # İşaret parmağı yukarı mı?
            is_index_up = idx_tip.y < idx_pip.y

            x_px, y_px = lm_to_px(idx_tip, w, h)

            # Smoothing
            if prev_x is not None:
                sx = int(prev_x * (1 - SMOOTHING) + x_px * SMOOTHING)
                sy = int(prev_y * (1 - SMOOTHING) + y_px * SMOOTHING)
            else:
                sx, sy = x_px, y_px

            # Çizim: sadece parmak ucu ile, pinch değilse
            if is_index_up and norm_dist(idx_tip, thumb_tip) >= PINCH_THRESHOLD:
                if prev_x is None and prev_y is None:
                    pts.clear()  # yeni çizim başlarken temizle
                pts.append((sx, sy))
                if len(pts) >= 2:
                    cv2.line(canvas, pts[-2], pts[-1], DRAW_COLOR, LINE_THICK)

            prev_x, prev_y = sx, sy

            # Pinch kontrolü
            if norm_dist(idx_tip, thumb_tip) < PINCH_THRESHOLD:
                pinch = True

            mp_drawing.draw_landmarks(drawing, hand, mp_hands.HAND_CONNECTIONS)
        else:
            prev_x, prev_y = None, None

        # Canvas overlay
        canvas_resized = cv2.resize(canvas, (w, h))
        alpha = 0.7
        mask_gray = cv2.cvtColor(canvas_resized, cv2.COLOR_BGR2GRAY)
        _, mask_bin = cv2.threshold(mask_gray, 250, 255, cv2.THRESH_BINARY_INV)
        mask_inv = cv2.bitwise_not(mask_bin)
        bg = cv2.bitwise_and(drawing, drawing, mask=mask_inv)
        fg = cv2.bitwise_and(canvas_resized, canvas_resized, mask=mask_bin)
        combined = cv2.addWeighted(bg, 1.0, fg, alpha, 0)

        # Mini preview sol üst
        mini = cv2.resize(canvas, (MINI_SIZE, MINI_SIZE))
        combined[0:MINI_SIZE, 0:MINI_SIZE] = mini

        cv2.putText(combined, "Pinch to finish | c=clear s=save q=quit", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 50), 1, cv2.LINE_AA)

        # OCR tetikleme
        if pinch and len(pts) >= MIN_POINTS_FOR_OCR and (time.time() - last_ocr_time) > OCR_COOLDOWN:
            prep = preprocess_for_ocr(canvas)
            last_ocr_time = time.time()
            if prep is not None:
                pred = ocr_single_char(prep)
                predicted = pred if pred else ""
                show_pred_until = time.time() + 1.2
                print("OCR prediction:", predicted if predicted else "[none]")

            canvas = np.ones((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8) * CANVAS_BG
            pts.clear()
            prev_x, prev_y = None, None

        # Büyük tahmin
        if predicted and time.time() < show_pred_until:
            cv2.putText(combined, predicted.upper(), (w//2 - 40, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 128, 0), 8, cv2.LINE_AA)

        cv2.imshow("Air Writing Fixed2", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas = np.ones((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8) * CANVAS_BG
            pts.clear()
            prev_x, prev_y = None, None
            print("Canvas cleared.")
        elif key == ord('s'):
            fname = f"airwrite_{int(time.time())}.png"
            cv2.imwrite(fname, canvas)
            print("Saved:", fname)

cap.release()
cv2.destroyAllWindows()
