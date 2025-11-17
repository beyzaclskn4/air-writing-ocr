# air_write_stable.py
import cv2
import numpy as np
import mediapipe as mp
import pytesseract
import time
from collections import deque
import math


,# ---------- CONFIG ----------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

CAM_ID = 0
FRAME_SIZE = (640, 480)
CANVAS_BG = 255
DRAW_COLOR = (0, 0, 0)
LINE_THICK = 30  # OCR için çok kalın çizgi (daha iyi tanıma)
SMOOTHING = 0.1  # Çok hafif smoothing - işaret parmağının ucunu direkt takip et

# İşaret parmağı kontrolü için eşikler
INDEX_EXTENDED_THRESHOLD = 0.15  # İşaret parmağı dışarı çıkmış mı?
FINGERS_CLOSED_THRESHOLD = 0.08  # Diğer parmaklar kapalı mı?
STOP_THRESHOLD = 0.06    # index+thumb birleşince dur
MIN_MOVE_DISTANCE = 3    # Minimum hareket mesafesi (piksel)
MIN_POINTS_FOR_OCR = 15  # OCR için daha fazla nokta (daha iyi tanıma)
OCR_COOLDOWN = 1.0
MINI_SIZE = 200

# ---------- SETUP ----------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(CAM_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])

# Canvas frame boyutlarına göre dinamik oluşturulacak (ilk frame'de)
canvas = None
canvas_h, canvas_w = 0, 0
pts = deque(maxlen=1024)
drawing_active = False
prev_x, prev_y = None, None

# ---------- FUNCTIONS ----------
def lm_to_px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)

def norm_dist(lm1, lm2):
    dx = lm1.x - lm2.x
    dy = lm1.y - lm2.y
    return (dx*dx + dy*dy)**0.5

def is_finger_extended(hand, finger_tip_idx, finger_pip_idx):
    """Parmağın açık olup olmadığını kontrol eder"""
    tip = hand.landmark[finger_tip_idx]
    pip = hand.landmark[finger_pip_idx]
    return tip.y < pip.y  # Tip PIP'ten yukarıda ise parmak açık

def is_index_pointing(hand):
    """Sadece işaret parmağı açık, diğer tüm parmaklar kapalı mı kontrol eder"""
    # İşaret parmağı: 8 (tip), 6 (PIP)
    # Orta parmak: 12 (tip), 10 (PIP)
    # Yüzük parmak: 16 (tip), 14 (PIP)
    # Serçe parmak: 20 (tip), 18 (PIP)
    
    index_tip = hand.landmark[8]
    index_pip = hand.landmark[6]
    
    middle_tip = hand.landmark[12]
    middle_pip = hand.landmark[10]
    
    ring_tip = hand.landmark[16]
    ring_pip = hand.landmark[14]
    
    pinky_tip = hand.landmark[20]
    pinky_pip = hand.landmark[18]
    
    # İşaret parmağı açık mı? (tip, pip'ten yukarıda olmalı)
    index_extended = index_tip.y < index_pip.y
    
    # Diğer parmaklar kapalı mı? (tip, pip'ten aşağıda olmalı)
    middle_closed = middle_tip.y > middle_pip.y
    ring_closed = ring_tip.y > ring_pip.y
    pinky_closed = pinky_tip.y > pinky_pip.y
    
    # İşaret parmağı diğer parmaklardan belirgin şekilde daha yukarıda mı?
    index_above_others = (index_tip.y < middle_tip.y - 0.05 and 
                         index_tip.y < ring_tip.y - 0.05 and 
                         index_tip.y < pinky_tip.y - 0.05)
    
    # Sadece işaret parmağı açık, diğer 3 parmak (orta, yüzük, serçe) kapalı olmalı
    return index_extended and middle_closed and ring_closed and pinky_closed and index_above_others

def analyze_letter_shape(pts_list):
    """Koordinat bazlı harf şekli analizi"""
    if len(pts_list) < 5:
        return None
    
    pts_array = np.array(pts_list)
    
    # Bounding box
    min_x, min_y = pts_array.min(axis=0)
    max_x, max_y = pts_array.max(axis=0)
    width = max_x - min_x
    height = max_y - min_y
    
    if width == 0 or height == 0:
        return None
    
    # Normalize koordinatlar (0-1 arası)
    normalized_pts = (pts_array - [min_x, min_y]) / [width, height]
    
    # Özellikler
    aspect_ratio = width / height if height > 0 else 1.0
    
    # Stroke yönleri analizi
    directions = []
    for i in range(1, len(pts_array)):
        dx = pts_array[i][0] - pts_array[i-1][0]
        dy = pts_array[i][1] - pts_array[i-1][1]
        if dx != 0 or dy != 0:
            angle = math.atan2(dy, dx)
            directions.append(angle)
    
    # Merkez noktası
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    # Yüksek ve alçak noktalar
    top_y = min_y
    bottom_y = max_y
    left_x = min_x
    right_x = max_x
    
    # Karakteristik noktalar (köşeler, uçlar)
    # En üst, en alt, en sol, en sağ noktalar
    top_idx = np.argmin(pts_array[:, 1])
    bottom_idx = np.argmax(pts_array[:, 1])
    left_idx = np.argmin(pts_array[:, 0])
    right_idx = np.argmax(pts_array[:, 0])
    
    return {
        'normalized_pts': normalized_pts,
        'aspect_ratio': aspect_ratio,
        'width': width,
        'height': height,
        'center': (center_x, center_y),
        'directions': directions,
        'top': top_y,
        'bottom': bottom_y,
        'left': left_x,
        'right': right_x,
        'extreme_points': {
            'top': pts_array[top_idx],
            'bottom': pts_array[bottom_idx],
            'left': pts_array[left_idx],
            'right': pts_array[right_idx]
        }
    }

def pattern_match_letter(shape_data):
    """Koordinat bazlı pattern matching - harf tahmini"""
    if shape_data is None:
        return None, 0.0
    
    aspect_ratio = shape_data['aspect_ratio']
    width = shape_data['width']
    height = shape_data['height']
    directions = shape_data['directions']
    
    # Basit pattern matching kuralları
    scores = {}
    
    # Yükseklik/genişlik oranına göre
    if aspect_ratio < 0.5:  # Dikey harfler (I, L, T, etc.)
        scores['I'] = 0.8
        scores['L'] = 0.7
        scores['T'] = 0.6
        scores['F'] = 0.6
    elif aspect_ratio > 1.5:  # Yatay harfler (Z, W, etc.)
        scores['Z'] = 0.8
        scores['W'] = 0.7
        scores['M'] = 0.6
    else:  # Kare/dengeli harfler (O, A, B, etc.)
        scores['O'] = 0.7
        scores['A'] = 0.6
        scores['B'] = 0.6
        scores['C'] = 0.6
        scores['D'] = 0.6
        scores['E'] = 0.6
        scores['G'] = 0.6
        scores['H'] = 0.6
        scores['K'] = 0.6
        scores['N'] = 0.6
        scores['P'] = 0.6
        scores['Q'] = 0.6
        scores['R'] = 0.6
        scores['S'] = 0.6
        scores['U'] = 0.6
        scores['V'] = 0.6
        scores['X'] = 0.6
        scores['Y'] = 0.6
    
    # Yön analizi (basit)
    if len(directions) > 0:
        # Dikey hareket varsa
        vertical_moves = sum(1 for d in directions if abs(d) > math.pi/3)
        if vertical_moves > len(directions) * 0.3:
            scores['I'] = scores.get('I', 0) + 0.2
            scores['L'] = scores.get('L', 0) + 0.1
    
    # En yüksek skorlu harfi bul
    if scores:
        best_letter = max(scores, key=scores.get)
        best_score = scores[best_letter]
        return best_letter, best_score
    
    return None, 0.0

def preprocess_for_ocr(img):
    """Ultra hassas OCR ön işleme - maksimum doğruluk için optimize"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Çizim alanını bul - daha hassas threshold
    ys, xs = np.where(gray < 245)  # Daha hassas (250'den 245'e)
    if len(xs) == 0 or len(ys) == 0:
        return None
    
    # Daha fazla padding (karakter için daha fazla alan)
    padding = 40  # 30'dan 40'a artırdık
    x1, x2 = max(xs.min()-padding, 0), min(xs.max()+padding, img.shape[1])
    y1, y2 = max(ys.min()-padding, 0), min(ys.max()+padding, img.shape[0])
    crop = gray[y1:y2, x1:x2]
    
    if crop.size == 0:
        return None
    
    h, w = crop.shape
    if h < 20 or w < 20:  # Minimum boyut artırıldı
        return None
    
    # Daha büyük minimum boyut (OCR için daha iyi)
    min_size = 200  # 150'den 200'e artırdık
    scale = max(1.0, min_size / max(h, w))
    if scale > 1.0:
        new_h, new_w = int(h * scale), int(w * scale)
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)  # Daha kaliteli
        h, w = new_h, new_w
    
    # Kare canvas oluştur (daha büyük boyut)
    size = max(h, w, 500)  # 400'den 500'e artırdık
    square = np.ones((size, size), dtype=np.uint8) * 255
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    square[y_off:y_off+h, x_off:x_off+w] = crop
    
    # Gürültü azaltma - daha hassas
    square = cv2.bilateralFilter(square, 5, 50, 50)  # Kenar koruyan filtre
    square = cv2.medianBlur(square, 3)  # Daha küçük kernel (daha hassas)
    
    # Histogram eşitleme (kontrast iyileştirme) - daha agresif
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))  # clipLimit artırıldı
    square = clahe.apply(square)
    
    # Gamma düzeltme (parlaklık optimizasyonu)
    gamma = 1.2
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    square = cv2.LUT(square, table)
    
    # Birden fazla thresholding yöntemi dene - daha hassas
    # Yöntem 1: Otsu thresholding
    _, thresh1 = cv2.threshold(square, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Yöntem 2: Adaptive thresholding - daha hassas parametreler
    thresh2 = cv2.adaptiveThreshold(square, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 8)  # Parametreler optimize edildi
    
    # Yöntem 3: Mean thresholding
    mean_val = np.mean(square)
    _, thresh3 = cv2.threshold(square, mean_val * 0.7, 255, cv2.THRESH_BINARY_INV)
    
    # En iyi thresholding sonucunu seç (karakter piksel oranına göre)
    scores = []
    for th in [thresh1, thresh2, thresh3]:
        black_ratio = np.sum(th == 0) / th.size
        # Optimal: %10-30 arası siyah piksel
        score = 1.0 - abs(black_ratio - 0.2)  # 0.2 (20%) ideal
        scores.append(score)
    
    best_idx = np.argmax(scores)
    thresh = [thresh1, thresh2, thresh3][best_idx]
    
    # Morphology - karakterleri kalınlaştır ve temizle - daha hassas
    # Daha küçük kernel (daha hassas işleme)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    
    # Close: Boşlukları doldur (daha az agresif)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    # Open: Gürültüyü temizle
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=1)
    # Dilate: Karakterleri hafifçe kalınlaştır
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    thresh = cv2.dilate(thresh, kernel_dilate, iterations=1)
    
    # Çok yüksek çözünürlük (OCR için çok önemli)
    target_size = 1200  # 800'den 1200'e çıkardık - maksimum hassasiyet
    if size < target_size:
        thresh = cv2.resize(thresh, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    elif size > target_size:
        thresh = cv2.resize(thresh, (target_size, target_size), interpolation=cv2.INTER_AREA)
    
    # Son kontrast ve keskinlik iyileştirmesi - daha hassas
    thresh = cv2.convertScaleAbs(thresh, alpha=1.4, beta=5)  # Beta eklendi
    
    # Unsharp masking (keskinlik artırma) - daha güçlü
    gaussian = cv2.GaussianBlur(thresh, (0, 0), 1.5)
    thresh = cv2.addWeighted(thresh, 2.0, gaussian, -1.0, 0)  # Daha güçlü keskinlik
    
    # Son temizlik: İzole pikselleri temizle
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(255 - thresh, connectivity=8)
    min_area = 10  # Çok küçük bölgeleri temizle
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            thresh[labels == i] = 255
    
    return thresh

def ocr_single_char(img_gray, shape_data=None):
    """Ultra optimize OCR - çoklu deneme, yüksek confidence threshold + koordinat analizi"""
    if img_gray is None or img_gray.size == 0:
        return ""
    
    # Koordinat bazlı pattern matching sonucu
    pattern_letter = None
    pattern_score = 0.0
    if shape_data is not None:
        pattern_letter, pattern_score = pattern_match_letter(shape_data)
        if pattern_letter:
            print(f"Pattern match: '{pattern_letter}' (score: {pattern_score:.2f})")
    
    # Minimum confidence threshold - daha düşük (daha fazla deneme)
    MIN_CONFIDENCE = 25  # %25 minimum güvenilirlik (30'dan düşürüldü)
    HIGH_CONFIDENCE = 50  # Yüksek güvenilirlik eşiği
    
    # Farklı OCR konfigürasyonları dene (en iyi sonucu bul) - daha fazla mod
    configs = [
        # PSM 10: Tek karakter (en yaygın ve genellikle en iyi)
        ("--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789", 1.0),
        # PSM 8: Tek kelime
        ("--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789", 0.95),
        # PSM 7: Tek metin satırı
        ("--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789", 0.9),
        # PSM 13: Ham satır
        ("--psm 13 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789", 0.85),
        # PSM 6: Tek uniform blok
        ("--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789", 0.8),
        # PSM 11: Sparse text (yeni eklendi)
        ("--psm 11 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789", 0.75),
        # PSM 12: Sparse text with OSD (yeni eklendi)
        ("--psm 12 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789", 0.7),
    ]
    
    results = []  # (text, confidence, weight)
    
    for config, weight in configs:
        try:
            # OCR yap
            data = pytesseract.image_to_data(img_gray, config=config, output_type=pytesseract.Output.DICT)
            
            # Sonuçları topla
            text_parts = []
            conf_parts = []
            for i in range(len(data['text'])):
                conf = int(data['conf'][i])
                if conf > MIN_CONFIDENCE:  # Sadece yüksek güvenilirlik sonuçları
                    txt = data['text'][i].strip()
                    if txt and txt.isalnum():  # Sadece alfanumerik
                        text_parts.append(txt)
                        conf_parts.append(conf)
            
            if text_parts:
                combined_text = ''.join(text_parts)
                avg_conf = sum(conf_parts) / len(conf_parts) if conf_parts else 0
                # Ağırlıklı skor
                weighted_score = avg_conf * weight
                results.append((combined_text, avg_conf, weighted_score))
        except Exception as e:
            continue
    
    # En yüksek ağırlıklı skora sahip sonucu seç - daha akıllı
    if results:
        # Önce en yüksek weighted score'a göre sırala
        results.sort(key=lambda x: x[2], reverse=True)
        
        # En iyi sonuçları filtrele - daha esnek
        best_results = [r for r in results if r[1] >= MIN_CONFIDENCE]
        
        # Yüksek güvenilirlik sonuçlarını önceliklendir
        high_conf_results = [r for r in best_results if r[1] >= HIGH_CONFIDENCE]
        if high_conf_results:
            best_results = high_conf_results
        
        if best_results:
            # En iyi sonuçları karakter bazında grupla
            char_scores = {}
            for text, conf, weighted in best_results:
                text = text.strip()
                if len(text) > 0 and len(text) <= 3:
                    char = text[0].upper()
                    if char.isalnum():
                        if char not in char_scores:
                            char_scores[char] = []
                        char_scores[char].append((conf, weighted))
            
            if char_scores:
                # Her karakter için ortalama skor hesapla
                char_avg_scores = {}
                for char, scores_list in char_scores.items():
                    avg_conf = sum(s[0] for s in scores_list) / len(scores_list)
                    avg_weighted = sum(s[1] for s in scores_list) / len(scores_list)
                    # En yüksek skorlu sonucu da ekle
                    max_conf = max(s[0] for s in scores_list)
                    char_avg_scores[char] = (avg_conf, avg_weighted, max_conf, len(scores_list))
                
                # En iyi karakteri seç (ortalama confidence + weighted score + max confidence)
                best_char = max(char_avg_scores.keys(), 
                              key=lambda c: char_avg_scores[c][0] * 0.4 + 
                                           char_avg_scores[c][1] * 0.4 + 
                                           char_avg_scores[c][2] * 0.2)
                
                avg_conf, avg_weighted, max_conf, count = char_avg_scores[best_char]
                
                # Pattern matching ile uyum kontrolü
                if pattern_letter and pattern_letter == best_char:
                    # OCR ve pattern matching uyumlu - güvenilirlik artır
                    final_conf = min(100, max_conf + 25)
                    print(f"OCR+Pattern: '{best_char}' (confidence: {final_conf:.1f}%, {count} matches, pattern match!)")
                elif pattern_letter and pattern_score > 0.75:
                    # Pattern matching çok güçlü ama OCR farklı
                    if max_conf < HIGH_CONFIDENCE:  # OCR düşük güvenilirlikse
                        print(f"Pattern preferred: '{pattern_letter}' (OCR: '{best_char}', conf: {max_conf:.1f}%)")
                        return pattern_letter
                    else:
                        print(f"OCR: '{best_char}' (confidence: {max_conf:.1f}%, {count} matches)")
                else:
                    print(f"OCR: '{best_char}' (confidence: {max_conf:.1f}%, {count} matches)")
                
                return best_char
    
    # Fallback 1: Daha düşük confidence threshold ile dene - daha hassas
    try:
        data = pytesseract.image_to_data(img_gray, 
            config="--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
            output_type=pytesseract.Output.DICT)
        
        fallback_results = []
        for i in range(len(data['text'])):
            conf = int(data['conf'][i])
            if conf > 15:  # Daha düşük threshold (20'den 15'e)
                txt = data['text'][i].strip()
                if txt and txt.isalnum() and len(txt) == 1:
                    fallback_results.append((txt.upper(), conf))
        
        if fallback_results:
            # En yüksek confidence'a sahip sonucu seç
            best_fallback = max(fallback_results, key=lambda x: x[1])
            char, conf = best_fallback
            print(f"OCR fallback: '{char}' (confidence: {conf:.1f}%)")
            return char
    except:
        pass
    
    # Fallback 2: Basit string OCR - birden fazla PSM modu
    for psm_mode in [10, 8, 7]:
        try:
            text = pytesseract.image_to_string(img_gray, config=f"--psm {psm_mode} --oem 3")
            text = text.strip()
            text = ''.join(c for c in text if c.isalnum())
            if len(text) > 0 and len(text) <= 2:
                char = text[0].upper()
                print(f"OCR final fallback (PSM {psm_mode}): '{char}'")
                return char
        except:
            continue
    
    # Fallback 3: Pattern matching (OCR başarısız olduysa)
    if pattern_letter and pattern_score > 0.6:
        print(f"Pattern only: '{pattern_letter}' (score: {pattern_score:.2f})")
        return pattern_letter
    
    return ""

# ---------- MAIN LOOP ----------
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

        # Canvas'ı frame boyutlarına göre oluştur (ilk seferinde veya boyut değiştiyse)
        if canvas is None or canvas_h != h or canvas_w != w:
            canvas = np.ones((h, w, 3), dtype=np.uint8) * CANVAS_BG
            canvas_h, canvas_w = h, w
            print(f"Canvas initialized: {canvas_w}x{canvas_h}")

        drawing = frame.copy()

        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            idx_tip = hand.landmark[8]
            thumb_tip = hand.landmark[4]

            x_px, y_px = lm_to_px(idx_tip, w, h)

            # Başlatma koşulu: Sadece işaret parmağı dışarı çıkmış (işaret yapma pozisyonu)
            # OCR sonucu gösterilirken yeni çizim başlamasın
            if not drawing_active:
                if is_index_pointing(hand) and time.time() >= show_pred_until:
                    drawing_active = True
                    pts.clear()
                    prev_x, prev_y = None, None
                    predicted = ""  # Önceki tahmini temizle
                    print("Drawing started - Index finger pointing.")

            # Durdurma koşulu: index + thumb birleşince
            if drawing_active:
                dist_stop = norm_dist(idx_tip, thumb_tip)
                if dist_stop < STOP_THRESHOLD:
                    drawing_active = False
                    
                    # Koordinat bazlı harf analizi (hassas algılama için)
                    shape_data = None
                    if len(pts) >= MIN_POINTS_FOR_OCR:
                        shape_data = analyze_letter_shape(list(pts))
                        if shape_data:
                            print(f"Letter shape: aspect_ratio={shape_data['aspect_ratio']:.2f}, "
                                  f"size={shape_data['width']:.0f}x{shape_data['height']:.0f}")
                    
                    # OCR preprocessing
                    prep = preprocess_for_ocr(canvas)
                    last_ocr_time = time.time()
                    if prep is not None and len(pts) >= MIN_POINTS_FOR_OCR:
                        # OCR + koordinat analizi birlikte (maksimum hassasiyet)
                        pred = ocr_single_char(prep, shape_data)
                        predicted = pred if pred else ""
                        show_pred_until = time.time() + 1.5
                        print("Final prediction:", predicted if predicted else "[none]")
                    # temizle
                    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * CANVAS_BG
                    pts.clear()
                    prev_x, prev_y = None, None
                    print("Drawing stopped - Index and thumb together.")

            # Çizim aktifse - İşaret parmağının ucundan direkt çiz
            if drawing_active:
                # Çizim sırasında işaret parmağı pozisyonunu kontrol et
                # Eğer diğer parmaklar açılırsa çizimi durdur
                if not is_index_pointing(hand):
                    drawing_active = False
                    print("Drawing stopped - Other fingers opened.")
                else:
                    # İşaret parmağının TAM UCUNDAN çiz - direkt pozisyonu kullan
                    # Smoothing sadece çizgiyi yumuşatmak için, başlangıç her zaman işaret parmağının ucu
                    if prev_x is None or prev_y is None:
                        # İLK NOKTA - İşaret parmağının tam ucundan başla (smoothing yok)
                        sx, sy = x_px, y_px
                        pts.append((sx, sy))
                        # İlk noktayı görsel olarak işaretle (kalem ucu gibi)
                        cv2.circle(canvas, (sx, sy), LINE_THICK//2, DRAW_COLOR, -1)
                        prev_x, prev_y = sx, sy
                        print(f"First point at: ({sx}, {sy})")
                    else:
                        # Sonraki noktalar - İşaret parmağının ucunu NEREDEYSE DİREKT kullan
                        # Çok hafif smoothing sadece titreşimleri azaltmak için
                        sx = int(prev_x * SMOOTHING + x_px * (1 - SMOOTHING))
                        sy = int(prev_y * SMOOTHING + y_px * (1 - SMOOTHING))
                        
                        # Minimum hareket mesafesi kontrolü (saçmalıkları engelle)
                        move_dist = ((sx - prev_x)**2 + (sy - prev_y)**2)**0.5
                        if move_dist < MIN_MOVE_DISTANCE:
                            # Çok küçük hareket, atla
                            pass
                        else:
                            # Çizgiyi önceki noktadan yeni noktaya çiz
                            pts.append((sx, sy))
                            if len(pts) >= 2:
                                cv2.line(canvas, pts[-2], pts[-1], DRAW_COLOR, LINE_THICK)
                            prev_x, prev_y = sx, sy
                
                # İşaret parmağının ucunda görsel gösterge (canlı çizim sırasında)
                cv2.circle(drawing, (x_px, y_px), 8, (0, 255, 0), -1)  # Yeşil nokta
                cv2.circle(drawing, (x_px, y_px), 10, (0, 255, 0), 2)  # Yeşil halka

            # İşaret parmağının ucunu her zaman göster (kalem ucu gibi)
            if not drawing_active:
                # Çizim aktif değilken işaret parmağının ucunu göster
                cv2.circle(drawing, (x_px, y_px), 6, (255, 0, 0), -1)  # Mavi nokta
                cv2.circle(drawing, (x_px, y_px), 8, (255, 0, 0), 2)  # Mavi halka
            
            mp_drawing.draw_landmarks(drawing, hand, mp_hands.HAND_CONNECTIONS)

        else:
            # El kaybolduğunda çizimi durdur
            if drawing_active:
                drawing_active = False
                print("Drawing stopped - Hand lost.")
            prev_x, prev_y = None, None

        # Canvas overlay - Canvas artık frame boyutlarında, resize gerek yok
        alpha = 0.7
        mask_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask_bin = cv2.threshold(mask_gray,250,255,cv2.THRESH_BINARY_INV)
        mask_inv = cv2.bitwise_not(mask_bin)
        bg = cv2.bitwise_and(drawing,drawing,mask=mask_inv)
        fg = cv2.bitwise_and(canvas,canvas,mask=mask_bin)
        combined = cv2.addWeighted(bg,1.0,fg,alpha,0)

        # Mini preview
        mini = cv2.resize(canvas,(MINI_SIZE,MINI_SIZE))
        combined[0:MINI_SIZE,0:MINI_SIZE]=mini

        # Metin
        cv2.putText(combined,"Start: Index finger only | Stop: Index+Thumb together | c=clear s=save q=quit",
                    (10,h-10),cv2.FONT_HERSHEY_SIMPLEX,0.4,(50,50,50),1,cv2.LINE_AA)
        
        # Çizim durumu göstergesi
        if drawing_active:
            cv2.putText(combined, "DRAWING...", (w-150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        # OCR sonucu
        if predicted and time.time() < show_pred_until:
            cv2.putText(combined,predicted.upper(),(w//2-50,h//2),
                        cv2.FONT_HERSHEY_SIMPLEX,4.0,(0,128,0),8,cv2.LINE_AA)

        cv2.imshow("Air Writing Stable",combined)

        key=cv2.waitKey(1)&0xFF
        if key==ord('q'):
            break
        elif key==ord('c'):
            canvas=np.ones((canvas_h, canvas_w, 3), dtype=np.uint8)*CANVAS_BG
            pts.clear()
            prev_x,prev_y=None,None
            print("Canvas cleared.")
        elif key==ord('s'):
            fname=f"airwrite_{int(time.time())}.png"
            cv2.imwrite(fname,canvas)
            print("Saved:",fname)

cap.release()
cv2.destroyAllWindows()
