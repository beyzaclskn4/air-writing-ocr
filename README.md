# Air Writing Kararlı OCR

Bu proje, kullanıcıların **işaret parmağıyla havada yazdığı harfleri** gerçek zamanlı olarak tanıyabilen bir sistemdir. Hem **OpenCV**, hem **Mediapipe** hem de **Tesseract OCR** kullanılarak geliştirilmiş, yüksek doğruluk ve hassasiyet sağlayan bir Python uygulamasıdır.

---

## 🚀 Özellikler

* **Gerçek zamanlı parmak ve el takibi**: İşaret parmağı çizim başlatma/durdurma için kullanılır.
* **Canvas tabanlı çizim**: Çizim, OpenCV penceresinde canlı olarak görüntülenir.
* **Ultra hassas OCR**:

  * Farklı thresholding yöntemleri (Otsu, Adaptive, Mean)
  * Morfolojik işlemler (doldurma, gürültü temizleme, dilate)
  * CLAHE ile kontrast artırma
  * Gamma düzeltme
  * Unsharp masking ile keskinlik artırma
* **Koordinat bazlı pattern matching**: Çizilen harfin şekline göre tahmin.
* **Fallback OCR stratejileri**: Birden fazla PSM modu ve ağırlıklı skor sistemi ile maksimum doğruluk.
* **Mini önizleme**: Çizim alanının küçük bir önizlemesi sol üst köşede.
* **Klavye kısayolları**:

  * `c` → Canvas temizle
  * `s` → Çizimi PNG olarak kaydet
  * `q` → Çıkış

---

## 🛠 Kullanılan Teknolojiler ve Modüller

| Teknoloji/Modül         | Kullanım Alanı                                                                               |
| ----------------------- | -------------------------------------------------------------------------------------------- |
| **Python 3**            | Projenin temel dili, tüm algoritmalar ve veri işleme burada yapılıyor                        |
| **OpenCV (cv2)**        | Görüntü yakalama, çizim, maskeleme, resize, thresholding, morphology, blur ve görüntü işleme |
| **Mediapipe**           | El ve parmak takibi; landmark koordinatları ile parmak pozisyon analizi                      |
| **NumPy**               | Sayısal işlemler, koordinat ve array hesaplamaları, normalize etme                           |
| **Pytesseract**         | Tesseract OCR için Python wrapper; tek karakter ve kelime tanıma                             |
| **Math**                | Stroke yönleri, açı hesaplama ve mesafe ölçümleri                                            |
| **Collections (deque)** | Çizilen noktaların saklanması ve sliding window işlemleri                                    |
| **Time**                | Çizim ve OCR cooldown, süre takibi                                                           |
| **OS (opsiyonel)**      | Çizimi kaydetme, dosya isimlendirme                                                          |

---

## ⚙️ Fonksiyon Açıklamaları

### 1. `lm_to_px(lm, w, h)`

* **Açıklama:** Normalized landmark koordinatlarını piksel değerlerine çevirir.
* **Girdi:** Landmark (x,y normalized), frame genişlik, frame yükseklik.
* **Çıktı:** Piksel koordinat (x, y).

### 2. `norm_dist(lm1, lm2)`

* **Açıklama:** İki landmark arasındaki öklidyen mesafeyi hesaplar.

### 3. `is_index_pointing(hand)`

* **Açıklama:** Sadece işaret parmağı açık ve diğer parmaklar kapalı mı kontrol eder.
* **Kullanım:** Çizimi başlatmak için koşul.

### 4. `analyze_letter_shape(pts_list)`

* **Açıklama:** Çizilen noktaları analiz ederek harfin şekil özelliklerini çıkarır.
* **Çıktı:** Aspect ratio, merkez, yönler, ekstrem noktalar.

### 5. `pattern_match_letter(shape_data)`

* **Açıklama:** Harf şekline göre tahmin (pattern matching).
* **Özellik:** Aspect ratio ve stroke yönleri üzerinden skorlama.

### 6. `preprocess_for_ocr(img)`

* **Açıklama:** OCR için ultra optimize ön işlem (threshold, morphology, CLAHE, gamma, blur).
* **Amaç:** OCR doğruluğunu maksimuma çıkarmak.

### 7. `ocr_single_char(img_gray, shape_data=None)`

* **Açıklama:** Tek karakter OCR işlemi, pattern matching ile birlikte.
* **Özellik:** Birden fazla PSM modu, fallback mekanizmaları, ağırlıklı skor hesaplama.

---

## 💻 Kurulum

1. Python 3 kurulumu
2. Gerekli modüller:

```bash
pip install opencv-python mediapipe pytesseract numpy
```

3. Tesseract OCR kurulumu

* Windows örnek: `C:\Program Files\Tesseract-OCR\tesseract.exe`
* Bu yol `air_write_stable.py` içinde `pytesseract.pytesseract.tesseract_cmd` olarak ayarlanmalı.

---

## 🎮 Kullanım

```bash
python air_write_stable.py
```

* **Başlatma:** Sadece işaret parmağınızı açın.
* **Durdurma:** İşaret parmağı ve baş parmak birleşince çizim durur.
* **Mini önizleme:** Sol üst köşede gösterilir.
* **OCR sonucu:** Ekranın ortasında gösterilir.

---

## 📈 Avantajlar

* Yüksek doğruluk için hem OCR hem pattern matching kullanılır.
* Farklı ışık koşullarına ve hızlı çizimlere dayanıklı.
* Minimum gecikme ile gerçek zamanlı kullanıcı deneyimi.
* Tek karakter veya kısa kelimeler için optimize edilmiş.

### **LICENSE** 

```text
MIT License

Copyright (c) 2025 Beyza

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.