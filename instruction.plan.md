Buatkan saya project Machine Learning Python bernama **FEREmotionDetector** menggunakan dataset **FER-2013**.

Tujuan utama project ini adalah agar saya memahami alur lengkap:

```text
Image
→ Pixel
→ Preprocessing
→ Feature Extraction
→ X dan y
→ Train/Test Split
→ model.fit()
→ Evaluation
→ Save Model
→ Webcam
→ Realtime Prediction
```

Saya sedang belajar Machine Learning, jadi prioritaskan kode yang sederhana, mudah dibaca, dan mudah dipahami.

Jangan overengineering.

## Tech Stack

Gunakan:

* Python
* OpenCV
* NumPy
* Pandas
* scikit-learn
* joblib
* scikit-image hanya jika diperlukan untuk HOG

Jangan gunakan:

* TensorFlow
* PyTorch
* YOLO
* Keras

Saya ingin memahami pendekatan traditional Machine Learning menggunakan scikit-learn terlebih dahulu.

---

# Dataset

Saya menggunakan dataset **FER-2013**.

Dataset memiliki 7 kelas ekspresi:

```text
angry
disgust
fear
happy
sad
surprise
neutral
```

Image FER-2013 biasanya berupa grayscale:

```text
48 × 48
```

Buat loader yang mendukung struktur dataset seperti:

```text
dataset/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── sad/
│   ├── surprise/
│   └── neutral/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── sad/
    ├── surprise/
    └── neutral/
```

Jika dataset ternyata hanya memiliki satu folder dengan subfolder class, program harus tetap mudah disesuaikan.

Jangan mengubah dataset asli.

---

# Project Structure

Buat struktur sederhana:

```text
FEREmotionDetector/
│
├── dataset/
│
├── models/
│   └── emotion_model.pkl
│
├── src/
│   ├── dataset_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   └── face_detection.py
│
├── inspect_dataset.py
├── visualize_features.py
├── train.py
├── evaluate.py
├── camera.py
├── requirements.txt
└── README.md
```

Jangan membuat file yang tidak diperlukan.

---

# 1. Dataset Inspector

Buat:

```text
inspect_dataset.py
```

Script harus menghitung jumlah image per class.

Output contoh:

```text
FER-2013 Dataset

angry      : 3995
disgust    : 436
fear       : 4097
happy      : 7215
sad        : 4830
surprise   : 3171
neutral    : 4965

Total Images: ...
```

Tampilkan juga:

```text
Example image shape: (48, 48)
dtype: uint8
min pixel: 0
max pixel: 255
```

Tujuannya agar saya memahami bahwa image sebenarnya hanyalah matrix angka.

---

# 2. Image Preprocessing

Buat function:

```python
def preprocess_image(image):
    ...
```

Lakukan:

1. baca image
2. convert grayscale jika belum grayscale
3. resize menjadi `48x48`
4. optional histogram equalization
5. normalize jika memang diperlukan

Jangan melakukan preprocessing berlebihan.

Ketika debug aktif, tampilkan:

```text
Original shape: (48, 48)
Processed shape: (48, 48)

Pixel example:
[42, 38, 41, 50, ...]
```

Jelaskan bahwa:

```text
48 × 48 = 2304 pixels
```

Jika di-flatten langsung maka satu image menjadi:

```text
2304 features
```

Tetapi saya ingin menggunakan feature extraction yang lebih baik daripada raw pixel jika memungkinkan.

---

# 3. HOG Feature Extraction

Gunakan **Histogram of Oriented Gradients (HOG)** sebagai feature utama.

Gunakan:

```python
from skimage.feature import hog
```

Buat function:

```python
def extract_features(image):
    ...
```

Implementasi harus sederhana.

Gunakan parameter HOG yang cocok untuk image `48x48`.

Misalnya:

```python
hog(
    image,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    feature_vector=True
)
```

Jika parameter tersebut kurang cocok untuk 48x48, gunakan parameter yang menurutmu lebih tepat dan jelaskan alasannya di README.

Tampilkan:

```text
Image shape:
(48, 48)

HOG feature shape:
(N,)
```

Print 10 feature pertama dalam debug mode:

```text
[0.12, 0.04, 0.31, ...]
```

---

# 4. Visualize Features

Buat:

```text
visualize_features.py
```

Ambil satu image dari dataset.

Tampilkan menggunakan OpenCV atau matplotlib:

```text
Original Image
Grayscale Image
HOG Visualization
```

Saya ingin melihat bahwa HOG menangkap pola garis/tepi wajah seperti:

* bentuk mata
* alis
* mulut
* kontur wajah

Jangan menggunakan subplot yang terlalu kompleks.

---

# 5. Create X and y

Dataset loader harus menghasilkan:

```python
X = []
y = []
```

Untuk setiap image:

```text
image
↓
preprocess
↓
HOG
↓
feature vector
↓
X
```

Sedangkan nama folder menjadi:

```text
y
```

Contoh:

```text
dataset/train/happy/123.jpg
```

menghasilkan:

```python
X[index] = [0.21, 0.04, 0.33, ...]
y[index] = "happy"
```

Setelah semua selesai, tampilkan:

```text
X shape: (28709, N)
y shape: (28709,)
```

Jelaskan di komentar:

```text
rows    = jumlah image
columns = jumlah features per image
```

---

# 6. Train/Test Split

Jika menggunakan satu dataset, gunakan:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Jika FER-2013 sudah memiliki folder train/test resmi, prioritaskan pembagian resmi tersebut dan jangan split ulang test set.

Jelaskan perbedaannya di README.

---

# 7. Model

Mulai dengan model sederhana.

Saya ingin mencoba:

```python
from sklearn.svm import LinearSVC
```

atau classifier scikit-learn lain yang cocok untuk HOG features.

Pilih model yang masuk akal untuk:

```text
~35,000 images
×
HOG features
```

Jangan langsung menggunakan RandomForest jika LinearSVC lebih cocok untuk dimensional feature vector seperti HOG.

Berikan alasan singkat di README.

Training:

```python
model.fit(X_train, y_train)
```

Tampilkan:

```text
Training started...

Training samples : ...
Features         : ...
Classes          : 7

Training completed.
```

---

# 8. Evaluation

Evaluasi menggunakan:

```python
accuracy_score
classification_report
confusion_matrix
```

Tampilkan output seperti:

```text
Accuracy: 0.xx

              precision    recall    f1-score

angry
disgust
fear
happy
sad
surprise
neutral
```

Tampilkan confusion matrix.

Perhatikan dataset FER-2013 memiliki class imbalance, khususnya class `disgust`.

Gunakan jika diperlukan:

```python
class_weight="balanced"
```

atau pendekatan sederhana lain.

Jangan melakukan oversampling rumit pada versi pertama.

---

# 9. Save Model

Simpan menggunakan joblib:

```python
joblib.dump(...)
```

ke:

```text
models/emotion_model.pkl
```

PENTING:

Model inference harus menggunakan preprocessing dan HOG configuration yang sama persis seperti training.

Hindari duplikasi parameter HOG antara training dan camera.

Buat satu function reusable:

```python
extract_features()
```

yang digunakan keduanya.

---

# 10. Webcam Realtime Emotion Detection

Buat:

```text
camera.py
```

Gunakan:

```python
cv2.VideoCapture(0)
```

Untuk setiap frame:

```text
Webcam
↓
Frame
↓
Detect Face
↓
Crop Face
↓
Grayscale
↓
Resize 48x48
↓
HOG
↓
model.predict()
↓
Emotion
```

Gunakan OpenCV Haar Cascade untuk face detection agar dependency tetap sederhana.

Misalnya:

```python
cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)
```

---

# 11. Bounding Box

Setelah wajah ditemukan:

```python
x, y, w, h
```

gambar rectangle:

```python
cv2.rectangle()
```

Tampilkan prediction di atas wajah:

```text
HAPPY
```

Contoh:

```text
┌───────────────────┐
│ HAPPY 72%         │
│                   │
│       FACE        │
│                   │
└───────────────────┘
```

---

# 12. Confidence Score

Jika model yang digunakan mendukung:

```python
decision_function()
```

gunakan untuk mendapatkan skor prediction.

Jangan menampilkan angka tersebut sebagai probabilitas kecuali memang dikonversi/calibrated menjadi probability.

Jika ingin probability yang benar, boleh menggunakan:

```python
CalibratedClassifierCV
```

tetapi buat optional karena training akan lebih berat.

Versi pertama cukup:

```text
Emotion: HAPPY
```

Jika confidence ditampilkan, beri label:

```text
Score
```

bukan `% probability` jika bukan probability sesungguhnya.

---

# 13. Multiple Faces

Jika webcam menemukan lebih dari satu wajah, lakukan prediction untuk setiap wajah.

Contoh:

```text
Person 1 → Happy
Person 2 → Neutral
Person 3 → Surprise
```

Setiap wajah memiliki bounding box masing-masing.

---

# 14. Performance

Karena prediction dilakukan realtime:

Jangan melakukan training di `camera.py`.

`camera.py` hanya:

```text
load model once
↓
open camera
↓
inference loop
```

Jangan load model ulang setiap frame.

Jika diperlukan, resize webcam frame sebelum face detection agar lebih cepat.

Contoh:

```text
640x480
```

---

# 15. Quit

Tekan:

```text
q
```

untuk keluar.

Pastikan:

```python
camera.release()
cv2.destroyAllWindows()
```

selalu dipanggil.

---

# 16. Prediction Overlay

Gunakan label:

```text
Angry
Disgust
Fear
Happy
Sad
Surprise
Neutral
```

Jangan menyebut prediction sebagai emosi sebenarnya yang sedang dirasakan orang.

Gunakan istilah:

```text
Facial Expression: Happy
```

karena model hanya membaca ekspresi visual wajah.

README harus menjelaskan:

```text
Facial expression classification ≠ mengetahui perasaan seseorang.
```

---

# 17. README

README harus menjadi tutorial singkat.

Jelaskan konsep berikut.

## Image menjadi angka

FER image:

```text
48 × 48
```

berarti:

```text
2304 pixel
```

Setiap pixel memiliki intensity:

```text
0 → hitam
255 → putih
```

Contoh:

```text
[
  [12, 14, 20],
  [24, 31, 40],
  [80, 94, 101]
]
```

---

## Feature Extraction

Jelaskan:

```text
Image
↓
48×48 pixel matrix
↓
HOG
↓
Feature Vector
```

HOG membantu menangkap struktur seperti:

```text
edge
gradient
bentuk mata
bentuk mulut
kontur wajah
```

---

## X dan y

Jelaskan:

```python
X.shape
```

misalnya:

```text
(28709, 900)
```

berarti:

```text
28709 image
900 feature setiap image
```

Sedangkan:

```python
y
```

berisi:

```text
happy
sad
neutral
...
```

---

## model.fit()

Jelaskan sederhana apa yang terjadi ketika:

```python
model.fit(X_train, y_train)
```

Model mencoba menemukan pola feature yang membedakan:

```text
Happy
vs
Sad
vs
Angry
vs
...
```

---

## Prediction

Jelaskan:

```python
face = preprocess_image(face)

features = extract_features(face)

prediction = model.predict([
    features
])
```

Mengapa terdapat:

```python
[features]
```

jelaskan bahwa scikit-learn mengharapkan input:

```text
(samples, features)
```

jadi satu wajah tetap harus berbentuk:

```text
(1, N)
```

bukan:

```text
(N,)
```

---

# 18. Avoid Data Leakage

Pastikan tidak terjadi data leakage.

Jangan menggunakan test dataset saat training.

Jika menggunakan scaler:

```python
StandardScaler
```

gunakan:

```python
Pipeline
```

agar scaler hanya belajar dari training data.

Contoh:

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearSVC(...))
])
```

Gunakan scaler hanya jika memang berguna untuk model yang dipilih.

---

# 19. Reproducibility

Gunakan:

```python
random_state=42
```

jika classifier mendukung.

Simpan konfigurasi seperti:

```text
image size
HOG parameters
class list
```

di satu tempat agar training dan inference konsisten.

---

# 20. Error Handling

Handle error sederhana:

* dataset folder tidak ditemukan
* image corrupt
* model belum di-training
* webcam tidak tersedia
* tidak ada wajah pada frame

Contoh:

```text
Model not found.
Run:

python train.py
```

Jangan membuat stack trace yang membingungkan untuk error sederhana.

---

# 21. Commands

Project harus bisa dijalankan kira-kira:

```bash
python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

python inspect_dataset.py

python visualize_features.py

python train.py

python evaluate.py

python camera.py
```

Untuk Windows dokumentasikan juga:

```bash
.venv\Scripts\activate
```

---

# 22. Code Style

Saya sedang belajar Machine Learning.

Karena itu:

* jangan overengineering
* function sederhana lebih baik daripada class yang tidak diperlukan
* variable harus jelas
* jangan membuat abstraction berlebihan
* jangan membuat enterprise architecture
* komentar hanya pada bagian penting
* gunakan type hints seperlunya
* print shape pada proses training
* code harus dapat saya baca dari atas ke bawah

Saya ingin `train.py` terlihat kurang lebih seperti:

```text
load dataset

extract X and y

print shape

create model

fit

predict

evaluate

save
```

dan bukan tersembunyi di balik banyak class.

---

# Final Goal

Ketika menjalankan:

```bash
python camera.py
```

webcam terbuka.

Ketika wajah saya terlihat, program membuat bounding box dan menampilkan salah satu:

```text
Facial Expression: Happy
Facial Expression: Neutral
Facial Expression: Surprise
...
```

Pipeline final:

```text
FER-2013
   ↓
48×48 images
   ↓
HOG feature extraction
   ↓
scikit-learn classifier
   ↓
emotion_model.pkl
   ↓

WEBCAM
   ↓
Face Detection
   ↓
Crop Face
   ↓
48×48 grayscale
   ↓
HOG
   ↓
model.predict()
   ↓
Facial Expression
```

Prioritas utama adalah **project yang benar-benar berjalan sekaligus mengajari saya fundamental image classification dengan scikit-learn**.
