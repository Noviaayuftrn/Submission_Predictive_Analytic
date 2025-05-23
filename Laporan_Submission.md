﻿# Laporan Proyek Machine Learning - Novia Ayu Fitriana

 ## Project Overview

Diabetes adalah salah satu penyakit kronis yang berdampak luas terhadap kualitas hidup masyarakat. Berdasarkan data dari World Health Organization (WHO), jumlah penderita diabetes meningkat secara signifikan setiap tahun. Oleh karena itu, prediksi dini terhadap potensi seseorang terkena diabetes sangat penting untuk mencegah komplikasi lebih lanjut.

Dalam proyek ini, saya membangun sistem prediksi diabetes berdasarkan data kesehatan pasien menggunakan pendekatan klasifikasi machine learning. Dataset yang digunakan adalah Diabetes Data Set dari Kaggle, yang berisi informasi kesehatan 768 pasien, termasuk fitur seperti kadar glukosa, tekanan darah, indeks massa tubuh (BMI), dan usia.

## Business Understanding

### Problem Statements

1. Bagaimana memprediksi apakah seseorang berisiko terkena diabetes berdasarkan data medis?
2. Bagaimana mendeteksi pasien dengan risiko tinggi diabetes lebih awal hanya berdasarkan data kesehatan rutin, sehingga dapat dilakukan pencegahan sebelum munculnya gejala klinis?
3. Algoritma klasifikasi mana yang paling efektif untuk memodelkan data ini?

### Goals

1. Membangun model klasifikasi untuk memprediksi risiko diabetes.
2. Mengembangkan sistem prediktif berbasis data medis non-invasif (tanpa tes lanjutan) yang dapat membantu dokter dan tenaga kesehatan dalam melakukan deteksi dini risiko diabetes, untuk memungkinkan intervensi gaya hidup dan perawatan lebih cepat.
3. Mengevaluasi performa model menggunakan metrik akurasi, precision, recall, dan AUC.

### Solution Approach

- **Model 1: Neural Network (MLP)** - Menggunakan model neural network (Multi-Layer Perceptron) untuk memprediksi diabetes. Model ini dipilih karena kemampuannya untuk menangani data yang non-linier.
- **Model 2: Random Forest** - Menggunakan algoritma Random Forest sebagai model klasifikasi. Random Forest dipilih karena kemampuannya untuk menangani data dengan banyak fitur dan kemampuan untuk mengurangi overfitting.

## Data Understanding

Dataset ini Dapat dunduh di Kaggle: [Diabetes Dataset](https://www.kaggle.com/datasets/mathchi/diabetes-data-set)

Dataset yang digunakan adalah Diabetes Data Set dari Kaggle, berikut fitur yang ada dalam dataset:

1. **Pregnancies**: Jumlah kehamilan.
2. **Glucose**: Konsentrasi glukosa darah pada 2 jam setelah tes oral.
3. **BloodPressure**: Tekanan darah diastolik (mm Hg).
4. **SkinThickness**: Ketebalan lipatan kulit triceps (mm).
5. **Insulin**: Konsentrasi insulin darah (mu U/ml).
6. **BMI**: Indeks massa tubuh (BMI).
7. **Age**: Usia (tahun).
8. **Outcome**: Kelas target (0 = tidak diabetes, 1 = diabetes).

### Jumlah Data

Dataset diabetes.csv terdiri dari:
- 768 baris
- 8 kolom

## Exploratory Data Analysis (EDA)
- **Distribusi Target (Outcome):** Pada distribusi target, terlihat bahwa jumlah pasien yang tidak menderita diabetes (Outcome = 0) jauh lebih banyak, dengan sekitar 500 pasien, dibandingkan dengan yang menderita diabetes (Outcome = 1) yang jumlahnya sekitar 260 orang.
- **Visualisasi Fitur:** Setiap fitur dalam dataset divisualisasikan menggunakan histogram dan boxplot untuk memeriksa distribusi dan potensi adanya outlier.
  - Pada visualisasi histogram terlihat bahwa sebagian besar fitur memiliki distribusi yang tidak simetris atau condong ke satu sisi (skewed). Pada fitur seperti Pregnancies, Insulin, DiabetesPedigreeFunction, dan Age menunjukkan distribusi miring ke kanan, yang berarti banyak nilai yang rendah dan sedikit nilai yang tinggi. Fitur Glucose, BloodPressure, dan BMI tampak lebih mendekati distribusi normal, meskipun tetap ada ketidakseimbangan kecil. SkinThickness menunjukkan distribusi yang agak miring ke kiri. Sementara itu, fitur Outcome bersifat kategorikal dan menunjukkan ketidakseimbangan kelas, dengan lebih banyak pasien yang tidak menderita diabetes (Outcome = 0) dibandingkan yang menderita (Outcome = 1).
- **Korelasi Antar Fitur:** Analisis korelasi antar fitur menunjukkan bahwa ada hubungan yang cukup signifikan antara fitur-fitur tertentu, seperti antara BMI dan Glukosa. Dari heatmap ini, terlihat bahwa Glucose memiliki korelasi paling tinggi terhadap Outcome (0.47), diikuti oleh BMI (0.29), Age (0.24), dan Pregnancies (0.22). Hal ini mengindikasikan bahwa kadar glukosa darah, indeks massa tubuh, usia, dan jumlah kehamilan memiliki hubungan yang cukup relevan dengan kemungkinan seseorang menderita diabetes. Sementara itu, fitur-fitur seperti BloodPressure, SkinThickness, dan DiabetesPedigreeFunction menunjukkan korelasi yang lemah terhadap Outcome.
- **Pairplot:**  Visualisasi pasangan fitur dengan target Outcome menunjukkan perbedaan distribusi antara pasien yang menderita diabetes dan yang tidak, khususnya pada fitur BMI, Glukosa, dan Usia. 
  - pairplot keseluruhan: dapat diamati bahwa fitur Glucose, BMI, dan Age memiliki perbedaan distribusi yang cukup mencolok antara pasien dengan dan tanpa diabetes. Pasien dengan Outcome 1 (diabetes) cenderung memiliki nilai glukosa dan BMI yang lebih tinggi, serta usia yang lebih tua. Selain itu, kombinasi seperti Glucose vs BMI dan Age vs Glucose menunjukkan pola yang sedikit lebih terpisah antara kedua outcome, menandakan bahwa fitur-fitur ini bisa memiliki kekuatan prediktif terhadap status diabetes.
  - scatter plot hubungan BMI (Body Mass Index) dan Glukosa: Grafik menunjukkan beberapa insight penting. Pemisahan kelas mulai terlihat di mana pasien dengan diabetes (oranye) cenderung memiliki nilai glukosa dan BMI yang lebih tinggi, sedangkan pasien tanpa diabetes (biru) lebih banyak tersebar di daerah dengan glukosa kurang dari 125 dan BMI di bawah 35. Secara visual, terdapat kecenderungan bahwa semakin tinggi nilai BMI, semakin tinggi pula kadar glukosa, dan semakin besar kemungkinan seseorang mengidap diabetes. Selain itu, terlihat adanya beberapa titik ekstrem dengan nilai glukosa atau BMI yang sangat rendah atau sangat tinggi

## Data Preparation
Data dipersiapkan dengan membagi dataset menjadi fitur dan label (Outcome), kemudian dilakukan normalisasi untuk fitur-fitur numerik menggunakan StandardScaler. Data kemudian dibagi menjadi tiga set: training, validation, dan testing, dengan 537 data sebesar 70% untuk training, 115 data sebesar 15% untuk validation, dan 116 data sebesar 15% untuk testing.

**Penjelasan Proses:**

- **Pengecekan Missing Value:**
```python
df.isnull().sum()
```
  | Kolom      | Jumlah Missing Values |
  |----------------|---------|
  | Pregnancies              | 0         |
  | Glucose                     | 0         |
  | BloodPressure                   | 0         |
  | SkinThickness                  | 0         |
  | Insulin                        | 0         |
  | BMI                    | 0         |
  | DiabetesPedigreeFunction                     | 0         |
  | Age                    | 0         |
  | Outcome                     | 0         |
  
Pada dataset ini, tidak ditemukan missing value, sehingga tidak diperlukan tindakan penghapusan baris data.
- **Pengecekan Data Duplicate:**
```python
df.duplicated().sum()
```
Pada dataset ini, tidak ditemukan data duplikasi, sehingga tidak diperlukan pembersihan lebih lanjut terkait data ganda.
- **Memisahkan Fitur dan Label:**
```python
X = df.drop('Outcome', axis=1)
y = df['Outcome']
```
Fitur Outcome digunakan sebagai label target (0 = tidak diabetes, 1 = diabetes). Ini dilakukan untuk mencegah dominasi fitur dengan skala besar terhadap fitur lain.
- **Standardisasi fitur:**
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```
Proses ini penting karena model seperti neural network sangat sensitif terhadap skala data. Dengan standardisasi (mean = 0, std = 1), konvergensi saat pelatihan model menjadi lebih cepat dan stabil. Dengan maksud menguji model secara adil dan mencegah overfitting.
- **Membagi data:**
```python
X_train, X_temp, y_train, y_temp = train_test_split(..., test_size=0.3, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(...)
```
Data dibagi menjadi 70% data latih, 15% validasi, dan 15% data uji, dengan metode stratified split agar distribusi label tetap proporsional di semua subset data.

## Modeling
Dua model digunakan dalam proyek ini untuk membandingkan hasil prediksi: Multilayer Perceptron (MLP) yang merupakan model neural network, dan Random Forest, yang merupakan algoritma ensemble.

**Penjelasan Proses:**

 ### MLP (Neural Network)

```python
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), verbose=0)
```
- Arsitektur: 3 layer — Dense(64) → Dense(32) → Dense(1 dengan sigmoid).
- Pemilihan 64 dan 32 neuron didasarkan pada eksperimen awal (32–32, 64–32, 128–64).
- Aktivasi ReLU digunakan untuk percepatan konvergensi dan mencegah vanishing gradient.
- Optimizer Adam dipilih karena adaptif dan cepat.
- Loss function: binary_crossentropy, sesuai untuk klasifikasi biner.
- Training selama 100 epoch; grafik menunjukkan stabil setelah ~80 epoch.
- Tidak dilakukan hyperparameter tuning lanjutan, hanya uji arsitektur dasar dan pengamatan learning curve.
- Dilatih dan dievaluasi sebanyak 3 kali untuk memastikan hasil konsisten.

### Random Forest

Random Forest adalah ensemble learning yang menggunakan banyak pohon keputusan (100 estimators) untuk meningkatkan akurasi dan mengurangi overfitting. Model ini dilatih dengan parameter random_state=42 agar hasil konsisten. Random Forest cenderung kuat terhadap noise dan mampu menangani data dengan fitur yang banyak serta tipe variabel campuran.
```python
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
```
- Cocok untuk data tabular dengan fitur numerik seperti dalam proyek ini.
- Performa stabil dan cenderung lebih baik dalam akurasi dan interpretabilitas dibanding MLP.

### Kelebihan dan Kekurangan Tiap Model
| Model             | Kelebihan                                                                       | Kekurangan                                                                                              |
| ----------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| **MLP**           | Mampu menangkap pola non-linear yang kompleks; fleksibel untuk berbagai dataset | Butuh tuning parameter lebih hati-hati; sensitif terhadap skala data; rawan overfitting                 |
| **Random Forest** | Robust terhadap overfitting; tidak perlu scaling; interpretasi relatif mudah    | Tidak sebaik neural network dalam menangani interaksi kompleks; hasil bisa tidak stabil pada data kecil |
---
### Model Terbaik
model Random Forest dipilih sebagai model terbaik untuk proyek ini. Hal ini didasarkan pada beberapa pertimbangan penting, yaitu: proses pelatihan dan pengujian Random Forest berlangsung lebih cepat dan efisien, model ini lebih tahan terhadap overfitting, serta memiliki keunggulan dalam interpretabilitas. Random Forest memungkinkan kita untuk memahami kontribusi setiap fitur melalui feature importance, yang sangat membantu dalam menjelaskan hasil model kepada pihak non-teknis atau stakeholder. Dengan performa yang hampir setara dan kompleksitas yang lebih rendah, Random Forest dinilai sebagai pilihan yang lebih praktis dan dapat diandalkan untuk diterapkan dalam konteks nyata proyek ini.

---

## Evaluasi Model
Dalam proyek ini, metrik evaluasi utama yang digunakan untuk mengukur kinerja model adalah akurasi dan AUC (Area Under Curve). Akurasi mengukur proporsi prediksi yang benar terhadap keseluruhan data uji, dihitung dengan rumus:

``` Akurasi = (TP + TN) / (TP + TN + FP + FN) ```

di mana TP adalah True Positive, TN adalah True Negative, FP adalah False Positive, dan FN adalah False Negative. Metrik ini cocok digunakan untuk mengevaluasi performa awal, tetapi pada konteks medis seperti deteksi risiko diabetes, akurasi saja tidak cukup karena kita ingin menghindari kesalahan dalam mengidentifikasi pasien yang benar-benar berisiko (FN).

Untuk itu, menggunakan AUC (Area Under the ROC Curve) karena metrik ini mempertimbangkan keseimbangan antara true positive rate (recall/sensitivity) dan false positive rate di berbagai threshold. AUC memberikan gambaran menyeluruh tentang kemampuan model dalam membedakan kelas positif dan negatif. AUC bernilai 0.5 berarti model tidak lebih baik dari tebak-tebakan, sementara AUC 1.0 berarti model sempurna.

Hasil evaluasi menunjukkan bahwa model MLP memiliki akurasi 0.76 dan AUC 0.82, sedikit lebih tinggi dibandingkan Random Forest dengan akurasi 0.75 dan AUC 0.81. Hal ini menunjukkan bahwa kedua model cukup baik dalam mengklasifikasi pasien yang berisiko terkena diabetes, namun MLP sedikit lebih unggul dari segi kemampuan klasifikasi.

Namun, mengingat konteks bisnis dan tujuan utama proyek — yaitu membangun sistem prediktif yang dapat membantu tenaga medis mendeteksi risiko diabetes sejak dini — kami memilih Random Forest sebagai solusi akhir. Meskipun performanya sedikit di bawah MLP, Random Forest lebih mudah diinterpretasikan dan lebih cepat saat dilatih maupun diuji, yang sangat penting dalam lingkungan klinis yang membutuhkan penjelasan model dan efisiensi waktu.

Secara keseluruhan, hasil evaluasi model telah sesuai dengan problem statement: model dapat memprediksi risiko diabetes dengan cukup akurat berdasarkan data medis rutin, tanpa memerlukan tes lanjutan yang mahal atau invasif. Solusi yang dibangun juga selaras dengan goals proyek, yaitu membantu deteksi dini dan intervensi lebih awal melalui teknologi prediktif yang efektif dan praktis.
