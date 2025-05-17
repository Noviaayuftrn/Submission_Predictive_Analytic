# Laporan Proyek Machine Learning - Novia Ayu Fitriana

 ## Project Overview

Diabetes adalah salah satu penyakit kronis yang berdampak luas terhadap kualitas hidup masyarakat. Berdasarkan data dari World Health Organization (WHO), jumlah penderita diabetes meningkat secara signifikan setiap tahun. Oleh karena itu, prediksi dini terhadap potensi seseorang terkena diabetes sangat penting untuk mencegah komplikasi lebih lanjut.

Dalam proyek ini, saya membangun sistem prediksi diabetes berdasarkan data kesehatan pasien menggunakan pendekatan klasifikasi machine learning. Dataset yang digunakan adalah Diabetes Data Set dari Kaggle (https://www.kaggle.com/datasets/mathchi/diabetes-data-set), yang berisi informasi kesehatan 768 pasien, termasuk fitur seperti kadar glukosa, tekanan darah, indeks massa tubuh (BMI), dan usia.

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

### Kondisi Data
- Missing Value: Pada dataset ini, tidak ditemukan missing value, sehingga tidak diperlukan tindakan penghapusan baris data.
- Duplicate: Pada dataset ini, tidak ditemukan data duplikasi, sehingga tidak diperlukan pembersihan lebih lanjut terkait data ganda.

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

- **Memisahkan Fitur da Label:**
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
```
  - Arsitektur: 2 hidden layer (64 dan 32 neuron), aktivasi ReLU, output sigmoid untuk klasifikasi biner.
  - Optimizer: adam
  - Loss: binary_crossentropy
  - Epoch: 100
  - Data validasi digunakan untuk memantau overfitting.

Model neural network dengan dua lapisan tersembunyi (64 dan 32 neuron) berhasil dilatih selama 100 epoch dengan menggunakan fungsi aktivasi ReLU dan sigmoid di output.
 - **Akurasi dan Loss:** Model ini mencapai akurasi 75% di data pengujian dengan loss sebesar 0.46.
 - **ROC AUC:** Nilai AUC mencapai 0.82, menunjukkan kemampuan model dalam membedakan kelas dengan baik.

### Random Forest
```python
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

```
 - Algoritma ensemble berbasis pohon keputusan.
 - Parameter penting: n_estimators=100 (jumlah pohon dalam hutan), random_state untuk replikasi hasil.
 - Cocok untuk dataset dengan fitur campuran, tidak perlu normalisasi data.

Model Random Forest diinisialisasi dengan 100 pohon keputusan dan dilatih menggunakan data yang sama. 
 - **Akurasi dan Loss:** Model ini mencapai akurasi 75% di data pengujian.
 - **ROC AUC:** Nilai AUC Random Forest mencapai 0.81, sedikit lebih rendah dibandingkan dengan MLP.

### Kelenihan dan Kekurangan Tiap Model
| Model             | Kelebihan                                                                       | Kekurangan                                                                                              |
| ----------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| **MLP**           | Mampu menangkap pola non-linear yang kompleks; fleksibel untuk berbagai dataset | Butuh tuning parameter lebih hati-hati; sensitif terhadap skala data; rawan overfitting                 |
| **Random Forest** | Robust terhadap overfitting; tidak perlu scaling; interpretasi relatif mudah    | Tidak sebaik neural network dalam menangani interaksi kompleks; hasil bisa tidak stabil pada data kecil |


## Evaluasi Model
Evaluasi model dilakukan menggunakan beberapa metrik, termasuk akurasi, precision, recall, f1-score, dan AUC.
- **MLP:**
  - **Confusion Matrix:**
    - 67 pasien tanpa diabetes (Outcome = 0) berhasil diprediksi dengan benar, namun 7 pasien tersebut salah diprediksi sebagai diabetes.
    - 19 pasien diabetes (Outcome = 1) berhasil diprediksi dengan benar, sementara 22 pasien tersebut salah diprediksi sebagai tidak diabetes.
  - **Precision, Recall, F1-Score:** Precision untuk Outcome 0 adalah 0.76 dan untuk Outcome 1 adalah 0.73. Recall untuk Outcome 0 adalah 0.91 dan untuk Outcome 1 adalah 0.46. F1-Score untuk Outcome 0 adalah 0.82 dan untuk Outcome 1 adalah 0.57.
  - **AUC:** Model ini menunjukkan ROC AUC sebesar 0.82, yang mengindikasikan performa yang baik dalam membedakan antara pasien diabetes dan non-diabetes.
    
- **Random Forest:**
  - **Confusion Matrix:**
    - 66 pasien tanpa diabetes berhasil diprediksi dengan benar, sementara 9 pasien tersebut salah diprediksi sebagai diabetes.
    - 21 pasien diabetes berhasil diprediksi dengan benar, dan 20 pasien tersebut salah diprediksi sebagai non-diabetes.
  - **Precision, Recall, F1-Score:** Precision untuk Outcome 0 adalah 0.77 dan untuk Outcome 1 adalah 0.70. Recall untuk Outcome 0 adalah 0.88 dan untuk Outcome 1 adalah 0.51. F1-Score untuk Outcome 0 adalah 0.82 dan untuk Outcome 1 adalah 0.59.
  - **AUC:** Model ini menunjukkan ROC AUC sebesar 0.81, sedikit lebih rendah dari MLP.


## Perbandingan Model

- **Akurasi:**
  - MLP: 75%
  - Random Forest: 75%
- **AUC:**
  - MLP: 0.82
  - Random Forest: 0.81

## Conclusion
Dua model yang diuji dalam proyek ini, Multilayer Perceptron (MLP) dan Random Forest, memberikan hasil yang kompetitif dengan akurasi dan AUC yang mirip. Model Neural Network (MLP) sedikit lebih baik dibandingkan dengan Random Forest dalam hal akurasi dan AUC, meskipun perbedaannya tidak signifikan. MLP memiliki keunggulan dalam memisahkan kelas dengan lebih baik, sedangkan Random Forest lebih unggul dalam hal recall untuk Outcome 0, namun kalah dalam hal precision dan recall untuk Outcome 1.
