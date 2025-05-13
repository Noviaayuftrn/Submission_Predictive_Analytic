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

Dataset yang digunakan adalah Diabetes Data Set dari Kaggle, yang berisi informasi tentang 768 pasien dengan 8 fitur yaitu:

1. **Pregnancies**: Jumlah kehamilan.
2. **Glucose**: Konsentrasi glukosa darah pada 2 jam setelah tes oral.
3. **BloodPressure**: Tekanan darah diastolik (mm Hg).
4. **SkinThickness**: Ketebalan lipatan kulit triceps (mm).
5. **Insulin**: Konsentrasi insulin darah (mu U/ml).
6. **BMI**: Indeks massa tubuh (BMI).
7. **Age**: Usia (tahun).
8. **Outcome**: Kelas target (0 = tidak diabetes, 1 = diabetes).

Dataset ini diimpor ke dalam Python dengan menggunakan `pandas` dan dapat dilihat dengan fungsi `head()`.

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
Data dipersiapkan dengan membagi dataset menjadi fitur dan label (Outcome), kemudian dilakukan normalisasi untuk fitur-fitur numerik menggunakan StandardScaler. Data kemudian dibagi menjadi tiga set: training, validation, dan testing, dengan 70% untuk training, 15% untuk validation, dan 15% untuk testing.

## Modeling
Dua model digunakan dalam proyek ini untuk membandingkan hasil prediksi: Multilayer Perceptron (MLP) yang merupakan model neural network, dan Random Forest, yang merupakan algoritma ensemble.

### MLP (Neural Network)
Model neural network dengan dua lapisan tersembunyi (64 dan 32 neuron) berhasil dilatih selama 100 epoch dengan menggunakan fungsi aktivasi ReLU dan sigmoid di output.
 - **Akurasi dan Loss:** Model ini mencapai akurasi 78% di data pengujian dengan loss sebesar 0.48.
 - **ROC AUC:** Nilai AUC mencapai 0.82, menunjukkan kemampuan model dalam membedakan kelas dengan baik.

### Random Forest
Model Random Forest diinisialisasi dengan 100 pohon keputusan dan dilatih menggunakan data yang sama. 
 - **Akurasi dan Loss:** Model ini mencapai akurasi 75% di data pengujian.
 - **ROC AUC:** Nilai AUC Random Forest mencapai 0.81, sedikit lebih rendah dibandingkan dengan MLP.

## Evaluasi Model
Evaluasi model dilakukan menggunakan beberapa metrik, termasuk akurasi, precision, recall, f1-score, dan AUC.
- **MLP:**
  - **Confusion Matrix:**
    - 67 pasien tanpa diabetes (Outcome = 0) berhasil diprediksi dengan benar, namun 8 pasien tersebut salah diprediksi sebagai diabetes.
    - 23 pasien diabetes (Outcome = 1) berhasil diprediksi dengan benar, sementara 18 pasien tersebut salah diprediksi sebagai tidak diabetes.
  - **Precision, Recall, F1-Score:** Precision untuk Outcome 0 adalah 0.79 dan untuk Outcome 1 adalah 0.74. Recall untuk Outcome 0 adalah 0.89 dan untuk Outcome 1 adalah 0.56. F1-Score untuk Outcome 0 adalah 0.84 dan untuk Outcome 1 adalah 0.64.
  - **AUC:** Model ini menunjukkan ROC AUC sebesar 0.82, yang mengindikasikan performa yang baik dalam membedakan antara pasien diabetes dan non-diabetes.
    
- **Random Forest:**
  - **Confusion Matrix:**
    - 66 pasien tanpa diabetes berhasil diprediksi dengan benar, sementara 9 pasien tersebut salah diprediksi sebagai diabetes.
    - 21 pasien diabetes berhasil diprediksi dengan benar, dan 20 pasien tersebut salah diprediksi sebagai non-diabetes.
  - **Precision, Recall, F1-Score:** Precision untuk Outcome 0 adalah 0.77 dan untuk Outcome 1 adalah 0.70. Recall untuk Outcome 0 adalah 0.88 dan untuk Outcome 1 adalah 0.51. F1-Score untuk Outcome 0 adalah 0.82 dan untuk Outcome 1 adalah 0.59.
  - **AUC:** Model ini menunjukkan ROC AUC sebesar 0.81, sedikit lebih rendah dari MLP.


## Perbandingan Model

- **Akurasi:**
  - MLP: 78%
  - Random Forest: 75%
- **AUC:**
  - MLP: 0.82
  - Random Forest: 0.81

## Conclusion
Dua model yang diuji dalam proyek ini, Multilayer Perceptron (MLP) dan Random Forest, memberikan hasil yang kompetitif dengan akurasi dan AUC yang mirip. Model Neural Network (MLP) sedikit lebih baik dibandingkan dengan Random Forest dalam hal akurasi dan AUC, meskipun perbedaannya tidak signifikan. MLP memiliki keunggulan dalam memisahkan kelas dengan lebih baik, sedangkan Random Forest lebih unggul dalam hal recall untuk Outcome 0, namun kalah dalam hal precision dan recall untuk Outcome 1.
