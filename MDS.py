import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.datasets import load_digits, make_blobs
from sklearn.metrics import pairwise_distances

st.title("Multi-Dimensional Scaling (MDS)")

st.markdown("""
    Teknik **MDS** (Multi-Dimensional Scaling) merupakan sebuah machine learning yang termasuk ke dalam
    unsupervised machine learning. MDS biasa digunakan untuk memetakan data berdimensi tinggi
    ke ruang berdimensi rendah (2D atau 3D) sambil mempertahankan jarak antar-poin.
    """)

st.markdown(""" MDS memiliki beberapa tipe, yaitu :
- Metric MDS : menempatkan pasangan pertama pada huruf kecil terpisah dari titik data
- non-Metric MDS : fokus pada menjaga urutan jarak dari nilai asli dari jarak.""")
    
st.markdown("""
MDS dianggap penting karena beberapa alasan, diantaranya :
- mampu menopang dan memahami kompleksitas suatu data lebih baik dari metode lainnya,
- membantu dalam memvisualisasikan data, 
- serta mendukung pengambilan keputusan pada berbagai aspek.""")

st.subheader("Mathematical Formulation of MDS")
st.markdown("""
ide dasar dalam menemukan proses umum Multi DImensional Scaling ini merupakan meminimumkan varians pada ruang dimensi besar dan dependen varians pada ruang dimensi yang lebih kecil.
tugas utama dalam mengurangi MDS dirumuskan sebagai berikut :""")

st.latex(r"\text{Stress} = \sqrt{\sum_i \sum_j W_{ij} (d_{ij} - D_{ij})^2}")

st.subheader("MDS pada Dataset Digits")
st.markdown("""
dataset ini terdiri atas :
- gambar grayscale dengan karakter alfanumerik (0 sampai 9) yang masing-masingnya diwakili oleh gambar piksel 8x8
- tiap gambar dilakukan pengonversian menjadi sebuah kombinasi 64 elemen. dengan kata lain, tiap data merupakan vektor dengan 64 nilai yang sesuai dengan piksel
- data sering digunakan dalam pengenalan serta pengelompokan angka""")

st.code("""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import MDS

# Load a sample dataset (e.g., the digits dataset)
data = load_digits()
X, y = data.data, data.target""", language="python")

# Load a sample dataset (e.g., the digits dataset)
data = load_digits()
X, y = data.data, data.target

("Original Dimension of X = ", X.shape)

st.code("""
n_components = 2  
mds = MDS(n_components=n_components)

# Fit the MDS model to your data
X_reduced = mds.fit_transform(X)""", language = "python")


st.write('Dimension of X after MDS = ', "(1797,2)")

st.code("""
plt.figure(figsize=(8, 6))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap=plt.cm.get_cmap("jet", 10))
plt.colorbar(label='Digit Label', ticks=range(10))
plt.title("MDS Visualization of Digits Dataset")
plt.xlabel("MDS Dimension 1")
plt.ylabel("MDS Dimension 2")
plt.show()""", language = "python")

st.image("https://raw.githubusercontent.com/naaufald/MultiDimensionalScaling_Multivariat/main/MDS 1.png")

st.markdown("titik dengan warna yang sama cenderung membuat cluster yang mengartikan bahwa digit yang memiliki bentuk serupa memiliki kemiripan piksel, semakin dekat dua titik, maka semakin mirip gambar angka aslinya, begitu juga dengan kebalikannya.")

st.code("""
X, _ = make_blobs(n_samples=100, n_features=3, centers=2, random_state=42)

print('Original Dimension of X : ', X.shape)
# Perform MDS to reduce the dimensionality to 2D

mds = MDS(n_components=2, random_state=42)
X_2d = mds.fit_transform(X)

print('Dimension of X after MDS : ', X_2d.shape)""", language="python")

('Original Dimension of X : ', "(100,3)")
('Dimension of X after MDS : ', "(100,2)")

st.code("""
plt.scatter(X_2d[:, 0], X_2d[:, 1])
plt.title("MDS Visualization")
plt.xlabel("MDS Dimension 1")
plt.ylabel("MDS Dimension 2")
plt.show()""", language="python")

st.image("https://raw.githubusercontent.com/naaufald/MultiDimensionalScaling_Multivariat/main/MDS 2.png")

st.markdown("berdasarkan hasil MDS, didapatkan bahwa data membentuk dua kelompok terpisah yang menunjukkan adanya pengelompokkan dalam data, dengan observasi antar kelompok menunjukkan adanya perbedaan signifikan")
st.subheader("Conclusion")
st.markdown("MDS merupakan sebuah metode yang cukup powerful untuk mereduksi dimensi serta melakukan visualisasi data kompleks, metode ini dapat membantu dalam mengungkap pola dan hubungan antar data dengan mengubah dimensi sehingga menjadi sebuah hal penting dalam sebuah analisis")