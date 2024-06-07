# Import Libraries
import poplib
from statistics import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Mengumpulkan Data
data = pd.read_csv('sales_data.csv')

# 2. Data Cleaning
# Mengatasi missing values
data = data.dropna()

# Mengatasi data duplikat
data = data.drop_duplicates()

# 3. Transformasi Data
# Mengubah tipe data jika diperlukan
data['Date'] = pd.to_datetime(data['Date'])

# 4. Exploratory Data Analysis (EDA)
# Melihat ringkasan data
print("Data Description:\n", data.describe())

# Visualisasi data penjualan dengan diagram garis
plt.figure(figsize=(10, 6))
sns.lineplot(x='Date', y='Sales_Amount', data=data)
plt.title('Sales Amount Over Time')
plt.xlabel('Date')
plt.ylabel('Sales Amount')
plt.show()

# Visualisasi data penjualan per produk dengan diagram batang
plt.figure(figsize=(10, 6))
sns.barplot(x='Product', y='Sales_Amount', data=data)
plt.title('Sales Amount by Product')
plt.xlabel('Product')
plt.ylabel('Sales Amount')
plt.show()

# Scatter plot antara Units Sold dan Sales Amount
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Units_Sold', y='Sales_Amount', data=data)
plt.title('Units Sold vs Sales Amount')
plt.xlabel('Units Sold')
plt.ylabel('Sales Amount')
plt.show()

# 5. Pemodelan Data
# Menyiapkan data untuk model
X = data[['Units_Sold']]
y = data['Sales_Amount']

# Membagi data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # type: ignore

# Membuat dan melatih model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Validasi dan Tuning Model
# Memprediksi data testing
y_pred = model.predict(X_test)

# Mengukur performa model
mse = mean_squared_error(y_test, y_pred) # type: ignore
r2 = r2_score(y_test, y_pred) # type: ignore
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# 7. Interpretasi dan Penyajian Hasil
# Menampilkan koefisien model
print(f'Coefficients: {model.coef_}')
print(f'Intercept: {model.intercept_}')

# Visualisasi hasil prediksi
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='black', label='Actual')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Predicted')
plt.title('Actual vs Predicted Sales Amount')
plt.xlabel('Units Sold')
plt.ylabel('Sales Amount')
plt.legend()
plt.show()

# 8. Deployment dan Monitoring
# Simpan model untuk digunakan di produksi
poplib.dump(model, 'sales_model.pkl')

# 9. Maintenance dan Iterasi
# Melakukan iterasi berdasarkan feedback dan data baru
# (ini tergantung pada konteks aplikasi dan lingkungan produksi)
