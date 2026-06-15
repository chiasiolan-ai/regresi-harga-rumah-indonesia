# ============================================================
#   ANALISIS REGRESI BERGANDA - HARGA RUMAH DI INDONESIA
#   Dataset: DATA RUMAH Jakarta Selatan (Kaggle)
#   Kolom   : NO, NAMA RUMAH, HARGA, LB, LT, KT, KM, GRS
#   Target  : HARGA (Rupiah) → prediksi harga rumah
# ============================================================

# ─────────────────────────────────────────────
# LANGKAH 1 – Import Library
# ─────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  REGRESI BERGANDA – HARGA RUMAH INDONESIA")
print("=" * 60)

# ─────────────────────────────────────────────
# LANGKAH 2 – Load Dataset
# ─────────────────────────────────────────────
# Pastikan file DATA_RUMAH.csv ada di folder yang sama
df = pd.read_csv("DATA_RUMAH.csv")
print(f"\n✅ Dataset berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")

# ─────────────────────────────────────────────
# LANGKAH 3 – Eksplorasi Data (EDA)
# ─────────────────────────────────────────────
print("\n" + "─" * 50)
print("LANGKAH 3 – Eksplorasi Data")
print("─" * 50)

print("\n📌 5 Baris Pertama:")
print(df.head())

print("\n📌 Statistik Deskriptif:")
print(df[['HARGA','LB','LT','KT','KM','GRS']].describe().round(2))

print("\n📌 Missing Values:")
print(df.isnull().sum())

# ─────────────────────────────────────────────
# LANGKAH 4 – Preprocessing
# ─────────────────────────────────────────────
print("\n" + "─" * 50)
print("LANGKAH 4 – Preprocessing Data")
print("─" * 50)

# Hapus baris kosong
df_clean = df.dropna().copy()

# Konversi HARGA ke Juta Rupiah supaya angkanya lebih mudah dibaca
df_clean['HARGA_JUTA'] = df_clean['HARGA'] / 1_000_000

# Fitur (X) dan Target (y)
kolom_fitur = ['LB', 'LT', 'KT', 'KM', 'GRS']   # 5 parameter
kolom_target = 'HARGA_JUTA'

X = df_clean[kolom_fitur]
y = df_clean[kolom_target]

print(f"✅ Data setelah dibersihkan : {len(df_clean)} baris")
print(f"✅ Fitur (5 parameter)      : {kolom_fitur}")
print(f"✅ Target                   : Harga (Juta Rupiah)")

# ─────────────────────────────────────────────
# LANGKAH 5 – Split Data Training & Testing
# ─────────────────────────────────────────────
print("\n" + "─" * 50)
print("LANGKAH 5 – Split Data (80% Train / 20% Test)")
print("─" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"✅ Data Training : {len(X_train)} baris")
print(f"✅ Data Testing  : {len(X_test)} baris")

# ─────────────────────────────────────────────
# LANGKAH 6 – Training Model Regresi Berganda
# ─────────────────────────────────────────────
print("\n" + "─" * 50)
print("LANGKAH 6 – Training Model Regresi Berganda")
print("─" * 50)

model = LinearRegression()
model.fit(X_train, y_train)
print("✅ Model berhasil dilatih!")

print("\n📌 Persamaan Regresi:")
persamaan = "HARGA = {:.2f}".format(model.intercept_)
for fitur, koef in zip(kolom_fitur, model.coef_):
    tanda = "+" if koef >= 0 else "-"
    persamaan += f" {tanda} {abs(koef):.2f}×{fitur}"
print(f"   {persamaan}")

print("\n📌 Detail Koefisien:")
for fitur, koef in zip(kolom_fitur, model.coef_):
    print(f"   {fitur:>4} : {koef:+.4f} juta")
print(f"   Intercept : {model.intercept_:.4f} juta")

# ─────────────────────────────────────────────
# LANGKAH 7 – Evaluasi Model
# ─────────────────────────────────────────────
print("\n" + "─" * 50)
print("LANGKAH 7 – Evaluasi Model")
print("─" * 50)

y_pred = model.predict(X_test)

r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n📊 Hasil Evaluasi pada Data Testing:")
print(f"   R² Score : {r2:.4f}  → model menjelaskan {r2*100:.2f}% variasi harga")
print(f"   MAE      : Rp {mae:,.0f} Juta  → rata-rata selisih prediksi vs aktual")
print(f"   RMSE     : Rp {rmse:,.0f} Juta")

if r2 >= 0.75:
    print("\n✅ Model BAIK – R² ≥ 0.75")
elif r2 >= 0.50:
    print("\n⚠️  Model CUKUP – R² antara 0.50–0.75")
else:
    print("\n❌ Model KURANG BAIK – R² < 0.50")

# ─────────────────────────────────────────────
# LANGKAH 8 – Visualisasi
# ─────────────────────────────────────────────
print("\n" + "─" * 50)
print("LANGKAH 8 – Visualisasi")
print("─" * 50)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Analisis Regresi Berganda – Harga Rumah Jakarta Selatan",
             fontsize=14, fontweight="bold")

# Plot 1 – Actual vs Predicted
ax1 = axes[0, 0]
ax1.scatter(y_test, y_pred, alpha=0.5, color="#2196F3", edgecolors="white", linewidth=0.4)
mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
ax1.plot([mn, mx], [mn, mx], "r--", lw=2, label="Ideal (y=x)")
ax1.set_xlabel("Harga Aktual (Juta Rp)")
ax1.set_ylabel("Harga Prediksi (Juta Rp)")
ax1.set_title(f"Actual vs Predicted  |  R²={r2:.4f}")
ax1.legend(); ax1.grid(True, alpha=0.3)

# Plot 2 – Residual Plot
ax2 = axes[0, 1]
residuals = y_test - y_pred
ax2.scatter(y_pred, residuals, alpha=0.5, color="#FF5722", edgecolors="white", linewidth=0.4)
ax2.axhline(0, color="black", linestyle="--", lw=2)
ax2.set_xlabel("Harga Prediksi (Juta Rp)")
ax2.set_ylabel("Residual")
ax2.set_title("Residual Plot")
ax2.grid(True, alpha=0.3)

# Plot 3 – Distribusi Residual
ax3 = axes[1, 0]
ax3.hist(residuals, bins=30, color="#4CAF50", edgecolor="white", alpha=0.85)
ax3.axvline(0, color="red", linestyle="--", lw=2)
ax3.set_xlabel("Residual")
ax3.set_ylabel("Frekuensi")
ax3.set_title("Distribusi Residual")
ax3.grid(True, alpha=0.3)

# Plot 4 – Koefisien Fitur
ax4 = axes[1, 1]
colors = ["#9C27B0","#E91E63","#FF9800","#00BCD4","#8BC34A"]
bars = ax4.barh(kolom_fitur, np.abs(model.coef_), color=colors, edgecolor="white")
ax4.set_xlabel("|Koefisien|")
ax4.set_title("Pengaruh Tiap Fitur terhadap Harga")
for bar, val in zip(bars, np.abs(model.coef_)):
    ax4.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
             f"{val:.1f}", va="center", fontsize=9)
ax4.grid(True, alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig("hasil_regresi_harga_rumah.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Grafik disimpan: hasil_regresi_harga_rumah.png")

# ─────────────────────────────────────────────
# LANGKAH 9 – Prediksi Rumah Baru
# ─────────────────────────────────────────────
print("\n" + "─" * 50)
print("LANGKAH 9 – Contoh Prediksi Rumah Baru")
print("─" * 50)

rumah_baru = pd.DataFrame({
    'LB' : [120],   # Luas Bangunan m²
    'LT' : [200],   # Luas Tanah m²
    'KT' : [3],     # Kamar Tidur
    'KM' : [2],     # Kamar Mandi
    'GRS': [1],     # Garasi
})

harga_pred = model.predict(rumah_baru)[0]
print(f"\n🏠 Spesifikasi Rumah:")
print(f"   Luas Bangunan : 120 m²  |  Luas Tanah: 200 m²")
print(f"   Kamar Tidur   : 3       |  Kamar Mandi: 2  |  Garasi: 1")
print(f"\n💰 Prediksi Harga : Rp {harga_pred:,.0f} Juta")
print(f"                  = Rp {harga_pred*1_000_000:,.0f}")

print("\n" + "=" * 60)
print("  SELESAI!")
print("=" * 60)
