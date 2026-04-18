import sys
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Dosya yollarını ayarla
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dt.dt import RotaryKilnDigitalTwin

# --- ZAMAN ÖLÇEĞİ (KURAL DEĞİL, SADECE DÖNÜŞÜM) ---
STEP_DURATION_SEC = 5 

# 1. Simülasyonu çalıştır
kiln = RotaryKilnDigitalTwin()
df = kiln.run_full_simulation(steps=3000)

# Adımları sadece 5 saniye üzerinden dakikaya ölçekle
df['Minutes'] = df['Step'] * STEP_DURATION_SEC / 60

# ---------------------------------------------------------
# 1. Pencere - Sıcaklık Analizi
# ---------------------------------------------------------
plt.figure(1, figsize=(12, 6))
plt.plot(df['Minutes'], df['Temperature'], color='#d62728', linewidth=1.5, label='Fırın Sıcaklığı')
plt.axhline(y=1450, color='g', linestyle='--', label='Setpoint (1450°C)')

plt.xlabel('Zaman (Dakika)')
plt.ylabel('Sıcaklık (°C)')
plt.title('Digital Twin: Fırın Isıl Karakteristik Analizi')
plt.legend()
plt.grid(True, alpha=0.3)

# ---------------------------------------------------------
# 2. Pencere - Yakıt, Fan ve O2 (Endüstriyel Renkler)
# ---------------------------------------------------------
plt.figure(2, figsize=(12, 10))

# YAKIT - O meşhur Kehribar Sarısı (#FFBF00)
plt.subplot(3, 1, 1)
plt.plot(df['Minutes'], df['Fuel'], color='#FFBF00', linewidth=1.8, label='Yakıt Akışı')
plt.ylabel('Yakıt')
plt.title('Operasyonel Değişkenlerin Zaman Serisi (Dakika Bazlı)')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right')

# FAN
plt.subplot(3, 1, 2)
plt.plot(df['Minutes'], df['Fan'], color='#17becf', linewidth=1.2, label='Fan Hızı (RPM)')
plt.ylabel('Fan')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right')

# O2
plt.subplot(3, 1, 3)
plt.plot(df['Minutes'], df['O2'], color='#9467bd', linewidth=1.2, label='O₂ Seviyesi (%)')
plt.xlabel('Zaman (Dakika)')
plt.ylabel('O₂ (%)')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right')

plt.tight_layout()

# ---------------------------------------------------------
# 3. Pencere - Korelasyon Matrisi (Tam Kare)
# ---------------------------------------------------------
plt.figure(3, figsize=(10, 8))
corr_columns = ['Fuel', 'Fan', 'Temperature', 'O2']
corr = df[corr_columns].corr()

sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", 
            square=True, linewidths=.5, cbar_kws={"shrink": .7})

plt.title('Değişken Etkileşim Analizi')
plt.xticks(rotation=45)
plt.yticks(rotation=0)

plt.show()

# --- Özet Çıktı ---
print(f"Toplam Simülasyon Süresi: {df['Minutes'].max():.2f} Dakika")
print(f"Son Sıcaklık Değeri: {df['Temperature'].iloc[-1]:.2f}°C")