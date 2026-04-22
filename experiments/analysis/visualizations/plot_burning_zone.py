import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.burning_zone.burning_zone import RotaryKilnPlant

# --- Ayarlar ---
TOTAL_MINUTES = 360
STEPS = TOTAL_MINUTES * 60  # 1 sn adımlarla
SETPOINT_TEMP = 1450

# 1. Veri Üretimi
plant = RotaryKilnPlant(seed=None)
df = plant.run(steps=STEPS)

# 2. Hesaplamalar
t_min = (df["step"]) / 60
# Verimlilik hesapla
df["efficiency"] = df["o2"].apply(lambda x: 1.0 / (1.0 + np.exp(-(x - 2.5))) * 100)

# İLİŞKİYİ GÖRÜNÜR KILAN ADIM: Hareketli Ortalama (Smoothing)
# Son 5 dakikanın (300 saniye) ortalamasını alarak gürültüyü temizliyoruz
df["eff_smooth"] = df["efficiency"].rolling(window=300).mean()

# --- GÖRSELLEŞTİRME (TEK PANEL, ÇİFT EKSEN) ---
fig, ax1 = plt.subplots(figsize=(15, 7))

# Sol Eksen: Sıcaklık
color_temp = 'tab:red'
ax1.set_xlabel('Zaman (Dakika)', fontsize=12)
ax1.set_ylabel('Sıcaklık (°C)', color=color_temp, fontsize=12, fontweight='bold')
line1 = ax1.plot(t_min, df["temp"], color=color_temp, linewidth=1.5, label="Fırın Sıcaklığı")
ax1.axhline(y=SETPOINT_TEMP, color='black', linestyle='--', alpha=0.5)
ax1.tick_params(axis='y', labelcolor=color_temp)
ax1.set_ylim(1400, 1500) # İlişkiyi görmek için zum yapıyoruz

# Sağ Eksen: Verimlilik
ax2 = ax1.twinx() 
color_eff = 'tab:green'
ax2.set_ylabel('Düzleştirilmiş Verimlilik (%)', color=color_eff, fontsize=12, fontweight='bold')
# Orijinal veriyi çok silik arka planda, düzleştirilmiş veriyi net gösteriyoruz
ax2.plot(t_min, df["efficiency"], color=color_eff, alpha=0.1) 
line2 = ax2.plot(t_min, df["eff_smooth"], color=color_eff, linewidth=2, label="Düzleştirilmiş Verimlilik (5dk Ort)")
ax2.tick_params(axis='y', labelcolor=color_eff)
ax2.set_ylim(20, 60) # Verimlilik aralığını grafiğe göre daraltıyoruz

# Estetik Ayarlar
plt.title("Yanma Verimliliği ve Sıcaklık Arasındaki Dinamik İlişki", fontsize=14)
ax1.grid(True, linestyle=':', alpha=0.6)

# Lejant Birleştirme
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')

plt.tight_layout()
plt.show()