import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
from src.burning_zone.burning_zone import RotaryKilnPlant
from src.filters.kalman import SelectiveKalmanFilter

def run_test():
    # 1. Config ve Sistem Kurulumu
    if not os.path.exists("config.yaml"):
        print("Hata: config.yaml bulunamadı!")
        return

    # DÜZELTME 1: encoding="utf-8" eklendi
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    plant = RotaryKilnPlant(seed=42)
    kf = SelectiveKalmanFilter(config)
    obs = plant.reset()

    step_duration = config["system"].get("step_duration_sec", 5)
    total_minutes = 1440 
    total_steps = int((total_minutes * 60) / step_duration)

    data = {'raw_o2': [], 'filt_o2': [], 'raw_p': [], 'filt_p': [], 'time_min': []}
    
    print(f"Test Başlatılıyor: {total_steps} adım ({total_minutes} dakika)...")

    # 2. Simülasyon Döngüsü
    for i in range(total_steps):
        new_obs = plant.step(fuel_cmd=18.0, fan_cmd=950.0) 
        
        # DÜZELTME 2: İsim uyuşmazlığı düzeltildi (update -> filter)
        # SelectiveKalmanFilter içindeki metot adı 'filter'dır.
        filt_o2, filt_p = kf.filter(new_obs['o2'], new_obs['pressure'])
        
        # Veri Toplama
        data['raw_o2'].append(new_obs['o2'])
        data['filt_o2'].append(filt_o2)
        data['raw_p'].append(new_obs['pressure'])
        data['filt_p'].append(filt_p)
        
        data['time_min'].append((i * step_duration) / 60)

    # 3. Görselleştirme (Grafik ayarları aynı kalabilir)
    # ... (Geri kalan plt kodların)

    # 3. Görselleştirme
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # O2 Grafiği
    ax1.plot(data['time_min'], data['raw_o2'], label="Ham O2 (Gürültülü)", alpha=0.3, color='red', lw=1.3)
    ax1.plot(data['time_min'], data['filt_o2'], label="Kalman Filtreli O2", color='blue', lw=1.3)
    ax1.set_title("Oksijen (O2) Filtreleme Performansı (24 Saat)")
    ax1.set_ylabel("O2 (%)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')

    # Basınç Grafiği
    ax2.plot(data['time_min'], data['raw_p'], label="Ham Basınç (Gürültülü)", alpha=0.3, color='orange', lw=1.3)
    ax2.plot(data['time_min'], data['filt_p'], label="Kalman Filtreli Basınç", color='green', lw=1.3)
    ax2.set_title("Basınç (Draft Pressure) Filtreleme Performansı (24 Saat)")
    ax2.set_ylabel("Basınç (Pa)")
    ax2.set_xlabel("Zaman (Dakika)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_test()