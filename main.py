import os
import json
import numpy as np
import matplotlib.pyplot as plt
from digital_twin.dt import RotaryKilnDigitalTwin
from mpc.mpc import MPC

def run_simulation():
    """Simülasyonu çalıştır ve verileri kaydet"""
    print("=" * 70)
    print("MPC CONTROLLER - Yeni Simülasyon Başlatılıyor")
    print("=" * 70)
    
    plant = RotaryKilnDigitalTwin()
    mpc = MPC(horizon=20) # Gecikme için optimize edilmiş değerler
    steps = 5000
    
    for i in range(steps):
        # 🌪️ Bozucu etkiler (Disturbances)
        plant.temp += np.random.normal(0, 1.0)
        
        # Senaryo bazlı ani değişimler (Test senaryoları)
        if 500 < i < 550: plant.temp += 3
        if 1200 < i < 1250: plant.temp -= 2
        if 1600 < i < 1650: plant.temp += 2.5
        
        # 🎯 MPC Optimizasyonu
        fuel = mpc.optimize(plant)
        
        # 🏭 Proses Adımı
        temp, o2, _ = plant.step(fuel, 1000.0)
        
        # 📝 Loglama
        mpc.log_step(i, fuel, temp, o2)
        
        if i % 200 == 0:
            error = temp - mpc.setpoint
            print(f"Step {i:4d}: T={temp:6.1f}°C | Error={error:+5.1f}°C | Fuel={fuel:5.2f}")
    
    mpc.save("mpc_log.json")
    print("\n✅ Simülasyon tamamlandı ve kaydedildi.")
    return mpc.log

def plot_results(data):
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.DataFrame(data)
    
    # Gürültüyü temizlemek için hareketli ortalama
    df['temp_smooth'] = df['temp'].rolling(window=15, min_periods=1).mean()
    df['fuel_smooth'] = df['fuel'].rolling(window=10, min_periods=1).mean()

    # --- GENEL ANALİZ GRAFİĞİ ---
    plt.style.use('default') 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.patch.set_facecolor('white')

    # Sıcaklık (Üst)
    ax1.plot(df['step'], df['temp'], color='red', alpha=0.15, linewidth=0.5)
    ax1.plot(df['step'], df['temp_smooth'], color='red', linewidth=1.5, label='Furnace Temperature (°C)')
    ax1.axhline(y=1450, color='black', linestyle='--', linewidth=1.2, label='Setpoint (1450°C)')
    ax1.set_title("Rotary Klin - Thermal Analysis", fontsize=14, fontweight='bold', color='black')
    ax1.set_ylabel("Temperature (°C)", color='black')
    ax1.set_ylim(1300, 1500)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Yakıt (Alt)
    ax2.plot(df['step'], df['fuel_smooth'], color='green', linewidth=1.5, label='Fuel Flow Rate (m³/h)')
    ax2.set_title("MPC Fuel Control Signal", fontsize=13, fontweight='bold', color='black')
    ax2.set_ylabel("Fuel rate", color='black')
    ax2.set_xlabel("Step", color='black')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    log_file = "mpc_log.json"
    
    if os.path.exists(log_file):
        print(f"\n📂 {log_file} bulundu. Mevcut veriler yükleniyor...")
        with open(log_file, "r") as f:
            log_data = json.load(f)
    else:
        log_data = run_simulation()
    
    plot_results(log_data)