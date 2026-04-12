"""
Ana Simülasyon Dosyası
MPC ile Döner Fırın Kontrolü - Sabit Setpoint: 1450°C
"""

from digital_twin.dt import RotaryKilnDigitalTwin
from mpc.mpc import MPC
import matplotlib.pyplot as plt
import numpy as np
import json


def run_simulation():
    """Simülasyonu çalıştır - Sabit setpoint 1450°C"""
    
    print("=" * 70)
    print("MPC CONTROLLER - Döner Fırın Simülasyonu")
    print(f"SABİT SETPOINT: 1450°C")
    print("=" * 70)
    
    # Başlat
    plant = RotaryKilnDigitalTwin()
    mpc = MPC(horizon=10, n_candidates=20)
    
    steps = 2000
    
    print("\nSimülasyon çalışıyor...")
    print("-" * 70)
    
    for i in range(steps):
        
        # 🌪️ BOZUCU ETKİLER
        disturbance = np.random.normal(0, 1.0)
        plant.temp += disturbance
        
        # Ani bozucu etkiler
        if 500 < i < 550:
            plant.temp += 3  # Isı kaybı
        if 1200 < i < 1250:
            plant.temp -= 2  # Aşırı ısınma
        if 1600 < i < 1650:
            plant.temp += 2.5  # Hammadde nem değişimi
        
        # 🎯 MPC optimizasyonu (SADECE plant - setpoint YOK!)
        fuel = mpc.optimize(plant)  # ✅ DÜZELTİLDİ
        
        # 🏭 Proses adımı
        temp, o2, _ = plant.step(fuel, 1000.0)
        
        # 📝 Loglama (setpoint otomatik eklenir)
        mpc.log_step(i, fuel, temp, o2)
        
        # 📊 Durum gösterimi (her 200 adımda bir)
        if i % 200 == 0:
            error = temp - mpc.setpoint
            print(f"Step {i:4d}: T={temp:6.1f}°C | SP={mpc.setpoint}°C | Error={error:+5.1f}°C | Fuel={fuel:5.2f}")
    
    print("-" * 70)
    print("Simülasyon tamamlandı!")
    mpc.save("mpc_log.json")
    print("Log kaydedildi: mpc_log.json")
    
    return mpc.log


def plot_results(data):
    """Grafikleri çiz"""
    
    steps = [d["step"] for d in data]
    temps = [d["temp"] for d in data]
    sps = [d["setpoint"] for d in data]
    fuels = [d["fuel"] for d in data]
    o2s = [d["o2"] for d in data]
    
    setpoint_value = sps[0] if sps else 1450
    
    # Style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # GRAFİK 1: Sıcaklık Takibi
    plt.figure(figsize=(14, 6))
    plt.plot(steps, temps, 'b-', label='Process Temperature', linewidth=1.5, alpha=0.8)
    plt.axhline(y=setpoint_value, color='r', linestyle='--', 
                label=f'Setpoint = {setpoint_value}°C', linewidth=2)
    plt.fill_between(steps, setpoint_value-2, setpoint_value+2, 
                      color='red', alpha=0.1, label='±2°C Band')
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Temperature (°C)', fontsize=12)
    plt.title(f'Dynamic MPC Tracking - Setpoint: {setpoint_value}°C', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(1430, 1480)
    plt.tight_layout()
    plt.show()
    
    # GRAFİK 2: Yakıt Kontrol Sinyali
    plt.figure(figsize=(14, 4))
    plt.plot(steps, fuels, 'g-', linewidth=1.5)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Fuel Flow Rate', fontsize=12)
    plt.title('MPC Control Signal (Adaptive Fuel)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim(12, 22)
    plt.tight_layout()
    plt.show()
    
    # GRAFİK 3: Takip Hatası
    plt.figure(figsize=(14, 4))
    error = np.array(temps) - setpoint_value
    plt.plot(steps, error, 'purple', linewidth=1)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Zero Error')
    plt.axhline(y=2, color='gray', linestyle=':', alpha=0.5)
    plt.axhline(y=-2, color='gray', linestyle=':', alpha=0.5)
    plt.fill_between(steps, -2, 2, alpha=0.1, color='green', label='Acceptable Band (±2°C)')
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Error (°C)', fontsize=12)
    plt.title('Tracking Error - MPC Performance', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # GRAFİK 4: Oksijen Seviyesi
    plt.figure(figsize=(14, 4))
    plt.plot(steps, o2s, 'orange', linewidth=1.5)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Oxygen Level (%)', fontsize=12)
    plt.title('Oxygen Level Monitoring', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim(1, 4)
    plt.tight_layout()
    plt.show()
    
    # PERFORMANS METRİKLERİ
    print("\n" + "=" * 70)
    print(f"PERFORMANS ÖZETİ - Setpoint: {setpoint_value}°C")
    print("=" * 70)
    print(f"📊 Ortalama Mutlak Hata (MAE):     {np.mean(np.abs(error)):.2f}°C")
    print(f"📈 Maksimum Mutlak Hata:           {np.max(np.abs(error)):.2f}°C")
    print(f"📉 Root Mean Square Error (RMSE):  {np.sqrt(np.mean(error**2)):.2f}°C")
    print(f"🎯 Standart Sapma:                 {np.std(error):.2f}°C")
    print(f"⛽ Yakıt Aralığı:                  [{min(fuels):.2f}, {max(fuels):.2f}]")
    print(f"🔥 Ortalama Yakıt Tüketimi:        {np.mean(fuels):.2f}")
    print(f"💨 Oksijen Aralığı:                [{min(o2s):.2f}, {max(o2s):.2f}]%")
    print("=" * 70)


# =========================================================
# ANA PROGRAM
# =========================================================
if __name__ == "__main__":
    # Simülasyonu çalıştır
    log_data = run_simulation()
    
    # Grafikleri çiz
    plot_results(log_data)
    
    print("\n✅ Tüm işlemler tamamlandı!")