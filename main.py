import os
import yaml
import pandas as pd
import numpy as np
import logging
from src.dt.dt import RotaryKilnPlant
from src.mpc.mpc import MPC 
from src.filters.kalman import SelectiveKalmanFilter

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_production_simulation():
    """
    Optuna'dan elde edilen en iyi parametrelerle 
    fırını gerçek zamanlı simülasyon modunda çalıştırır.
    """
    
    # 1. Güncel Config'i Yükle (Trial 24 Değerleri İçeride)
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 2. Bileşenleri Başlat
    plant = RotaryKilnPlant(seed=None) # Test için gürültü açık
    obs = plant.reset() 
    
    mpc = MPC(config) 
    # Başlangıç değerlerini eşitle
    mpc.last_fuel = float(obs['fuel'])
    mpc.last_fan = float(obs['fan'])
    
    kf = SelectiveKalmanFilter(config) 
    
    # Simülasyon Ayarları (120 Dakika = 1440 Step)
    total_steps = 1440  
    history = []
    
    print(f"🚀 Fırın 'Altın Reçete' ile Yayına Alındı | Hedef: {config['system']['setpoint']}°C")

    # 3. Ana Döngü
    for i in range(total_steps):
        # Filtreleme
        kf.update(obs['o2'], obs['pressure'])
        
        try:
            # MPC Karar Mekanizması (Trial 24 Ağırlıkları Kullanılıyor)
            fuel_cmd, fan_cmd = mpc.get_action(
                current_temp=obs['temp'], 
                current_o2=obs['o2'], 
                fuel_history=plant.history_fuel
            ) 
        except Exception as e:
            logger.error(f"Kritik Hata - Step {i}: {e}")
            break
            
        # Dünyayı (Plant) Bir Adım İlerlet
        obs = plant.step(fuel_cmd, fan_cmd)
        
        # Verileri Kaydet
        history.append({
            'step': i,
            'temp': obs['temp'],
            'o2': obs['o2'],
            'fuel': fuel_cmd,
            'fan': fan_cmd,
            'error': abs(obs['temp'] - config['system']['setpoint'])
        })

        # Her 10 dakikada bir (120 step) durum raporu
        if i % 120 == 0:
            current_mae = np.mean([h['error'] for h in history])
            print(f"⏱️ Dakika {i//12}: Sıcaklık: {obs['temp']:.2f}°C | MAE: {current_mae:.4f}")

    # 4. Performans Analizi ve Raporlama
    df = pd.DataFrame(history)
    final_mae = df['error'].mean()
    
    print("\n" + "="*30)
    print(f"🏆 SİMÜLASYON TAMAMLANDI")
    print(f"📊 Ortalama Hata (MAE): {final_mae:.4f}")
    print(f"🔥 Yakıt Ortalaması: {df['fuel'].mean():.2f}")
    print(f"💨 Fan Ortalaması: {df['fan'].mean():.2f}")
    print("="*30)

    # Sonuçları CSV olarak kaydet (Analiz için)
    df.to_csv("simulation_results_v4.csv", index=False)
    print("💾 Sonuçlar 'simulation_results_v4.csv' dosyasına kaydedildi.")

if __name__ == "__main__":
    run_production_simulation()