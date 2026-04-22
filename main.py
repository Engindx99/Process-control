import os
import yaml
import pandas as pd
import numpy as np
import logging
from src.dt.dt import RotaryKilnPlant
from src.mpc.mpc import MPC 
from src.filters.kalman import SelectiveKalmanFilter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_high_resolution_simulation():
    """
    6 Saatlik Simülasyon: 
    - 1sn çözünürlükte veri takibi (21600 step)
    - 10sn çözünürlükte MPC kontrolü (2160 kontrol sinyali)
    """
    
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Bileşenler
    plant = RotaryKilnPlant(seed=None) 
    obs = plant.reset() 
    
    mpc = MPC(config) 
    kf = SelectiveKalmanFilter(config) 
    
    # Simülasyon Parametreleri
    total_seconds = 21600
    control_interval = 10  # MPC her 10 saniyede bir çalışacak
    history = []
    
    # Başlangıç aksiyonları
    fuel_cmd = float(obs['fuel'])
    fan_cmd = float(obs['fan'])

    print(f"🚀 MULTIRATE SIMÜLASYON BAŞLADI")
    print(f"📊 Veri Takibi: {total_seconds} step | Kontrol Periyodu: {control_interval}s")

    for t in range(total_seconds):
        # 1. KALMAN GÜNCELLEME (Her saniye veri okuyoruz)
        kf.update(obs['o2'], obs['pressure'])
        
        # 2. MPC KONTROLÜ (Sadece 10 saniyede bir çalışır)
        if t % control_interval == 0:
            try:
                fuel_cmd, fan_cmd = mpc.get_action(
                    current_temp=obs['temp'], 
                    current_o2=obs['o2'], 
                    fuel_history=plant.history_fuel
                )
            except Exception as e:
                logger.error(f"MPC Hatası - Saniye {t}: {e}")
                break
        
        # 3. DİNAMİK MODEL ADIMI (Her saniye ilerliyor)
        # Not: plant.step fonksiyonun 1sn mi yoksa 5sn mi ilerlediği 
        # DT sınıfının içindeki dt parametresine bağlıdır. 
        obs = plant.step(fuel_cmd, fan_cmd)
        
        # 4. KAYIT
        history.append({
            'second': t,
            'temp': obs['temp'],
            'o2': obs['o2'],
            'fuel': fuel_cmd,
            'fan': fan_cmd,
            'is_control_step': (t % control_interval == 0),
            'error': abs(obs['temp'] - config['system']['setpoint'])
        })

        # Raporlama (Her 1 saatte bir)
        if t % 3600 == 0 and t > 0:
            avg_mae = np.mean([h['error'] for h in history])
            print(f"⏱️ Saat {t//3600}: Sıcaklık: {obs['temp']:.2f}°C | Kümülatif MAE: {avg_mae:.4f}")

    # Analiz ve Kayıt
    df = pd.DataFrame(history)
    print("\n" + "="*40)
    print(f"🏆 TEST TAMAMLANDI")
    print(f"📊 Final MAE: {df['error'].mean():.4f}")
    print(f"🎮 Toplam Kontrol Sinyali: {df['is_control_step'].sum()}")
    print("="*40)

    df.to_csv("simulation_21600_steps.csv", index=False)

if __name__ == "__main__":
    run_high_resolution_simulation()