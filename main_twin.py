import yaml
import pandas as pd
import numpy as np
import logging
from src.dt.dt import RotaryKilnPlant
from src.mpc.mpc import MPC 
from src.filters.kalman import SelectiveKalmanFilter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_twin_engine():
    # 1. Config Yükle
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 2. Bileşenleri Başlat
    plant = RotaryKilnPlant(seed=None) 
    obs = plant.reset() 
    mpc = MPC(config) 
    kf = SelectiveKalmanFilter(config) 
    
    # Zaman Tanımları
    TOTAL_SECONDS = 21600   # 6 Saatlik veri takibi
    CONTROL_INTERVAL = 10   # 10 saniyede bir MPC kararı
    SETPOINT = config['system']['setpoint']
    
    # İlk durum
    fuel_cmd = float(obs['fuel'])
    fan_cmd = float(obs['fan'])
    history = []

    print(f"🚀 Dijital İkiz Motoru Başlatıldı...")
    print(f"⏱️ Hedef: {TOTAL_SECONDS} Saniye | Kontrol: {TOTAL_SECONDS // CONTROL_INTERVAL} Karar")

    for t in range(TOTAL_SECONDS):
        # Kalman Güncelleme (Her saniye)
        kf.update(obs['o2'], obs['pressure'])
        
        # MPC Kontrolü (10 saniyede bir)
        if t % CONTROL_INTERVAL == 0:
            try:
                fuel_cmd, fan_cmd = mpc.get_action(
                    current_temp=obs['temp'], 
                    current_o2=obs['o2'], 
                    fuel_history=plant.history_fuel
                )
            except Exception as e:
                print(f"Hata: {e}")
                break
        
        # Fırın Adımı (Her saniye)
        obs = plant.step(fuel_cmd, fan_cmd)
        
        # Veri Kaydı
        current_error = abs(obs['temp'] - SETPOINT)
        history.append({
            'second': t,
            'temp': obs['temp'],
            'o2': obs['o2'],
            'fuel': fuel_cmd,
            'fan': fan_cmd,
            'error': current_error
        })

        # --- ENTEGRE RAPORLAMA ---
        if t % 3600 == 0 and t > 0:
            # Tüm zamanların MAE ortalaması
            cum_mae = np.mean([h['error'] for h in history])
            # Son 10 dakikanın (600 saniye) MAE ortalaması (Stabiliteyi ölçmek için)
            recent_mae = np.mean([h['error'] for h in history[-600:]])
            
            print(f"✅ Saat {t//3600} Tamamlandı | Sıcaklık: {obs['temp']:.2f}°C | Kümülatif MAE: {cum_mae:.4f} | Son 10dk MAE: {recent_mae:.4f}")

    # CSV Kaydet
    df = pd.DataFrame(history)
    df.to_csv("kiln_6h_results.csv", index=False)
    print("\n" + "="*50)
    print(f"🏆 SİMÜLASYON TAMAMLANDI")
    print(f"📊 Final Genel MAE: {df['error'].mean():.4f}")
    print(f"💾 Veriler 'kiln_6h_results.csv' dosyasına kaydedildi.")
    print("="*50)

if __name__ == "__main__":
    run_twin_engine()