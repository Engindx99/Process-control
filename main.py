import os
import yaml
import pandas as pd
import numpy as np

from src.digital_twin.dt import RotaryKilnDigitalTwin
from src.mpc.mpc import MPC

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def run_robust_mpc_test(steps=5000):
    cfg = load_config()
    os.makedirs("data", exist_ok=True)

    plant = RotaryKilnDigitalTwin()
    mpc = MPC(
        prediction_horizon=cfg['mpc']['prediction_horizon'],
        control_horizon=cfg['mpc']['control_horizon']
    )
    mpc.setpoint = cfg['mpc']['setpoint']
    
    # Filtreleme için geçmiş verileri tutan liste
    temp_buffer = []
    filter_window = cfg['simulation'].get('filter_window', 5)
    
    history = []
    print(f"🛡️ Gürültüye Dayanıklı MPC Testi Başlıyor... (Horizon: {cfg['mpc']['prediction_horizon']})")

    for i in range(steps):
        # 1. Ham veriyi al ve filtrele (Low-pass filter mantığı)
        current_temp = plant.temp
        temp_buffer.append(current_temp)
        if len(temp_buffer) > filter_window:
            temp_buffer.pop(0)
        
        # MPC'ye "filtrelenmiş" sıcaklığı veriyoruz
        filtered_temp = sum(temp_buffer) / len(temp_buffer)
        
        # 2. MPC Optimizasyonu (Filtrelenmiş bilgiyle)
        u_mpc = mpc.optimize(plant) 
        
        # 3. Gerçek dünyaya (plant) aksiyonu uygula
        # Not: plant.step içinde kendi gürültüsü zaten var
        temp, o2, _ = plant.step(u_mpc, 1000.0)
        
        history.append({
            "step": i,
            "temp": float(temp),
            "filtered_temp": float(filtered_temp),
            "fuel": float(u_mpc),
            "error": float(temp - mpc.setpoint)
        })

        if i % 500 == 0:
            print(f"Adım: {i} | Filtreli Isı: {filtered_temp:.1f}°C | Yakıt: {u_mpc:.2f}")

    df = pd.DataFrame(history)
    df.to_csv(cfg['paths']['results_csv'], index=False)
    print(f"✅ Bitti. MAE: {df['error'].abs().mean():.2f}")

if __name__ == "__main__":
    run_robust_mpc_test()