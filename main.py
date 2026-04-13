import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from digital_twin.dt import RotaryKilnDigitalTwin
from mpc.mpc import MPC

def calculate_metrics(log_data):
    """Simülasyon sonuçlarını analiz eder ve performans metriklerini hesaplar."""
    temps = np.array([step['temp'] for step in log_data])
    setpoints = np.array([step['setpoint'] for step in log_data])
    errors = temps - setpoints
    
    mse = np.mean(errors**2)
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(mse)
    max_error = np.max(np.abs(errors))
    
    return {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "Max Error": max_error,
        "errors": errors
    }

def run_simulation():
    """Simülasyonu çalıştır ve verileri kaydet"""
    print("=" * 70)
    print("MPC CONTROLLER - Yeni Simülasyon Başlatılıyor")
    print("=" * 70)
    
    plant = RotaryKilnDigitalTwin()
    mpc = MPC(prediction_horizon=30, control_horizon=3) 
    steps = 5000
    
    for i in range(steps):
        # Bozucu etkiler
        plant.temp += np.random.normal(0, 1.0)
        if 500 < i < 550: plant.temp += 3
        if 1200 < i < 1250: plant.temp -= 2
        if 1600 < i < 1650: plant.temp += 2.5
        
        fuel = mpc.optimize(plant)
        temp, o2, _ = plant.step(fuel, 1000.0)
        mpc.log_step(i, fuel, temp, o2)
        
        if i % 500 == 0:
            print(f"Step {i:4d}: T={temp:6.1f}°C | Fuel={fuel:5.2f}")
    
    mpc.save("mpc_log.json")
    return mpc.log

def plot_results(data, metrics):
    df = pd.DataFrame(data)
    df['error'] = metrics['errors']
    df['temp_smooth'] = df['temp'].rolling(window=15, min_periods=1).mean()

    plt.style.use('default') 
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.patch.set_facecolor('white')

    # 1. Grafik: Termal Analiz (Üstte)
    ax1.plot(df['step'], df['temp'], color='red', alpha=0.15, linewidth=0.5)
    ax1.plot(df['step'], df['temp_smooth'], color='red', linewidth=1.5, label='Fırın Sıcaklığı (°C)')
    ax1.axhline(y=1450, color='black', linestyle='--', linewidth=1.2, label='Setpoint (1450°C)')
    ax1.set_title("Rotary Kiln - Thermal Analysis", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Temperature (°C)")
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # 2. Grafik: Yakıt Kontrolü (Ortada)
    ax2.plot(df['step'], df['fuel'], color='green', linewidth=1.2, label='Fuel Flow Rate (m³/h)')
    ax2.set_title("MPC Fuel Control Signal", fontsize=13, fontweight='bold')
    ax2.set_ylabel("Fuel Rate")
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle=':', alpha=0.6)

    # 3. Grafik: Kontrol Hatası (En Altta)
    ax3.fill_between(df['step'], df['error'], color='darkorange', alpha=0.4, label='Error Value (°C)')
    ax3.plot(df['step'], df['error'], color='darkorange', linewidth=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1.0)
    ax3.set_title("Control Error", fontsize=13, fontweight='bold')
    ax3.set_ylabel("Error (°C)")
    ax3.set_xlabel("Step")
    ax3.legend(loc='upper right')
    ax3.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    log_file = "mpc_log.json"
    
    if os.path.exists(log_file):
        print(f"\n📂 {log_file} yüklendi, analiz yapılıyor...")
        with open(log_file, "r") as f:
            log_data = json.load(f)
    else:
        log_data = run_simulation()
    
    metrics = calculate_metrics(log_data)
    
    print("\n" + "="*40)
    print(f"📊 MPC PERFORMANCE REPORT")
    print("-" * 40)
    print(f"MAE:  {metrics['MAE']:.4f} °C")
    print(f"RMSE: {metrics['RMSE']:.4f} °C")
    print(f"Max:  {metrics['Max Error']:.4f} °C")
    print("="*40)
    
    plot_results(log_data, metrics)