import pandas as pd
import numpy as np
import traceback
import os
from simulation.config import KilnConfig
from models.kiln import KilnModel
from utils.visualizer import KilnVisualizer
from control.casadi_mpc import KilnCasadiMPC

def run_simulation(config_path="config.yaml", show_plots=False, verbose=False): # verbose eklendi
    """
    Optuna ve manuel testler için simülasyonu koşturan ana motor.
    verbose=False yapıldığında terminale log basmaz.
    """
    try:
        cfg = KilnConfig.from_yaml(config_path)
    except Exception as e:
        if verbose: print(f"[HATA] Config yüklenemedi: {e}", flush=True)
        return 0.0, 0.0, 1e6

    model = KilnModel(cfg)
    mpc = KilnCasadiMPC(cfg)
    
    save_interval = getattr(cfg, 'save_interval', 1000)
    flame_changes = []
    prev_flame = None

    try:
        for step in range(cfg.simulation_steps):
            # MPC Kontrol Adımı
            if step % cfg.control_interval == 0:
                state_vec = [
                    model.state.T_solid[-1], 
                    model.state.c_caco3[-1], 
                    model.state.c_cao[-1],   
                    model.state.c_c2s[-1],   
                    model.state.c_c3s[-1]    
                ]
                try:
                    optimized_fuel_temp = mpc.compute_control(state_vec)
                    model.state.T_gas[-1] = optimized_fuel_temp 
                    if prev_flame is not None:
                        flame_changes.append(abs(optimized_fuel_temp - prev_flame))
                    prev_flame = optimized_fuel_temp
                except:
                    pass

            model.step()
            
            # LOGLAMA BURADA KONTROL EDİLİYOR
            if verbose and step % save_interval == 0:
                s = model.state
                calc_pct = (1.0 - s.c_caco3[-1]) * 100
                c3s_val = s.c_c3s[-1] * 100
                print(f"Adım: {step:>6} | Zaman: {model.time:>7.1f}s | T_çıkış: {s.T_solid[-1]:>6.1f}K | "
                      f"Kals: %{calc_pct:>6.2f} | C3S: %{c3s_val:>8.6f} | Alev: {s.T_gas[-1]:>6.1f}K", flush=True)

        final_s = model.state
        return float(final_s.c_c3s[-1]), float(final_s.c_c2s[-1]), (np.mean(flame_changes) if flame_changes else 0)

    except Exception as e:
        if verbose: traceback.print_exc()
        return 0.0, 0.0, 1e6

def main():
    # Bu kısım sadece python main.py yazarsan çalışır
    print("--- Manuel Simülasyon Başlatılıyor ---", flush=True)
    c3s, c2s, penalty = run_simulation("config.yaml", show_plots=True)
    print(f"\n--- SONUÇ ---")
    print(f"Final C3S: %{c3s*100:.4f} | Final C2S: %{c2s*100:.4f} | Stabilite Cezası: {penalty:.2f}", flush=True)

if __name__ == "__main__":
    main()