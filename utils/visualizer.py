import matplotlib.pyplot as plt
import numpy as np
import os

class KilnVisualizer:
    def __init__(self, config):
        self.cfg = config
        plt.style.use('ggplot')

    def plot_state(self, state, current_time):
        # 3 Panelli grafik (Sıcaklık, Kalsinasyon, Mineraller)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        x = np.linspace(0, self.cfg.length, self.cfg.n_zones)
        
        # --- 1. Panel: Sıcaklık Profili ---
        ax1.plot(x, state.T_gas, label='Gaz Sıcaklığı (K)', color='crimson', linewidth=2, alpha=0.8)
        ax1.plot(x, state.T_solid, label='Katı Sıcaklığı (K)', color='darkorange', linewidth=2.5)
        ax1.set_ylabel('Sıcaklık (K)')
        ax1.set_title(f'Rotary Kiln Dijital İkiz Analizi (Zaman: {current_time:.1f}s)')
        ax1.legend(loc='upper left')
        ax1.grid(True)

        # --- 2. Panel: Kalsinasyon (CaCO3 -> CaO) ---
        ax2.plot(x, state.c_caco3 * 100, label='CaCO3 %', color='forestgreen', linewidth=2)
        ax2.plot(x, state.c_cao * 100, label='Serbest Kireç (CaO) %', color='gold', linestyle='--', linewidth=1.5)
        ax2.set_ylabel('Konsantrasyon (%)')
        ax2.legend(loc='center left')
        ax2.grid(True)

        # --- 3. Panel: Sinterleme / Klinker Mineralleri ---
        ax3.plot(x, state.c_c2s * 100, label='C2S (Belit) %', color='royalblue', linewidth=2)
        ax3.plot(x, state.c_c3s * 100, label='C3S (Alit) %', color='purple', linewidth=2.5)
        ax3.set_ylabel('Mineral Oranı (%)')
        ax3.set_xlabel('Fırın Uzunluğu (m)')
        ax3.set_ylim(-5, 105) # Okunabilirlik için
        ax3.legend(loc='upper left')
        ax3.grid(True)

        plt.tight_layout()
        
        # Çıktıyı otomatik kaydetmek için opsiyonel
        if not os.path.exists('data/plots'):
            os.makedirs('data/plots')
        plt.savefig(f'data/plots/kiln_state_{int(current_time)}.png')
        
        plt.show()