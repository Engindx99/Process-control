import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dt.dt import RotaryKilnDigitalTwin
import matplotlib.pyplot as plt

# Simülasyonu çalıştır
kiln = RotaryKilnDigitalTwin()
df = kiln.run_full_simulation(steps=3000)

# 1. Pencere - Sıcaklık
plt.figure(1, figsize=(12, 5))
plt.plot(df['Step'], df['Temperature'], 'r-', linewidth=1)
plt.axhline(y=1450, color='g', linestyle='--', label='Setpoint (1450°C)')
plt.xlabel('Adım')
plt.ylabel('Sıcaklık (°C)')
plt.title('Fırın Sıcaklığı')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Pencere - Yakıt, Fan ve O2
plt.figure(2, figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(df['Step'], df['Fuel'], 'b-', linewidth=1)
plt.ylabel('Yakıt')
plt.title('Yakıt, Fan ve O₂ Değişimi')
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 2)
plt.plot(df['Step'], df['Fan'], 'orange', linewidth=1)
plt.ylabel('Fan')
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 3)
plt.plot(df['Step'], df['O2'], 'purple', linewidth=1)
plt.xlabel('Adım')
plt.ylabel('O₂ (%)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()