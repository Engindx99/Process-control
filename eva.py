import pandas as pd
import matplotlib.pyplot as plt

# Verileri oku
hybrid = pd.read_csv("data/training_results.csv")
mpc = pd.read_csv("data/pure_mpc_results.csv")

plt.figure(figsize=(12, 6))
plt.plot(mpc['temp'], label='Saf MPC', color='gray', linestyle='--')
plt.plot(hybrid['temp'], label='Yeni Hibrit (0.96 EV)', color='blue')
plt.axhline(1450, color='red', label='HEDEF')
plt.title("Gerçek Hibrit Performansı")
plt.legend()
plt.show()