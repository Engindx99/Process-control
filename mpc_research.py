import pandas as pd

# Örnek: CSV'den okuma (sen kendi kaynağına göre değiştir)
df = pd.read_csv("data/mpc_results.csv")

# İlgili kolonlar
cols = ["temp", "o2", "co2", "pressure", "fuel", "fan", "step"]

# Maksimum değerleri bul
max_values = df[cols].max()

print(max_values)