import pandas as pd
df = pd.read_csv("data/training_results.csv")
print(f"Ortalama Sıcaklık: {df['temp'].mean():.2f}")
print(f"Standart Sapma: {df['temp'].std():.2f}")
print(f"Maksimum Hata: {abs(df['temp'] - 1450).max():.2f}")