import numpy as np

class KalmanFilter:
    def __init__(self, config):
        """
        Gelişmiş Selective Kalman Filter: O2 ve Basınç sinyallerini normalize eder ve temizler.
        """
        # Durumlar: [o2, pressure]
        self.x = np.array([2.5, -2.0], dtype=float) 
        n = len(self.x)
        
        # P: İlk güven seviyesi (Düşük değer = Başlangıç tahminine yüksek güven)
        self.P = np.eye(n) * 0.1
        
        # Q: Süreç Gürültüsü (Fırın dinamiği ne kadar hızlı değişebilir?)
        # O2 daha yavaş (0.0001), Basınç daha hızlı (0.001) değişebilir.
        q_o2 = config.get('kalman', {}).get('q_o2', 0.0001)
        q_press = config.get('kalman', {}).get('q_press', 0.001)
        self.Q = np.diag([q_o2, q_press])
        
        # R: Ölçüm Gürültüsü (Sensör kalitesizliği)
        # 0.05 - 0.1 arası değerler sensördeki gürültüyü %80 oranında bastırır.
        r_val = config.get('kalman', {}).get('measurement_noise', 0.08)
        self.R = np.eye(n) * r_val
        
        self.F = np.eye(n) # Random walk modeli
        self.H = np.eye(n)

    def filter(self, o2_raw, pressure_raw):
        """
        Ham veriyi alır, Kalman adımlarını uygular ve temizlenmiş veriyi döner.
        """
        z = np.array([o2_raw, pressure_raw], dtype=float)
        
        # 1. TAHMİN (A priori)
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        
        # 2. GÜNCELLEME (A posteriori)
        y = z - (self.H @ x_pred)  # İnovasyon
        S = self.H @ P_pred @ self.H.T + self.R
        
        # Kalman Kazancı (Sistemin ne kadarı ölçümden, ne kadarı modelden gelecek?)
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        
        # Yeni Durum ve Kovaryans
        self.x = x_pred + K @ y
        I = np.eye(len(self.x))
        self.P = (I - K @ self.H) @ P_pred
        
        return self.x[0], self.x[1] # [filt_o2, filt_p]