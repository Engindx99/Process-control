import numpy as np

class SelectiveKalmanFilter:
    def __init__(self, config):
        """
        Selective Kalman Filter: O2 ve Basınç sinyallerini temizler.
        Durum Vektörü x: [o2, pressure]
        """
        # Başlangıç tahminleri (Fırın genelde bu değerlerde başlar)
        self.x = np.array([2.5, -2.0], dtype=float) 
        n = len(self.x)
        
        # Kovaryans Matrisi (Güven seviyesi)
        self.P = np.eye(n) * 1.0
        
        # Parametreleri Config'den Çek
        # Q: Süreç gürültüsü (Modeline ne kadar güveniyorsun?)
        q_val = config.get('kalman', {}).get('process_noise', 0.001)
        self.Q = np.eye(n) * q_val
        
        # R: Ölçüm gürültüsü (Sensörüne ne kadar güvenmiyorsun?)
        r_val = config.get('kalman', {}).get('measurement_noise', 0.08)
        self.R = np.eye(n) * r_val
        
        # F: Durum Geçiş Matrisi (Gelecek adım = Mevcut adım kabul ediyoruz)
        self.F = np.eye(n)
        
        # H: Gözlem Matrisi
        self.H = np.eye(n)

    def update(self, o2_raw, pressure_raw):
        """
        Ham sensör verilerini alır ve süzülmüş değerleri döner.
        """
        z = np.array([o2_raw, pressure_raw], dtype=float)
        
        # 1. TAHMİN (Predict)
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        
        # 2. GÜNCELLEME (Update / Correct)
        y = z - (self.H @ x_pred)  # İnovasyon (Ölçüm artığı)
        S = self.H @ P_pred @ self.H.T + self.R # Sistem Kovaryansı
        
        # Kalman Kazancı (K)
        # S 2x2 olduğu için inv() çok hızlı çalışır
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        
        # Yeni Durum
        self.x = x_pred + K @ y
        
        # Yeni Kovaryans (Joseph Formu veya standart form)
        I = np.eye(len(self.x))
        self.P = (I - K @ self.H) @ P_pred
        
        return self.x.tolist() # [filt_o2, filt_p]