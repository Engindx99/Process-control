import numpy as np

class CycloneStage:
    def __init__(self, stage_id, config):
        self.stage_id = stage_id
        self.efficiency = config.get('efficiency', 0.92) 
        self.heat_transfer_coeff = config.get('h_coeff', 0.8) 
        
        self.T_mat = 25.0
        self.T_gas = 300.0

    def step(self, T_mat_in, T_gas_in, m_mat_in, m_gas_in, dt):
        cp_mat = 1.02 
        cp_gas = 1.15 
        tau = 5.0 # Termal atalet (saniye)
        
        # Enerji Dengesi Hedefi
        T_equilibrium = (m_mat_in * cp_mat * T_mat_in + m_gas_in * cp_gas * T_gas_in) / \
                        (m_mat_in * cp_mat + m_gas_in * cp_gas + 1e-6)
        
        # Dinamik Güncelleme
        self.T_mat += (T_equilibrium - self.T_mat) * (dt / tau)
        self.T_gas += (T_equilibrium - self.T_gas) * (dt / tau)
        
        # Kütle Dengesi: Toz tutma verimi
        m_mat_out = m_mat_in * self.efficiency
        
        return self.T_mat, self.T_gas, m_mat_out

class PreheaterTower:
    def __init__(self, config=None):
        self.config = config if config else {}
        self.stages = [CycloneStage(i, self.config) for i in range(5)]
        
    def compute_tower(self, T_raw_meal, T_kiln_gas, m_meal_total, m_gas_total, dt=1):
        temp_history_mat = []
        current_m_meal = m_meal_total # İlk siklona giren tam kütle

        # 5 Katmanlı Simülasyon
        for i in range(5):
            t_m_in = T_raw_meal if i == 0 else self.stages[i-1].T_mat
            t_g_in = T_kiln_gas if i == 4 else self.stages[i+1].T_gas
            
            # Kütle akışı: Her katta verim kadar malzeme bir alta geçer
            t_m_out, t_g_out, m_meal_next = self.stages[i].step(t_m_in, t_g_in, current_m_meal, m_gas_total, dt)
            
            current_m_meal = m_meal_next # Bir sonraki siklona inen hammadde
            temp_history_mat.append(t_m_out)

        return {
            "feed_to_kiln_temp": self.stages[4].T_mat,
            "feed_to_kiln_mass": current_m_meal,
            "exit_gas_temp": self.stages[0].T_gas,
            "stage_temps": temp_history_mat
        }

# --- TEST BLOĞU ---
if __name__ == "__main__":
    tower = PreheaterTower()
    # 100 saniye boyunca sistemi simüle et
    for _ in range(100):
        res = tower.compute_tower(T_raw_meal=25, T_kiln_gas=1100, m_meal_total=150, m_gas_total=100)
    
    print(f"Bacaya giden gaz: {res['exit_gas_temp']:.2f} °C")
    print(f"Fırına giren malzeme: {res['feed_to_kiln_temp']:.2f} °C")
    print(f"Kule içi sıcaklık gradyanı: {[round(t,1) for t in res['stage_temps']]}")