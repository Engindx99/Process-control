from core.physics import calc_convection, calc_radiation
from core.kinetics import calc_calcination_rate, calc_reaction_heat

class BaseZone:
    def __init__(self, config):
        self.cfg = config

    def compute_energy_exchange(self, t_g, t_s, c_caco3):
        # Tüm zonlarda ortak olan ısı transferleri
        q_conv = calc_convection(t_g, t_s)
        q_rad = calc_radiation(t_g, t_s)
        return q_conv + q_rad

class CalciningZone(BaseZone):
    def update_physics(self, t_g, t_s, c_caco3):
        # Isı transferi + Kimyasal yük
        q_transfer = self.compute_energy_exchange(t_g, t_s, c_caco3)
        
        rate = calc_calcination_rate(t_s, c_caco3)
        q_reaction = calc_reaction_heat(rate)
        
        # Net enerji değişimi (Solid için)
        # Isı transferinden gelen enerji - Reaksiyonun yuttuğu enerji
        net_q_solid = q_transfer - q_reaction
        return net_q_solid, rate