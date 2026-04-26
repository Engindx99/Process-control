import numpy as np
from .constants import R

def calc_calcination_rate(T_solid, c_caco3, A=1.5e6, Ea=150000):
    """
    Arrhenius denklemi kullanılarak kalsinasyon hızı hesabı.
    k = A * exp(-Ea / (R * T))
    """
    if T_solid < 750:  # 750K altındaki reaksiyon ihmal edilebilir
        return 0.0
    
    k = A * np.exp(-Ea / (R * T_solid))
    # Reaksiyon hızı konsantrasyona bağlıdır (1. derece reaksiyon)
    rate = k * c_caco3
    return rate

def calc_reaction_heat(rate):
    """
    Reaksiyonun sistemden çektiği enerji (Endotermik).
    H_rxn ~ 1780 kJ/kg_caco3
    """
    h_rxn = 1780000  # J/kg
    return rate * h_rxn

def calc_clinker_kinetics(T_solid, c_cao, c_c2s, c_c3s):
    """
    Klinker oluşum reaksiyonları (C2S ve C3S).
    Bu reaksiyonlar ekzotermiktir.
    """
    if T_solid < 1250: # 1250K altı sıvı faz yetersiz
        return 0, 0, 0

    # C2S Oluşumu (CaO + SiO2 -> C2S)
    rate_c2s = 0.5 * np.exp(-150000 / (8.314 * T_solid)) * c_cao
    
    # C3S Oluşumu (C2S + CaO -> C3S) - Daha yüksek sıcaklık ister
    rate_c3s = 0.8 * np.exp(-220000 / (8.314 * T_solid)) * c_c2s * (c_cao > 0.05)

    # Ekzotermik Isı Salınımı (J/kg)
    # Kalsinasyonun aksine sisteme ısı verirler
    q_exothermic = (rate_c2s * 500000) + (rate_c3s * 200000)
    
    return rate_c2s, rate_c3s, q_exothermic