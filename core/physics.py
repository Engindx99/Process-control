import numpy as np
from .constants import SIGMA

def calc_convection(t_gas, t_solid, h_coef=200.0):
    """
    Gaz ve katı arasındaki taşınım ile ısı transferi.
    Q = h * A * (Tg - Ts)
    """
    return h_coef * (t_gas - t_solid)

def calc_radiation(t_gas, t_solid, epsilon_g=0.96, epsilon_s=0.96):
    """
    Gaz ve katı arasındaki ışınım ile ısı transferi.
    Stefan-Boltzmann yasası temelli.
    """
    # Basitleştirilmiş efektif emite katsayısı
    eff_epsilon = 1 / (1/epsilon_g + 1/epsilon_s - 1)
    return eff_epsilon * SIGMA * (t_gas**4 - t_solid**4)

def calc_conduction_loss(t_shell, t_amb, h_combined=0.015):
    """
    Fırın dış yüzeyinden çevreye olan kombine (taşınım+ışınım) ısı kaybı.
    h_combined: Genelde 0.01 ile 0.02 arası seçilir.
    """
    return h_combined * (t_shell - t_amb)