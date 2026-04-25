import numpy as np
import cma

# =====================================================
# ROTARY KILN MODEL (CMA-ES READY)
# =====================================================
class RotaryKilnV2:

    def __init__(self, dt=0.2):
        self.dt = dt

        # tunable parameters (CMA-ES optimizes these externally)
        self.Cp_solid = 1.0
        self.gas_capacity = 10000
        self.h_eff = 0.48
        self.LHV = 7200

        self.mass = {
            "burning": 8000,
            "calcination": 12000,
            "preheat": 15000,
            "cooling": 10000
        }

        self.T = {
            "burning": 1450.0,
            "calcination": 1085.0,
            "preheat": 830.0,
            "cooling": 1000.0
        }

        self.T_gas = 1750.0
        self.fuel = 18.5
        self.fan = 950.0

    def step(self, fuel_cmd, fan_cmd):

        # actuator lag
        self.fuel += 0.02 * (fuel_cmd - self.fuel)
        self.fan  += 0.03 * (fan_cmd - self.fan)

        air_flow = 0.8 + 1.7 / (1 + np.exp(-(self.fan - 900)/170))

        lambda_excess = (air_flow * 10) / (self.fuel * 10 + 1e-6)
        lambda_excess = np.clip(lambda_excess, 0.8, 1.3)

        eta = np.exp(-((lambda_excess - 1.05)/0.25)**2)

        Q_comb = self.fuel * self.LHV * eta

        # heat transfer
        Q_gas_to_solid = self.h_eff * 600 * (self.T_gas - self.T["burning"])

        # clamp safety
        Q_gas_to_solid = np.clip(Q_gas_to_solid, 0, 0.95 * Q_comb)

        # solid balance
        conv_loss = 0.05 * (self.T["burning"] - 300)

        T_b = self.T["burning"] + 273.15
        rad_loss = 1.2e-9 * (T_b**4 - 1050**4)

        loss = conv_loss + rad_loss

        dT = (Q_gas_to_solid - loss) / (self.mass["burning"] * self.Cp_solid)

        self.T["burning"] += self.dt * dT

        # gas balance (energy conservation)
        dE = Q_comb - (Q_gas_to_solid + loss)
        self.T_gas += self.dt * (dE / self.gas_capacity)

        # safety clamp
        for k in self.T:
            self.T[k] = np.clip(self.T[k], 300, 3500)

        return self.T["burning"]


# =====================================================
# CMA-ES OBJECTIVE
# =====================================================
def objective(params):

    h_eff, gas_capacity, LHV = params

    model = RotaryKilnV2()
    model.h_eff = h_eff
    model.gas_capacity = gas_capacity
    model.LHV = LHV

    error = 0.0
    prev = None

    for _ in range(150):

        T = model.step(18.5, 950)

        # safety rejection
        if np.isnan(T) or T > 4000 or T < 300:
            return 1e12

        error += (T - 1450)**2

        if prev is not None:
            error += 0.1 * (T - prev)**2

        prev = T

    return error


# =====================================================
# CMA-ES RUN
# =====================================================
if __name__ == "__main__":

    x0 = [0.48, 10000, 7200]
    sigma = 0.2

    es = cma.CMAEvolutionStrategy(x0, sigma, {
        'popsize': 8,
        'maxiter': 20
    })

    es.optimize(objective)

    print("\nBEST PARAMETERS:")
    print(es.result.xbest)