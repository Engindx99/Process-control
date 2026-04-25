import numpy as np

from src.Rotaryklin.Rotary_klin import KilnPlant
from src.Rotaryklin.mpc_conroller import CasADiMPC
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def run():

    plant = KilnPlant(N=60)
    mpc = CasADiMPC(N=20)

    z = 0.0
    u_prev = 0.6   # 🔥 EKLENEN KRİTİK STATE

    for t in range(300):

        # plant step
        Ts, Tg, X = plant.step(u_prev)

        # MPC solve (FIXED CALL)
        u, z = mpc.solve(
            Ts[-1],
            Tg[-1],
            X[-1],
            z,
            u_prev,     # 🔥 FIX
            Tref=1450
        )

        # plant apply
        Ts, Tg, X = plant.step(u)

        # update previous control
        u_prev = u

        print(f"Step {t} | Tg_out={Tg[-1]:.1f} | u={u:.3f}")


if __name__ == "__main__":
    run()