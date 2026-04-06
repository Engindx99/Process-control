import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from ml.rl_agent import RotaryKilnEnv


def run_industrial_simulation(model_path=None):

    # ---------------- ENV ----------------
    env = RotaryKilnEnv()

    # ---------------- MODEL ----------------
    if model_path is not None:
        print(f"Model yükleniyor: {model_path}")
        model = PPO.load(model_path)
    else:
        print("Sıfırdan eğitim başlıyor...")
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=5000)
        model.save("models/rotary_kiln_v1")

    # ---------------- SIMULATION ----------------
    obs, _ = env.reset()

    history = {
        "temp": [],
        "fuel": [],
        "setpoint": []
    }

    print("Simülasyon başladı...")

    for i in range(100):

        # RL action (setpoint offset)
        action, _ = model.predict(obs, deterministic=True)

        # step
        obs, reward, done, truncated, info = env.step(action)

        # ---------------- logging ----------------
        temp = obs[0]

        # env içine fuel koymadıysan:
        fuel = info.get("fuel", np.nan)

        # setpoint reconstruction (clean)
        setpoint = env.setpoint

        history["temp"].append(temp)
        history["fuel"].append(fuel)
        history["setpoint"].append(setpoint)

    plot_results(history)


# ---------------- PLOT ----------------
def plot_results(history):

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel("Time step")

    ax1.set_ylabel("Temperature (°C)")
    ax1.plot(history["temp"], label="Temperature", linewidth=2)
    ax1.plot(history["setpoint"], "--", label="Setpoint", alpha=0.7)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Fuel")
    ax2.plot(history["fuel"], label="Fuel", color="orange")

    plt.title("Hybrid RL + MPC System Performance")
    plt.show()


if __name__ == "__main__":
    run_industrial_simulation()