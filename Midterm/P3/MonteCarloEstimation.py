import random
import math
import numpy as np
import matplotlib.pyplot as plt

random.seed(15679995792765019345)
NUM_TRIALS = 10_000_000


def generate_convergence_data(num_trials):
    sample_indices = set(np.unique(np.logspace(
        0, np.log10(num_trials), 500).astype(int)))
    trials = []
    estimates = []
    running_sum = 0

    for i in range(1, num_trials + 1):
        s, count = 0.0, 0
        while s <= 1.0:
            s += random.random()
            count += 1
        running_sum += count

        if i in sample_indices:
            trials.append(i)
            estimates.append(running_sum / i)

    return np.array(trials), np.array(estimates)


def plot_convergence(trials, estimates, save_path="monte_carlo_convergence.png"):
    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.plot(trials, estimates, color="#3266ad",
            linewidth=1.5, label="Running estimate")
    ax.axhline(y=math.e, color="#E24B4A", linewidth=1.2,
               linestyle="--", label=f"True $e$ = {math.e:.6f}")
    ax.annotate(
        f"  {estimates[-1]:.6f}",
        xy=(trials[-1], estimates[-1]),
        fontsize=10,
        color="#3266ad",
        fontweight="bold",
        va="center",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Number of trials (log scale)", fontsize=12)
    ax.set_ylabel("Estimated value", fontsize=12)
    ax.set_title("Monte Carlo convergence to $e$",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(math.e - 1, math.e + 1)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.show()


trials, estimates = generate_convergence_data(NUM_TRIALS)
plot_convergence(trials, estimates)
