
import numpy as np
import matplotlib.pyplot as plt
def smooth(data, window=40):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')
  
def plot_convergence_two(m20_rewards, m80_rewards, filename):

    plt.figure(figsize=(14, 5))

    # ---------- (a) m = 20 ----------
    plt.subplot(1, 2, 1)
    plt.plot(m20_rewards, color="lightgray", linewidth=1)
    plt.plot(smooth(m20_rewards), color="steelblue", linewidth=2)
    plt.title("(a) Federation size m = 20")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True, linestyle="--", alpha=0.4)

    # ---------- (b) m = 80 ----------
    plt.subplot(1, 2, 2)
    plt.plot(m80_rewards, color="lightgray", linewidth=1)
    plt.plot(smooth(m80_rewards), color="steelblue", linewidth=2)
    plt.title("(b) Federation size m = 80")
    plt.xlabel("Episode")
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_policy_comparison_three(drl_rewards, ga_rewards, ra_rewards, filename, title):

    plt.figure(figsize=(7, 5))

    plt.plot(ra_rewards, 'b--', label="RA")
    plt.plot(ga_rewards, 'k-', label="GA")
    plt.plot(drl_rewards, 'r-.', label="DRL-Dispatcher")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
def aggregate_into_bins(values, scores, bins):
    values = np.array(values)
    scores = np.array(scores)
    grouped = []

    for i in range(len(bins) - 1):
        mask = (values >= bins[i]) & (values < bins[i+1])
        grouped.append(np.mean(scores[mask]) if np.any(mask) else 0)
    return grouped


def plot_task_length_effect_paper(psi, V, U, filename_profit, filename_delay):
    bins = [0, 200, 300, 400, 500, 600]
    labels = ["1x", "2x", "3x", "4x", "5x"]

    avg_profit = aggregate_into_bins(psi, V, bins)
    avg_delay = aggregate_into_bins(psi, U, bins)

    # PROFIT
    plt.figure(figsize=(7,5))
    plt.plot(labels, avg_profit, 'r-o')
    plt.xlabel("Task's length")
    plt.ylabel("Estimated profit")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(filename_profit)
    plt.close()

    # DELAY
    plt.figure(figsize=(7,5))
    plt.plot(labels, avg_delay, 'r-o')
    plt.xlabel("Task's length")
    plt.ylabel("Estimated response delay")
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(filename_delay)
    plt.close()



def plot_task_size_effect_paper(b, V, U, filename_profit, filename_delay):
    bins = [0, 2, 4, 6, 8, 10]
    labels = ["1x", "2x", "3x", "4x", "5x"]

    avg_profit = aggregate_into_bins(b, V, bins)
    avg_delay = aggregate_into_bins(b, U, bins)

    plt.figure(figsize=(7,5))
    plt.plot(labels, avg_profit, 'r-o')
    plt.xlabel("Task's size")
    plt.ylabel("Estimated profit")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(filename_profit)
    plt.close()

    plt.figure(figsize=(7,5))
    plt.plot(labels, avg_delay, 'r-o')
    plt.xlabel("Task's size")
    plt.ylabel("Estimated response delay")
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(filename_delay)
    plt.close()
