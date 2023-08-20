import numpy as np
import matplotlib.pyplot as plt

def plot_with_exponential_averaging(x, y, label, alpha):
    y_ema = [y[0],] 
    for y_i in y[1:]:
        y_ema.append(y_ema[-1] * alpha + y_i * (1 - alpha))
    
    p = plt.plot(x, y_ema, label=label)
    
    plt.plot(x, y, color=p[0].get_color(), alpha=0.2)


def plot_train_result(result, label="", alpha=0.95, save_path="./"):
    rewards = [r['r'] for r in result]
    lengths = [r['l'] for r in result]

    plot_with_exponential_averaging(np.cumsum(lengths), rewards, label, alpha)
    plt.axhline(y=int(max(rewards)*1.1), color='grey', linestyle='-')  # 500 is the maximum score!
    plt.xlabel("Training Steps")
    plt.ylabel("Episode Reward")

    plt.legend()
    plt.title(label)
    plt.savefig(save_path)
    plt.cla()
