import numpy as np
import matplotlib.pyplot as plt

def plot_with_exponential_averaging(x, y, label, alpha):
    y_ema = [y[0],] 
    for y_i in y[1:]:
        y_ema.append(y_ema[-1] * alpha + y_i * (1 - alpha))
    
    p = plt.plot(x, y_ema, label=label)
    
    plt.plot(x, y, color=p[0].get_color(), alpha=0.2)


def plot_dqn_train_result(result, label="", alpha=0.95):
    plot_with_exponential_averaging(np.cumsum(result), result, label, alpha)
