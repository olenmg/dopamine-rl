import gymnasium as gym
import matplotlib.pyplot as plt

from utils.config import Config
from trainer import Trainer
from plot import plot_dqn_train_result

def train_dqn(config, steps=50000):
    trainer = Trainer(config)
    result = trainer.train(steps)

    return result


config = Config(
    run_name="Plain",
    env_id="CartPole-v1",
    optim_kwargs={'lr': 2.3e-3},
    eps_end=0.001,
    discount_rate=0.99,
    n_steps=1,
    buffer_size=50000,
    batch_size=64,
    learning_starts=512,
    train_freq=2,
    target_update_freq=512,
    device="cpu",
    verbose=True
)
train_result = train_dqn(config, steps=50000)

plot_dqn_train_result(train_result, label="DQN", alpha=0.9)
plt.axhline(y=500, color='grey', linestyle='-')  # 500 is the maximum score!
plt.xlabel("steps")
plt.ylabel("Episode Reward")

plt.legend()
plt.title("Training DQN")
plt.show()
