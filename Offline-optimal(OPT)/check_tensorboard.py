from stable_baselines3 import A2C, SAC, PPO
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
# model = A2C("MlpPolicy", "CartPole-v1", verbose=1, tensorboard_log="./a2c_cartpole_tensorboard/")
# model.learn(total_timesteps=10_000, tb_log_name="first_run")


# model = SAC("MlpPolicy", "Pendulum-v1", tensorboard_log="/tmp/sac1/", verbose=1)
model = SAC("MlpPolicy", "Pendulum-v1", tensorboard_log="./sac/", verbose=1)

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        value = np.random.random()
        self.logger.record("random_value", value)
        return True


model.learn(50000, callback=TensorboardCallback())