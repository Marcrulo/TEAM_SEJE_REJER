import os
import time
import numpy as np
import gym
import pygame
import contextlib
import io
import matplotlib.pyplot as plt
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy

# Import the core game functions and state
from src.game.core import initialize_game_state, update_game, STATE, MAX_TICKS, MAX_MS, SCREEN_WIDTH, SCREEN_HEIGHT

class RaceCarEnv(gym.Env):
    """
    Custom Gym environment for the Race Car Challenge.
    Observation: distances from each of the 16 sensors (0-1000 px).
    Action space: Discrete actions ['NOTHING', 'ACCELERATE', 'DECELERATE', 'STEER_LEFT', 'STEER_RIGHT'].
    Reward: forward velocity (distance traveled each step).
    Done: crash or max ticks reached.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, seed: int = None, show: bool = False):
        super(RaceCarEnv, self).__init__()
        self._seed = seed if seed is not None else int(time.time())
        self.state = initialize_game_state(api_url="http://example.com/api/predict", seed_value=self._seed)

        self.action_map = ['NOTHING', 'ACCELERATE', 'DECELERATE', 'STEER_LEFT', 'STEER_RIGHT']
        self.action_space = spaces.Discrete(len(self.action_map))
        n_sensors = len(self.state.sensors)
        # TODO: The model likely benefits from normalizing the observation space
        self.observation_space = spaces.Box(low=0.0, high=1000.0, shape=(n_sensors,), dtype=np.float32)
        
        self.show = show

        self.screen = None
        if self.show:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Race Car Game")

    def seed(self, seed=None):
        self._seed = int(time.time())
        return [self._seed]

    def reset(self):
        # self.state = initialize_game_state(api_url="http://example.com/api/predict", seed_value=self._seed)

        print(f"Drove for: {round(self.state.distance, 2)} px in {self.state.elapsed_game_time} ms with an average speed of {round(self.state.distance / ((self.state.elapsed_game_time / 1000)+ (1e-06)), 2)} px/s")

        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            self.state = initialize_game_state(api_url="http://example.com/api/predict", seed_value=self._seed)


        return self._get_obs()

    def step(self, action: int):

        action_str = self.action_map[action]
        self.state = update_game(action_str, self.show, screen=self.screen)

        obs = self._get_obs()
        
        # TODO: Refine the reward function
        # Shaped reward: distance traveled, minus crash penalty, minus proximity penalty
        dist_reward = float(self.state.distance)
        crash_penalty = -20000.0 if self.state.crashed else 0.0
        # Penalty for being too close to obstacles (lower sensor = higher penalty)
        readings = [sensor.reading for sensor in self.state.sensors if sensor.reading is not None]
        min_sensor = min(readings) if readings else 1000.0
        proximity_penalty = - (1.0 / (min_sensor + 1.0))
        raw_reward = dist_reward + crash_penalty + 0.1 * proximity_penalty

        '''

        print("dist_reward:", dist_reward)
        print("crash_penalty:", crash_penalty)
        print("min_sensor:", min_sensor)
        print("proximity_penalty:", proximity_penalty)
        print("raw_reward:", raw_reward)
        print()

        '''

        reward = float(np.clip(raw_reward, -100.0, 1000.0))

        done = bool(self.state.crashed or self.state.ticks > MAX_TICKS or self.state.elapsed_game_time > MAX_MS)
        return obs, reward, done, {}

    def _get_obs(self):
        raw = np.array([sensor.reading or 0.0 for sensor in self.state.sensors], dtype=np.float32)
        normed = raw / 1000.0  # normalize to [0, 1]
        return np.clip(normed, 0.0, 1.0)

    # TODO: Rendering the model while training could give a glimpse into how it is learning
    def render(self, mode='human'):
        pass


if __name__ == "__main__":

    pygame.init()
    
    initial_seed = int(time.time())

    # Good baseline for training: 
    iterations = 20
    total_timesteps = 100000

    # Good for final training:
    # iterations = 100
    # total_timesteps = 1_000_000
    
    timesteps_per_iter = total_timesteps // iterations

    # Build and wrap environments
    base_env = DummyVecEnv([lambda: RaceCarEnv(seed=initial_seed)])
    monitored = VecMonitor(base_env)
    env = VecNormalize(monitored, norm_obs=True, norm_reward=True, clip_obs=10.0)
    env.seed(initial_seed)

    # Ensure output dirs
    os.makedirs("ppo_racecar_tensorboard", exist_ok=True)

    # Instantiate PPO
    # TODO: Explore the hyperparameters for a better performance
    model = PPO(
        policy='MlpPolicy',
        env=env,
        learning_rate=2.5e-4,
        n_steps=2048,
        batch_size=64,
        clip_range=0.2,
        gae_lambda=0.95,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="ppo_racecar_tensorboard/"
    )

    mean_rewards = []
    std_rewards = []
    entropy_losses = []
    policy_gradient_losses = []
    approx_kls = []
    training_losses = []
    exaplined_variances = []

    # Single eval env for efficiency
    eval_env = DummyVecEnv([lambda: RaceCarEnv(seed=int(time.time()), show=True)])
    eval_monitored = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_monitored, norm_obs=True, norm_reward=True, clip_obs=10.0)
    eval_env.seed(int(time.time()))

    for i in range(iterations):
        model.learn(
            total_timesteps=timesteps_per_iter,
            reset_num_timesteps=False
        )

        model_statistics = list(model.logger.name_to_value.values())
        (lr, entropy_loss, policy_gradient_loss,value_loss, approx_kl, clip_fraction, training_loss, exaplined_variance, n_updates, clip_range, *_ignored) = model_statistics

        # Build separate evaluation environment (unnormalized)
        # evaluation_seed = int(time.time())
        # eval_env = DummyVecEnv([lambda: RaceCarEnv(seed=evaluation_seed)])
        # eval_env.seed(evaluation_seed)


        # TODO: Maybe model.get_env() is a better way to make an envirnment
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5)
        print(f"Iteration {i+1}/{iterations} completed. Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

        mean_rewards.append(mean_reward)
        entropy_losses.append(entropy_loss)
        policy_gradient_losses.append(policy_gradient_loss)
        approx_kls.append(approx_kl)
        training_losses.append(training_loss)
        exaplined_variances.append(exaplined_variance)
    

        '''

        if (i % 3) == 0 and i != 0:
            # Plot the values
            plt.plot(mean_rewards, marker='o', linestyle='-', color='b')
            plt.xlabel('Iteration')
            plt.ylabel('Reward')
            plt.title('Reward Over Iterations')
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        '''



    # Plot the values
    plt.plot(mean_rewards, marker='o', linestyle='-', color='b')
    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.title('Reward Over Iterations')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    # Plot the values
    plt.plot(entropy_losses, marker='o', linestyle='-', color='b')
    plt.xlabel('Iteration')
    plt.ylabel('Entropy Loss')
    plt.title('Entropy loos over Iterations')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.plot(policy_gradient_losses, marker='o', linestyle='-', color='b')
    plt.xlabel('Iteration')
    plt.ylabel('Policy Gradient Loss')
    plt.title('Policy Gradient Loss over Iterations')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.plot(approx_kls, marker='o', linestyle='-', color='b')
    plt.xlabel('Iteration')
    plt.ylabel('KL Divergence')
    plt.title('KL divergence over Iterations')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.plot(training_losses, marker='o', linestyle='-', color='b')
    plt.xlabel('Iteration')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Over Iterations')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.plot(exaplined_variances, marker='o', linestyle='-', color='b')
    plt.xlabel('Iteration')
    plt.ylabel('Explained Variance')
    plt.title('Explained Variance Over Iterations')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
    print()

    pygame.quit()

    # Persist model and normalizer
    # model.save("ppo_racecar_model")
    # env.save("ppo_racecar_vecnormalize")

