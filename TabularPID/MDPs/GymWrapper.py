import gymnasium as gym
from stable_baselines3.common.atari_wrappers import AtariWrapper

def create_environment(env_name, slow_motion=1):
    """
    Return the environment with the given name as well as the stopping criterion.
    The stopping criterion is the function that takes the reward for the current episode,
    and returns True if the agent has solved the environment and we should stop training, and False otherwise.
    The criteria are taken from the gymnasium documentation when possible.

    Possible Environments include:
    - CartPole-v1: The time between actions is 0.02 * slow_motion.
        The stopping reward is 195.
    - LunarLander-v2: The gravity is -10 * slow_motion.
        The stopping reward is 200 (taken from the gymnasium documentation).
    - Acrobot-v1: The time between actions is 0.2 * slow_motion
        The stopping reward is -100.
    - MountainCar-v0: The max speed is self.max_speed * slow_motion
        The stopping reward is -100.
    - PongNoFrameskip-v4, BreakoutNoFrameskip-v4, SpaceInvadersNoFrameskip-v4: The Atari envs don't support any slow motion.
        The stopping reward is 15.

    Otherwise, slow_motion is ignored, and the environment is created as normal, and the criteria always returns False.
    """
    # If we are using a classic control environment:
    if env_name == "CartPole-v1":
        env = gym.make(env_name, render_mode='rgb_array')
        env.tau *= slow_motion
        return env, lambda n: n >= 195
    elif env_name == "LunarLander-v2":
        return gym.make(env_name, gravity=-10 * slow_motion, render_mode='rgb_array'), lambda n: n >= 200
    elif env_name == "Acrobot-v1":
        env = gym.make(env_name, render_mode='rgb_array')
        env.dt *= slow_motion
        return env, lambda n: n >= -100
    elif env_name == "MountainCar-v0":
        env = gym.make(env_name, render_mode='rgb_array')
        env.max_speed *= slow_motion
        return env, lambda n: n >= -100
    elif env_name in {"PongNoFrameskip-v4", "BreakoutNoFrameskip-v4", "SpaceInvadersNoFrameskip-v4"}:
        return AtariWrapper(gym.make(env_name, render_mode='rgb_array')), lambda n: n >= 18
    else:
        return gym.make(env_name, render_mode='rgb_array'), lambda n: False

"""
class GymWrapperClassicControl(gym.Wrapper):
    \"""A gym environment with slow motion\"""

    def __init__(self, env_name, slow_motion=1):
        \"""Create a gym environment with the given name.
        The slow_motion parameter controls how much slower the environment should run.
        prg is a random number generator, which should have a rand() method.
        The assumption is that slow_motion is in [0, 1].
        \"""
        super().__init__(gym.make(env_name, render_mode='rgb_array'))
        self.env._max_episode_steps *= 1/slow_motion  # Slow down the environment, but keep the same number of steps
        self.unwrapped_env = self.env.unwrapped
        self.env.reset()
        self.slow_motion = slow_motion

    def step(self, action):
        \"""Take a step in the environment, and return the next state, reward, and done flag.\"""
        current_state = self.unwrapped_env.state
        next_state, reward, done, truncated, info = self.env.step(action)

        # Slow motion
        self.unwrapped_env.state = self.slow_motion * next_state + (1 - self.slow_motion) * current_state

        # Set the true reward to zero with probability 1 - self.slow_motion
        if self.env.np_random.random() < 1 - self.slow_motion:
            reward = 0

        return next_state, reward, done, truncated, info
    
    def reset(self, **kwargs):
        \"""Reset the environment, and return the initial state.\"""
        reset_output = self.env.reset(**kwargs)
        self.current_state = reset_output[0]
        return reset_output
"""