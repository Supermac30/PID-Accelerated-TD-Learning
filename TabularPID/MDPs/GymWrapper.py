import gymnasium as gym

def create_gym_wrapper(env_name, slow_motion=1):
    # If we are using a classic control environment:
    if "CartPole" in env_name or "Acrobot" in env_name or "MountainCar" in env_name or "Pendulum" in env_name:
        return GymWrapperClassicControl(env_name, slow_motion)
    elif "LunarLander" in env_name:
        return gym.make(env_name, gravity=-10 * slow_motion, render_mode='rgb_array')
    else:
        return gym.make(env_name, render_mode='rgb_array')
    

class GymWrapperClassicControl(gym.Wrapper):
    """A gym environment with slow motion"""

    def __init__(self, env_name, slow_motion=1):
        """Create a gym environment with the given name.
        The slow_motion parameter controls how much slower the environment should run.
        prg is a random number generator, which should have a rand() method.
        The assumption is that slow_motion is in [0, 1].
        """
        super().__init__(gym.make(env_name, render_mode='rgb_array'))
        self.env._max_episode_steps *= 1/slow_motion  # Slow down the environment, but keep the same number of steps
        self.unwrapped_env = self.env.unwrapped
        self.env.reset()
        self.slow_motion = slow_motion

    def step(self, action):
        """Take a step in the environment, and return the next state, reward, and done flag."""
        current_state = self.unwrapped_env.state
        next_state, reward, done, truncated, info = self.env.step(action)

        # Slow motion
        self.unwrapped_env.state = self.slow_motion * next_state + (1 - self.slow_motion) * current_state

        # Set the true reward to zero with probability 1 - self.slow_motion
        if self.env.np_random.random() < 1 - self.slow_motion:
            reward = 0

        return next_state, reward, done, truncated, info
    
    def reset(self, **kwargs):
        """Reset the environment, and return the initial state."""
        reset_output = self.env.reset(**kwargs)
        self.current_state = reset_output[0]
        return reset_output