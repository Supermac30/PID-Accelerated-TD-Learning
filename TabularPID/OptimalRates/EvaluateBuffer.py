"""
Code that takes the (state, action) pairs in the replay buffer in models/$env/buffer.npy
and runs the trained model in models/$env/$env.zip on it,
getting a Q-value for each (state, action) pair by finding the discounted
reward of a monte carlo simulation.

The ((state, action), Q value) pairs are then saved in models/$env/bufferQValues.npy
"""
def run_simulation(model, env, action, gamma, seed):
    """
    Runs a monte carlo simulation starting from the given state and action,
    returning the discounted reward.
    """
    set_seed(env, seed)
    state, _, _, _, _ = env.step(int(action))
    total_reward = 0
    discount = 1

    done = False
    truncated = False
    while not (done or truncated):
        action = model.predict(state)[0].item()
        state, reward, done, truncated, _ = env.step(action)
        total_reward += discount * reward
        discount *= gamma
    return total_reward


def set_seed(env, seed):
    """
    A hacky way to set the random seed of an environment without resetting the state.
    Gymnasium really doesn't make it easy to do this.
    """
    unwrapped_env = env.unwrapped
    super(type(unwrapped_env), unwrapped_env).reset(seed=int(seed))
