import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
import logging
import multiprocess as mp
import gymnasium as gym

from TabularPID.MDPs.MDP import PolicyEvaluation, Control, Control_Q
from TabularPID.MDPs.Environments import *
from TabularPID.MDPs.Policy import Policy
from stable_baselines3.dqn.dqn import DQN

base_directory = '/h/bedaywim/PID-Accelerated-TD-Learning'

def normalize(arr):
    """Normalize the array by the first value. If the array is empty, or starts with zero, return it.
    """
    if arr.size == 0 or arr[0] == 0:
        return arr
    
    return arr / arr[0]

def pick_seed(seed):
    """Return a seed if one is inputted, and -1 otherwise. Log the seed chosen."""
    if seed == -1:
        seed = np.random.randint(0, 1000000)
    logging.info("Seed chosen is %d", seed)
    return seed

def build_test_function(norm, V_pi):
    """Return the function that tests how far away our current estimate is from V_pi
    in the format that the Agent class expects.
    """
    if norm == "inf":
        return lambda V, Vp, BR: np.max(np.abs(V - V_pi))
    
    elif type(norm) == type("") and norm[:4] == "diff":
        # Check if the norm is of the form diff <num-1> <num-2> or if is of the form diff <num>
        if len(norm.split(" ")) == 3:
            i1, i2 = map(int, norm.split(" ")[1:])
            return lambda Q, Qp, BR: Q[i1][i2] - V_pi[i1][i2]
        else:
            index = int(norm[5:])
            return lambda V, Vp, BR: V[index] - V_pi[index]
    
    elif norm == 2:
        return lambda V, Vp, BR: (V - V_pi).T @ (V - V_pi)
    elif norm == 1:
        return lambda V, Vp, BR: np.sum(np.abs(V - V_pi))
    else:
        # How do we fix this bug?
        #TypeError: array type float128 is unsupported in linalg
        
        return lambda V, Vp, BR: np.linalg.norm(V - V_pi, ord=norm)

def get_env_policy(name, seed):
    """Return an environment, policy tuple given its string name as input. A seed is inputted as well
    for reproducibility.
    The environments are as follows:
        - "garnet": The garnet problem with the default settings in PAVIA
        - "chain walk": The chain walk problem with 50 states, and a policy that always moves left
        - "chain walk random": The chain walk problem with 50 states, and a policy that takes random choices
        - "cliff walk": The Cliff walk problem as implemented in OS-Dyna
    """
    if name[:6] == "garnet":
        # The name is of the form garnet <seed> <num_states>
        seed, num_states = map(int, name[7:].split(" "))
        return garnet_problem(num_states, 3, 5, 10, seed)
    elif name == "chain walk":
        return chain_walk_left(50, 2, seed)
    elif name == "chain walk random":
        return chain_walk_random(50, 2, seed)
    elif name == "cliff walk":
        return cliff_walk(seed)
    elif name == "cliff walk optimal":
        return cliff_walk_optimal(seed)
    elif name == "identity":
        return identity(seed)
    elif name[:6] == "normal":
        variance = float(name[7:])
        return normal_env(variance, seed)
    elif name == "bernoulli":
        return bernoulli_env(seed)
    elif name in {"CartPole-v1", "Acrobot-v1", "MountainCar-v0", "LunarLander-v2"}:
        return gym.make(name), optimal_gym_policy(name)
    else:
        raise Exception("Environment not indexed")

def get_model(env_name):
    """Return the model with the same env_name from the models directory"""
    # The model is in a directory that starts with the same name as the environment
    model_dir = next(iter(filter(lambda x: x.startswith(env_name), os.listdir(f"{base_directory}/models"))))
    return DQN.load(f"{base_directory}/models/{model_dir}/{model_dir}.zip")

def optimal_gym_policy(name):
    model = get_model(name)
    return lambda state: np.argmax(model.predict(state.reshape(1, -1))[0])

def bernoulli_env(seed):
    env = BernoulliApproximation(seed)
    return env, Policy(env.num_actions, env.num_states, env.prg, None)

def normal_env(variance, seed):
    env = NormalApproximation(variance, seed)
    return env, Policy(env.num_actions, env.num_states, env.prg, None)

def cliff_walk(seed):
    """Return the CliffWalk Environment with the policy that moves randomly"""
    env = CliffWalk(0.9, seed)
    return env, Policy(env.num_actions, env.num_states, env.prg, None)

def cliff_walk_optimal(seed):
    """Return the CliffWalk Environment with the optimal policy"""
    env = CliffWalk(0.9, seed)
    policy = np.zeros((env.num_states, env.num_actions))

    for i in range(env.num_states):
        if i in {0, 12, 24}:
            policy[i, 2] = 1
        elif i in range(6, 12) or i in range(18, 23) or i in range(30, 35):
            policy[i, 1] = 1
        else:
            policy[i, 0] = 1

    return env, Policy(env.num_actions, env.num_states, env.prg, policy)

def garnet_problem(num_states, num_actions, bP, bR, seed):
    """Return the Garnet Environment with the policy that chooses the first move at each iteration"""
    env = Garnet(
        num_states, num_actions, bP, bR, seed
    )
    return env, Policy(num_actions, num_states, env.prg, None)


def PAVIA_garnet_settings(seed):
    """Return Garnet used in the experiments of PAVIA."""
    return garnet_problem(50, 1, 3, 5, seed)


def chain_walk_random(num_states, num_actions, seed):
    """Return the chain walk environment with the policy that takes random moves."""
    env = ChainWalk(num_states, seed)
    return env, Policy(num_actions, num_states, env.prg, None)


def chain_walk_left(num_states, num_actions, seed):
    """Return the chain walk environment with the policy that always moves left."""
    env = ChainWalk(num_states, seed)
    policy = np.zeros((num_states, num_actions))
    for i in range(num_states):
        policy[i,0] = 1
        policy[i,1] = 0
    return env, Policy(num_actions, num_states, env.prg, policy)

def identity(seed):
    """Return the identity environment the policy that picks the only available action."""
    env = IdentityEnv(1, seed)
    return env, Policy(1, 1, env.prg, None)


def find_optimal_learning_rates(agent, value_function_estimator, learning_rates={}, update_I_rates={}, update_D_rates={}, verbose=False, repeat=3):
    """Run a grid search for values of N and alpha that makes the
    value_function_estimator have the lowest possible error.

    agent should be an Agent object.
    value_function_estimator should be a function that runs agent.estimate_value_function with
        the correct parameters, and does not plot

    Find the learning rates that make the error decrease below the threshold the fastest. If no such rate exists,
    pick the learning rates that minimize the error at the end.

    Return the optimal parameters and the associated history.

    WARNING: This causes spooky action at a distance, changing the learning rates.
    """
    def try_params(params):
        """Run the current value function with the parameters set, and return
        the optimal length, optimal params, and optimal history.
        params: an object representing the parameters we initialized the value_function_estimator to
        """
        agent.reset(reset_environment=True)
        results = []
        for param in params:
            agent.set_learning_rates(*param)
            threshold = 0.2
            if verbose:
                logging.info(f"trying {param}")
            history = repeat_experiment(value_function_estimator, repeat)
            # Find the first index in history where the error is below 0.2, using vectorization
            indices = np.where(history / history[0] < threshold)[0]
            # Check if we go below 0.2, and if so, check if we stay below 0.2 at the end
            if len(indices) > 0 and history[-1] / history[0] < threshold:
                index = indices[0]
            else:
                index = float("inf")

            results.append((param, index, history))
            
        return results

    parameter_combinations = []
    for alpha, beta, gamma in itertools.product(learning_rates, update_I_rates, update_D_rates):
        for N, M, L in itertools.product(learning_rates[alpha], update_I_rates[beta], update_D_rates[gamma]):
            parameter_combinations.append((alpha, N, beta, M, gamma, L))
    num_chunks = mp.cpu_count()
    chunked_params = [parameter_combinations[i::num_chunks] for i in range(num_chunks)]
    pool = mp.Pool()
    results = pool.map(try_params, chunked_params)
    pool.close()
    pool.join()

    # Combine results to find the best parameters
    minimum_index = float("inf")
    minimum_history = float("inf")
    minimum_params_index = None
    minimum_params_history = None
    for result in results:
        for params, index, history in result:
            if index < minimum_index:
                minimum_index = index
                minimum_params_index = params
            if history[-1] / history[0] < minimum_history:
                minimum_history = history[-1] / history[0]
                minimum_params_history = params

    # If we don't reach the threshold, return inf
    if minimum_index == float("inf"):
        logging.info("No parameters reached the threshold")
        logging.info(f"Minimum history: {minimum_history}")

        logging.info(f"The best parameters are {minimum_params_history}")
        return minimum_index, minimum_params_history
    
    logging.info(f"Minimum index: {minimum_index}")
    logging.info(f"The best parameters are {minimum_params_index}")

    return minimum_index, minimum_params_index


def repeat_experiment(value_function, num_times):
    """Run the experiment num_times times and return the average history.
    Take as input the parameters to the estimate_value_function function.

    Outputs must be of the same size.
    """
    average_history = 0

    for _ in range(num_times):
        history = value_function()
        average_history += np.array(history)

    average_history /= num_times

    return average_history


def find_Vpi(env, policy, gamma):
    """Find a good approximation of the value function of policy in an environment.
    """
    oracle = PolicyEvaluation(
        env.num_states,
        env.num_actions,
        env.build_policy_reward_vector(policy),
        env.build_policy_probability_transition_kernel(policy),
        1,0,0,0,0,
        gamma
    )
    return oracle.value_iteration(num_iterations=10000)


def find_Vstar(env, gamma):
    """Find a good approximation of the value function of the optimal policy in an environment.
    """
    oracle = Control(
        env.num_states,
        env.num_actions,
        env.build_reward_matrix(),
        env.build_probability_transition_kernel(),
        1,0,0,0,0,
        gamma
    )
    return oracle.value_iteration(num_iterations=10000)


def find_Qstar(env, gamma):
    """Find a good approximation of the value function of the optimal policy in an environment.
    """
    oracle = Control_Q(
        env.num_states,
        env.num_actions,
        env.build_reward_matrix(),
        env.build_probability_transition_kernel(),
        1,0,0,0,0,
        gamma
    )
    return oracle.value_iteration(num_iterations=100000)


def save_array(nparr, name, normalize=False, directory="", subdir=""):
    """Save nparr in a file with name name.
    Creates the npy and txt files if they don't exist to store the numpy arrays.
    If sub_dir is specified, we save the file in a specific subdirectory of npy,
    creating it if it doesn't exist.
    """
    if normalize:
        # Normalize the array by the first non-zero element:
        nparr = nparr / nparr[np.nonzero(nparr)[0][0]]

    if not os.path.isdir(f"{directory}/npy"):
        os.mkdir(f"{directory}/npy")
    if not os.path.isdir(f"{directory}/txt"):
        os.mkdir(f"{directory}/txt")

    if subdir != "":
        if not os.path.isdir(f"{directory}/npy/{subdir}"):
            os.mkdir(f"{directory}/npy/{subdir}")
        if not os.path.isdir(f"{directory}/txt/{subdir}"):
            os.mkdir(f"{directory}/txt/{subdir}")
        
        np.save(f"{directory}/npy/{subdir}/" + name + ".npy", nparr)
        np.savetxt(f"{directory}/txt/{subdir}/" + name + ".txt", nparr)
    else:
        np.save(f"{directory}/npy/" + name + ".npy", nparr)
        np.savetxt(f"{directory}/txt/" + name + ".txt", nparr)


def plot_comparison(fig, ax1, ax2, title1, title2, ylabel, show_fig=True, fig_name="plot", is_log=False):
    """Configure and plot a comparison between the learning
    of two algorithms given pyplot objects"""
    plt.subplots_adjust(hspace=0.7)
    fig.set_figheight(10)
    fig.set_figwidth(10)

    ax1.set_title(title1)
    ax2.set_title(title2)

    if is_log:
        ax1.set_yaxis('log')

    ax1.legend()
    ax2.legend()
    ax1.set(xlabel='Iteration', ylabel=ylabel)
    ax2.set(xlabel='Iteration', ylabel=ylabel)

    fig.savefig(fig_name)
    if show_fig: plt.show()

def create_label(ax, norm, normalize, is_q, is_v_star=False):
    if is_q:
        current = 'Q_k'
        goal = 'Q^*'
        start = 'Q_0'
    else:
        current = 'V_k'
        if is_v_star:
            goal = 'V^*'
        else:
            goal = 'V^\pi'
        start = 'V_0'
    if norm == 'inf':
        if normalize:
            ax.set_ylabel(f'$\\frac{{||{current} - {goal}||_{{\infty}}}}{{||{start} - {goal}||_{{\infty}}}}$')
        else:
            ax.set_ylabel(f'$||{current} - {goal}||_{{\infty}}$')
    if type(norm) == str and norm[:4] == 'diff':
        state = norm[5:]
        if normalize:
            ax.set_ylabel(f'$\\frac{{{current}[{state}] - {goal}[{state}]}}{{{start}[{state}] - {goal}[{state}]}}$')
        else:
            ax.set_ylabel(f'${current}[{state}] - {goal}[{state}]$')
        ax.axhline(y=0, color='k', linestyle='--')
    else:
        if normalize:
            ax.set_ylabel(f'$\\frac{{||{current} - {goal}||_{{{norm}}}}}{{||{start} - {goal}||_{{{norm}}}}}$')
        else:
            ax.set_ylabel(f'$||{current} - {goal}||_{{{norm}}}$')
