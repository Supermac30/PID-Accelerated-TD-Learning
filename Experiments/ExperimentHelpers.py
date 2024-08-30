import numpy as np
import itertools
import os
import logging
import multiprocess as mp
from pathlib import Path
import pickle

from TabularPID.EmpericalTester import get_optimal_Q, get_optimal_TD, get_optimal_TD_Q


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

    elif norm == "BR":
        return lambda V, Vp, BR: np.max(BR)
    
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
        return lambda V, Vp, BR: np.linalg.norm(V - V_pi, ord=norm)


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

    # Don't show info logging
    # logging.getLogger().setLevel(logging.WARNING)

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

    # Show info logging again
    logging.getLogger().setLevel(logging.INFO)


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
    return get_optimal_TD(env, policy, gamma).V

def find_Qpi(env, policy, gamma):
    return get_optimal_TD_Q(env, policy, gamma).Q

def find_Vstar(env, gamma):
    """Find a good approximation of the value function of the optimal policy in an environment.
    """
    return get_optimal_Q(env, gamma).Q.max(axis=1)

def find_Qstar(env, gamma):
    """Find a good approximation of the value function of the optimal policy in an environment.
    """
    return get_optimal_Q(env, gamma).Q


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
        Path(f"{directory}/npy").mkdir(parents=True, exist_ok=True)
    if not os.path.isdir(f"{directory}/txt"):
        Path(f"{directory}/txt").mkdir(parents=True, exist_ok=True)

    if subdir != "":
        if not os.path.isdir(f"{directory}/npy/{subdir}"):
            Path(f"{directory}/npy/{subdir}").mkdir(parents=True, exist_ok=True)
        if not os.path.isdir(f"{directory}/txt/{subdir}"):
            Path(f"{directory}/txt/{subdir}").mkdir(parents=True, exist_ok=True)
        
        np.save(f"{directory}/npy/{subdir}/" + name + ".npy", nparr)
        np.savetxt(f"{directory}/txt/{subdir}/" + name + ".txt", nparr)
    else:
        np.save(f"{directory}/npy/" + name + ".npy", nparr)
        np.savetxt(f"{directory}/txt/" + name + ".txt", nparr)


def create_label(ax, norm, normalize, is_q, is_star=False, fontsize=None):
    if is_q:
        current = 'Q_t'
        if is_star:
            goal = 'Q*'
        else:
            goal = "Q^\pi"
    else:
        current = 'V_t'
        if is_star:
            goal = 'V*'
        else:
            goal = 'V^\pi'
    if norm == 'inf':
        if normalize:
            if fontsize is None:
                ax.set_ylabel(f'Normalized $||{current} - {goal}||_{{\infty}}$')
            else:
                ax.set_ylabel(f'Normalized $||{current} - {goal}||_{{\infty}}$', fontsize=fontsize)
        else:
            if fontsize is None:
                ax.set_ylabel(f'$||{current} - {goal}||_{{\infty}}$')
            else:
                ax.set_ylabel(f'$||{current} - {goal}||_{{\infty}}$', fontsize=fontsize)
    if norm == 'fro':
        norm = 'F'
    if norm == "BR":
        if normalize:
            if fontsize is None:
                ax.set_ylabel(f'Normalized Bellman Residual')
            else:
                ax.set_ylabel(f'Normalized Bellman Residual', fontsize=fontsize)
        else:
            if fontsize is None:
                ax.set_ylabel(f'Bellman Residual')
            else:
                ax.set_ylabel(f'Bellman Residual', fontsize=fontsize)
    if type(norm) == str and norm[:4] == 'diff':
        state = norm[5:]
        if normalize:
            if fontsize is None:
                ax.set_ylabel(f'Normalized ${current}[{state}] - {goal}[{state}]$')
            else:
                ax.set_ylabel(f'Normalized ${current}[{state}] - {goal}[{state}]$', fontsize=fontsize)
        else:
            if fontsize is None:
                ax.set_ylabel(f'${current}[{state}] - {goal}[{state}]$')
            else:
                ax.set_ylabel(f'${current}[{state}] - {goal}[{state}]$', fontsize=fontsize)
        ax.axhline(y=0, color='k', linestyle='--')
    else:
        if normalize:
            if fontsize is None:
                ax.set_ylabel(f'Normalized $||{current} - {goal}||_{{{norm}}}$')
            else:
                ax.set_ylabel(f'Normalized $||{current} - {goal}||_{{{norm}}}$', fontsize=fontsize)
        else:
            if fontsize is None:
                ax.set_ylabel(f'$||{current} - {goal}||_{{{norm}}}$')
            else:
                ax.set_ylabel(f'$||{current} - {goal}||_{{{norm}}}$', fontsize=fontsize)


def save_time(num_states, mean, std_dev, directory, model_name, subdir="time"):
    """Save the time taken in a file storing time if it exists, otherwise create one."""
    path = f"{directory}/{subdir}/{model_name}.pkl"
    if not os.path.isdir(f"{directory}/{subdir}"):
        Path(f"{directory}/{subdir}").mkdir(parents=True, exist_ok=True)
        time_data = {num_states: (mean, std_dev)}
    else:
        with open(path, 'rb') as file:
            time_data = pickle.load(file)
        time_data[num_states] = (mean, std_dev)
    
    with open(path, 'wb') as file:
        pickle.dump(time_data, file)
