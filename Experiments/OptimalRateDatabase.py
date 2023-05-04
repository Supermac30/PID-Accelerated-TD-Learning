import pickle
FILE_NAME = "Experiments/optimal_learning_rates.pickle"

def get_stored_optimal_rate(model, env_name, gamma):
    """If the optimal rate is in FILE_NAME then return it,
    otherwise return None

    model: A consistent description of the model name, either a string or tuple
    env_name: A consistent description of the environment, a string

    If the file is not found, raise a FileNotFoundException
    """
    with open(FILE_NAME, 'rb') as f:
        optimal_rates = pickle.load(f)
    if (model, env_name, gamma) in optimal_rates:
        return optimal_rates[(model, env_name, gamma)]
    return None

def store_optimal_rate(model, env_name, optimal_rate, gamma):
    """Store the optimal rate in FILE_NAME.

    model: A consistent description of the model name, either a string or tuple
    env_name: A consistent description of the environment, a string
    optimal_rate: A tuple of learning rates for each component

    If the file is not found, raise a FileNotFoundException
    """
    with open(FILE_NAME, 'rb') as f:
        optimal_rates = pickle.load(f)

    optimal_rates[(model, env_name, gamma)] = optimal_rate

    # save the updated dictionary to the same pickle file
    with open(FILE_NAME, 'wb') as f:
        pickle.dump(optimal_rates, f, protocol=pickle.HIGHEST_PROTOCOL)