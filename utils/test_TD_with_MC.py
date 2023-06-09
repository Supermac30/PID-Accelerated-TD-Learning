from Agents import MonteCarloPE, Hard_PID_TD
from Experiments.ExperimentHelpers import *

policy, env = garnet_problem(50, 4, 3, 5, -1)

agent1 = MonteCarloPE(policy, env, 0.99)
agent2 = Hard_PID_TD(policy, env, 0.99, learning_rate_function(1, 0))

agent1.estimate_value_function(10000)

controller = P_Controller(np.identity(50))

agent2.estimate_value_function(
    controller,
    num_iterations=10000,
    test_function=test_function,
    stop_if_diverging=True
)