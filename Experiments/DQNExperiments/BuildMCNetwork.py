import hydra
import numpy as np
import torch as th

from wandb.integration.sb3 import WandbCallback
import wandb

from Experiments.ExperimentHelpers import *
from TabularPID.AgentBuilders.DQNBuilder import build_PID_DQN, build_PID_FQI, build_gain_adapter
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import evaluation
from stable_baselines3.common.utils import set_random_seed

# Warning: Change this when running on a different machine
base_directory = "/h/bedaywim/PID-Accelerated-TD-Learning"

@hydra.main(version_base=None, config_path="../../config/DQNExperiments", config_name="DQNExperiment")
def control_experiment(cfg):
    """Attempt to replicate results in figure 2 of PID Accelerated VI"""
    logging.info(f"The chosen seed is: {cfg['seed']}")

    # Create a prg with this seed
    seed_prg = np.random.RandomState(cfg['seed'])

    log_name = f"{cfg['kp']} {cfg['ki']} {cfg['kd']}{'*' if cfg['tabular_d'] else ''} {cfg['alpha']} {cfg['beta']} {cfg['d_tau']}" \
          + f" {cfg['gain_adapter']} {cfg['epsilon']} {cfg['meta_lr']}"

    env_cfg = next(iter(cfg['env'].values()))
    # Adaptation configs for logging
    log_cfg = {
        'kp': cfg['kp'], 'ki': cfg['ki'], 'kd': cfg['kd'],
        'alpha': cfg['alpha'], 'beta': cfg['beta'], 'd_tau': cfg['d_tau'],
        'epsilon': cfg['epsilon'], 'meta_lr': cfg['meta_lr'],
        'use_previous_BRs': cfg['use_previous_BRs'], 'gain_adapter': cfg['gain_adapter']
    }.update(env_cfg)

    run_seed = seed_prg.randint(0, 2**32)
    set_random_seed(run_seed)

    if cfg['FQI']:
        agent = build_PID_FQI(
            env_cfg['env'], env_cfg['gamma'], env_cfg['optimizer'],
            env_cfg['replay_memory_size'], env_cfg['batch_size'], env_cfg['learning_rate'],
            env_cfg['initial_eps'], env_cfg['exploration_fraction'],
            env_cfg['minimum_eps'], env_cfg['gradient_steps'], env_cfg['train_freq'],
            env_cfg['target_update_interval'], env_cfg['inner_size'],
            cfg['slow_motion'], env_cfg['learning_starts'],
            tensorboard_log=cfg['tensorboard_log'],
            seed=run_seed, log_name=log_name,
        )
    else:
        agent = build_PID_DQN(
            env_cfg['env'], env_cfg['gamma'], env_cfg['optimizer'],
            env_cfg['replay_memory_size'], env_cfg['batch_size'], env_cfg['learning_rate'],
            env_cfg['tau'], env_cfg['initial_eps'], env_cfg['exploration_fraction'],
            env_cfg['minimum_eps'], env_cfg['gradient_steps'], env_cfg['train_freq'],
            env_cfg['target_update_interval'], env_cfg['inner_size'],
            cfg['slow_motion'], env_cfg['learning_starts'], tabular_d=cfg['tabular_d'],
            tensorboard_log=cfg['tensorboard_log'], seed=run_seed,
            log_name=log_name, name_append=f"run {i}", should_stop=cfg['should_stop']
        )

    run = wandb.init(
        project="PID Accelerated RL",
        config=log_cfg,
        save_code=True,
        group=f"{cfg['experiment_name']}-{str(cfg['seed'])}",
        job_type=log_name,
        reinit=True,
        name=str(run_seed),
        sync_tensorboard=True,
    )

    agent = agent.learn(
        total_timesteps=env_cfg['num_iterations'],
        log_interval=cfg['log_interval'],
        progress_bar=cfg['progress_bar'],
        tb_log_name=log_name,
        callback=[
            WandbCallback(verbose=2),
            CustomEvalCallback(get_model(env_cfg['env']))
        ]
    )
    run.finish()

    agent.visualize_episode()

def get_model(env_name):
    """Return the model with the same env_name from the models directory"""
    # The model is in a directory that starts with the same name as the environment
    model_dir = next(iter(filter(lambda x: x.startswith(env_name), os.listdir(f"{base_directory}/models"))))
    return DQN.load(f"{base_directory}/models/{model_dir}/{model_dir}.zip")


class CustomEvalCallback(BaseCallback):
    def __init__(self, trained_model, verbose=0):
        super(CustomEvalCallback, self).__init__(verbose)
        self.trained_model = trained_model

    def _on_step(self) -> bool:
        if self.num_timesteps % 250 != 0:
            return True

        mean, std = evaluation.evaluate_policy(self.model, self.model.get_env(), n_eval_episodes=10)
        self.logger.record("eval/mean_reward", mean)
        self.logger.record("eval/std_reward", std)

        if self.model.gain_adapter.adapts_single_gains:
            self.logger.record("eval/k_p", self.model.gain_adapter.kp)
            self.logger.record("eval/k_i", self.model.gain_adapter.ki)
            self.logger.record("eval/k_d", self.model.gain_adapter.kd)

        # Sample 20 points from the replay buffer
        states, actions, *_ = self.model.replay_buffer.sample(20)

        # Evaluate the average distance between the Q values of the model and the trained_model
        with th.no_grad():
            q_values = self.model.policy.q_net(states)
            q_values = th.gather(q_values, dim=1, index=actions.long())
            trained_q_values = self.trained_model.policy.q_net(states)
            trained_q_values = th.gather(trained_q_values, dim=1, index=actions.long())
        self.logger.record(
            "eval/distance_from_optimal_values",
            th.mean(th.linalg.vector_norm(q_values - trained_q_values)).item()
        )

if __name__ == "__main__":
    control_experiment()