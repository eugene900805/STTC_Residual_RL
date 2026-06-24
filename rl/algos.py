"""Algorithm factory so SAC / PPO / TD3 / DDPG can be trained and evaluated
through one code path, with identical observation, action, network size and
training budget. This keeps the head-to-head comparison fair: the only thing
that changes between runs is the RL algorithm itself.
"""
import numpy as np
from stable_baselines3 import SAC, PPO, TD3, DDPG
from stable_baselines3.common.noise import NormalActionNoise

ALGOS = {"sac": SAC, "ppo": PPO, "td3": TD3, "ddpg": DDPG}

# shared across all algorithms for a fair comparison
NET_ARCH = [256, 256]
LR = 3e-4
GAMMA = 0.99


def make_model(algo, venv, seed=0, n_actions=2):
    """Build an SB3 model. Hyper-parameters are matched where the algorithms
    share them; algorithm-specific knobs use sensible SB3 defaults."""
    algo = algo.lower()
    cls = ALGOS[algo]
    # device='cpu': the policies are tiny MLPs, so the bottleneck is env
    # stepping, not the net; CPU is faster and keeps all algos on equal footing.
    common = dict(policy="MlpPolicy", env=venv, learning_rate=LR, gamma=GAMMA,
                  seed=seed, verbose=1, device="cpu",
                  policy_kwargs=dict(net_arch=NET_ARCH))

    if algo == "ppo":
        # on-policy: no replay buffer; collect rollouts then update
        return PPO(**common, n_steps=2048, batch_size=256, n_epochs=10,
                   gae_lambda=0.95, ent_coef=0.0)

    # off-policy shared settings
    off = dict(buffer_size=300_000, batch_size=256, tau=0.005,
               train_freq=1, gradient_steps=1, learning_starts=5_000)
    if algo == "sac":
        return SAC(**common, **off)

    # TD3 / DDPG benefit from exploration noise on a deterministic policy
    action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                     sigma=0.1 * np.ones(n_actions))
    return cls(**common, **off, action_noise=action_noise)


def load_model(algo, path):
    return ALGOS[algo.lower()].load(path)
