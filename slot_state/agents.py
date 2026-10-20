"""The three actors: FDEdge with feedback on, and the public SAC and DQN baselines.

Hyperparameters are those of the public mains (third_party/fdedge/fdedge_main.py and baselines/*/*_main.py).
FDEdge differs from the released code only where the feedback input is switched on; see FeedbackDiffusion.
"""
from functools import cache

import numpy as np
import torch

from . import config, upstream

HYPERPARAMETERS = {
    'fdedge': dict(actor_lr=1e-4, critic_lr=1e-3, alpha=0.05, alpha_lr=3e-4, hidden_dim=128, gamma=0.95, tau=0.005,
                   train_buffer_size=10000, batch_size=64, target_entropy=-1, denoising_steps=5),
    'sac': dict(actor_lr=1e-4, critic_lr=1e-3, alpha=0.01, alpha_lr=1e-3, hidden_dim=128, gamma=0.9, tau=0.005,
                train_buffer_size=10000, batch_size=64, target_entropy=-1),
    'dqn': dict(learn_rate=1e-3, hidden_dim=128, gamma=0.95, target_update=200, epsilon=0.99,
                train_buffer_size=10000, batch_size=64),
}
# One update per slot once the replay buffer holds more transitions than this (public training loops).
WARMUP = {'fdedge': 500, 'sac': 300, 'dqn': 500}


@cache
def _fdedge_classes():
    diffusion = upstream.fdedge('feedback_diffusion')
    model = upstream.fdedge('fdsac_model')

    class FeedbackDiffusion(diffusion.Diffusion):
        """Diffusion actor whose reverse process starts from the stored action probabilities of the same
        (slot, position), as the FDEdge paper describes. The released code builds this input but starts from noise."""

        def p_sample_loop(self, state, latent_action_probs):
            batch_size = state.shape[0]
            x = latent_action_probs
            for i in reversed(range(self.n_timesteps)):
                timesteps = torch.full((batch_size,), i, device=self.betas.device, dtype=torch.long)
                x = self.p_sample(x, timesteps, state)
            return x

    class FeedbackReplayBuffer(model.ReplayBuffer):
        """Stores copies, because the feedback bank that the stored probabilities come from is updated in place."""

        def add(self, state, action, latent_action_probs, reward, next_state, next_latent_action_probs):
            super().add(np.array(state, copy=True), action, np.array(latent_action_probs, copy=True), reward,
                        np.array(next_state, copy=True), np.array(next_latent_action_probs, copy=True))

    class FDEdgeAgent(model.FDSAC):
        def __init__(self, device):
            h = HYPERPARAMETERS['fdedge']
            super().__init__(config.STATE_DIM, h['hidden_dim'], config.NUM_SERVERS, h['actor_lr'], h['critic_lr'],
                             h['alpha'], h['alpha_lr'], h['target_entropy'], h['tau'], h['gamma'],
                             h['denoising_steps'], device)
            # Replacing the actor after the public constructor keeps the random draws, and thus the initial
            # weights of a training seed, identical to the runs reported in the paper.
            self.actor = fdedge_policy(device)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=h['actor_lr'])

        def take_action(self, state, latent_action_probs):
            with torch.no_grad():
                probs = self.actor(torch.as_tensor(np.asarray([state]), dtype=torch.float32, device=self.device),
                                   torch.as_tensor(np.asarray([latent_action_probs]), dtype=torch.float32,
                                                   device=self.device))
                if not torch.isfinite(probs).all():
                    raise RuntimeError('Non-finite action probabilities')
                return int(probs.argmax(dim=-1).item()), probs.cpu().numpy()

    return FeedbackDiffusion, FeedbackReplayBuffer, FDEdgeAgent, model


def fdedge_policy(device):
    FeedbackDiffusion, _, _, model = _fdedge_classes()
    h = HYPERPARAMETERS['fdedge']
    return FeedbackDiffusion(config.STATE_DIM, config.NUM_SERVERS,
                             model.MLP_PolicyNet(config.STATE_DIM, h['hidden_dim'], config.NUM_SERVERS),
                             beta_schedule='vp', denoising_steps=h['denoising_steps']).to(device)


def make_agent(actor, device):
    """-> (agent, replay buffer) as built by the public training code."""
    h = HYPERPARAMETERS[actor]
    if actor == 'fdedge':
        _, FeedbackReplayBuffer, FDEdgeAgent, _ = _fdedge_classes()
        return FDEdgeAgent(device), FeedbackReplayBuffer(h['train_buffer_size'])
    model = upstream.baseline(actor, 'model')
    if actor == 'sac':
        agent = model.SAC(config.STATE_DIM, h['hidden_dim'], config.NUM_SERVERS, h['actor_lr'], h['critic_lr'], h['alpha'],
                          h['alpha_lr'], h['target_entropy'], h['tau'], h['gamma'], device)
    else:
        agent = model.DQN(config.STATE_DIM, h['hidden_dim'], config.NUM_SERVERS, h['learn_rate'], h['gamma'], h['epsilon'],
                          h['target_update'], device, e_greedy_increment=None)
    return agent, model.ReplayBuffer(h['train_buffer_size'])


# The network that chooses actions at evaluation (argmax of its output).
POLICY_ATTRIBUTE = {'fdedge': 'actor', 'sac': 'actor', 'dqn': 'q_net'}


def policy_network(actor, device):
    if actor == 'fdedge':
        return fdedge_policy(device)
    model = upstream.baseline(actor, 'model')
    network = model.PolicyNet if actor == 'sac' else model.Qnet
    return network(config.STATE_DIM, HYPERPARAMETERS[actor]['hidden_dim'], config.NUM_SERVERS).to(device)
