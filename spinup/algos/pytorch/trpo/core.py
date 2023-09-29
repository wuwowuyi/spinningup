import numpy as np
import scipy.signal
from gymnasium.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal


EPS = 1e-8


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def flat_concat(xs):
    return torch.cat([x.view(-1) for x in xs])


def diagonal_gaussian_kl(old_pi, new_pi):
    """
    From
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions,
    we can see KL-divergence between two diagonal Gaussians is simply the sum of k pairs univariate Gaussians.
    The KL between two univariate Gaussians is: (var0 + (mu0 - mu1)^2)/2*var1  - 0.5 + log(std1/std0)
    """
    mu0, std0 = old_pi.mean, old_pi.stddev
    mu1, std1 = new_pi.mean, new_pi.stddev
    pre_sum = (std0**2 + (mu1 - mu0)**2)/(2 * std1**2 + EPS) - 0.5 + torch.log(std1) - torch.log(std0)
    all_kls = pre_sum.sum(dim=1)
    return all_kls.mean()


class Actor(nn.Module):
    """Policy net (Actor). """

    def _distribution(self, obs):
        """
        Generate action distribution with given observation.

        :param obs: obveration
        :return: action distribution
        """
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        """
        Compute log-likelihood of given action act.

        :param pi: Probability distribution of actions.
        :param act: an action
        :return: log likelihood of act.
        """
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)  # initial std. e^(-0.5)=0.6
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):
    """Value function (Critic).
    """

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)  # Critical to ensure v has right shape.


class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            raise Exception("Support of discrete action space is not implemented.")

        # build value function
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        """
        Sample an action based on observation obs.
        Returned values are all in numpy arrays.

        :param obs: observation
        :return:
         - action proposed
         - value of observation from critic
         - log likelihood of action proposed
        """
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()  # sample an action from actor
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)  # sample value from critic
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]
