import functools
import json
import time
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

import spinup.algos.pytorch.trpo.core as core
from spinup.utils.mpi_tools import proc_id, mpi_statistics_scalar


class VPGBuffer:
    """
    A buffer for storing trajectories experienced by a VPG agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs  # observation o_t
        self.act_buf[self.ptr] = act  # action a_t
        self.rew_buf[self.ptr] = rew  # reward r_t
        self.val_buf[self.ptr] = val  # value function v(o_t)
        self.logp_buf[self.ptr] = logp  # log likelihood of a_t
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        # from o_t (s_t), we get action a_t, and critic's value v(o_t)
        # apply action a_t to env to get o_t+1 (s_t+1), rews_t (not rews_t+1)

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = self.rew_buf[path_slice]
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        # delta_t = r(o_t, a_t) + gamma * v(o_t+1) - v(o_t)
        deltas = rews + self.gamma * vals[1:] - vals[:-1]  # each deltas[t] is TD error at t.
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)  # advantages

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)  # discounted sum of rewards for an episode

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


def trpo(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
         steps_per_epoch=4000, epochs=50, gamma=0.99, delta=0.01, vf_lr=1e-3,
         train_v_iters=80, damping_coeff=0.1, cg_iters=10, backtrack_iters=10,
         backtrack_coeff=0.8, lam=0.97, max_ep_len=1000, save_freq=10, algo='trpo'):

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn(max_ep_len)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    # log hyperparameters
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])

    # setup tensorboard
    writer = SummaryWriter(f'runs/{env.spec.id}-{datetime.now().strftime("%y-%m-%d %H%M%S")}')
    writer.add_text('hyperparameters',
        json.dumps(dict(env=env.spec.id, seed=seed,
                        vf_lr=vf_lr, gamma=gamma, lam=lam, delta=delta,
                        damping_coeff=damping_coeff, cg_iters=cg_iters,
                        backtrack_iters=backtrack_iters, backtrack_coeff=backtrack_coeff,
                        pi_n_param=var_counts[0].item(), v_n_param=var_counts[1].item(),
                        ac_kwargs=','.join(str(i) for i in ac_kwargs['hidden_sizes']),
                        steps_per_epoch=steps_per_epoch, epochs=epochs, max_ep_len=max_ep_len,
                        train_v_iters=train_v_iters)))

    # Set up experience buffer
    buf = VPGBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret) ** 2).mean()  # least square

    def cg(Ax, b):
        """
        Conjugate gradient algorithm
        (see https://en.wikipedia.org/wiki/Conjugate_gradient_method)
        """
        x = torch.zeros_like(b)
        r = b.clone().detach() # Note: should be 'b - Ax(x)', but for x=0, Ax(x)=0. Change if doing warm start.
        p = r.clone().detach()
        r_dot_old = (r * r).sum()
        for _ in range(cg_iters):
            z = Ax(p)
            alpha = r_dot_old / ((p * z).sum() + core.EPS)
            x += alpha * p
            r -= alpha * z
            r_dot_new = (r * r).sum()
            p = r + (r_dot_new / r_dot_old) * p
            r_dot_old = r_dot_new
        return x

    def Hvp(old_pi, obs, act, v):
        """Compute product of Fv, without explicitly storing and computing F,
        where F is the Fisher Information Matrix, v is the given gradient vector.
        F is acquired via second order Taylor of KL-divergence D_kl(old||new),
        evaluated at the old policy.
        """
        ac.pi.zero_grad()
        new_pi, _ = ac.pi(obs, act)
        kl = core.diagonal_gaussian_kl(old_pi, new_pi)
        grads = torch.autograd.grad(kl, ac.pi.parameters(),
                                    create_graph=True)  # create_graph to compute higher order gradient
        kl_grads = core.flat_concat(grads)
        # by chain rule, d/dp(kl_v) = d/dp(kl_grads) @ v + kl_grads @ d/dp(v)
        # d/dp(v) i.e., second oder of loss can be ignored,
        # so we have hvp = d/dp(kl_grads) @ v
        kl_v = (kl_grads * v).sum()
        hvp = torch.autograd.grad(kl_v, ac.pi.parameters())  # Hessian vector product
        hvp_flat = core.flat_concat(hvp).clone().detach()
        return hvp_flat + v * damping_coeff

    def update_module_param(module, new_params):
        params = core.flat_concat(module.parameters())
        params.copy_(new_params)

    def set_flat_params_to(model, flat_params):
        prev_ind = 0
        for param in model.parameters():
            flat_size = int(np.prod(list(param.size())))
            param.data.copy_(
                flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
            prev_ind += flat_size

    def linesearch(max_step, current_pi, current_params, current_loss, compute_loss):
        for i in range(backtrack_iters):
            new_params = current_params - backtrack_coeff**i * max_step
            set_flat_params_to(ac.pi, new_params)
            new_pi, new_loss = compute_loss()  # recompute loss
            if new_loss <= current_loss:
                kl = core.diagonal_gaussian_kl(current_pi, new_pi)  # recompute kl-divergence
                if kl <= delta:
                    return True
        # restore old params
        update_module_param(ac.pi, current_params)
        return False

    def compute_pi_loss(obs, act, adv, old_logp, grad_enabled=False):
        if not grad_enabled:
            with torch.no_grad():
                pi, logp = ac.pi(obs, act)
        else:
            ac.pi.zero_grad()
            pi, logp = ac.pi(obs, act)

        return pi, -(torch.exp(logp - old_logp) * adv).mean()

    def update(epoch):
        data = buf.get()
        obs, act, ret, adv = data['obs'], data['act'], data['ret'], data['adv']

        with torch.no_grad():
            old_pi, old_logp = ac.pi(obs, act)

        _, loss_pi = compute_pi_loss(obs, act, adv, old_logp, True)
        grads = torch.autograd.grad(loss_pi, ac.pi.parameters())
        loss_grad = core.flat_concat(grads).clone().detach()

        hvp = functools.partial(Hvp, old_pi, obs, act)
        s = cg(hvp, loss_grad)
        beta = torch.sqrt(2 * delta / (s * hvp(s)).sum())
        old_params = core.flat_concat(ac.pi.parameters()).clone().detach()
        result = linesearch(beta * s, old_pi, old_params, loss_pi,
                   functools.partial(compute_pi_loss, obs, act, adv, old_logp, False))
        print(f'line search was successful: {result}')

        # Value function learning
        for i in range(train_v_iters):  # multiple steps on value function
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            vf_optimizer.step()

        writer.add_scalars('Pi Loss', dict(LossPi=loss_pi), epoch)
        writer.add_scalars('Value Loss', dict(LossV=loss_v), epoch)
        #writer.add_scalar('KL divergence', kl, epoch)
        #writer.add_scalar('Entropy', ent, epoch)

    # Set up optimizers for policy and value function
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    # Prepare for interaction with environment
    start_time = time.time()
    (o, _), ep_ret, ep_len = env.reset(), 0, 0  # ep_ret: undiscounted return of episode

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(steps_per_epoch):
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))

            next_o, r, terminated, truncated, _ = env.step(a)
            d = terminated or truncated  # done
            ep_ret += r
            ep_len += 1

            # save and log
            buf.store(o, a, r, v, logp)
            #logger.store(VVals=v)

            # Update obs (critical!)
            o = next_o

            epoch_ended = (t == (steps_per_epoch - 1))
            if d or epoch_ended:
                if epoch_ended and not d:
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if truncated or epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                buf.finish_path(v)
                n_step = epoch * steps_per_epoch + t
                writer.add_scalar('Episode Return', ep_ret, n_step)
                writer.add_scalar('Episode Length', ep_len, n_step)
                (o, _), ep_ret, ep_len = env.reset(), 0, 0

        # Save model
        # if (epoch % save_freq == 0) or (epoch == epochs - 1):
        #     logger.save_state({'env': env}, None)

        update(epoch)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Reacher-v4')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='trpo')
    args = parser.parse_args()

    trpo(lambda x: gym.make(args.env, max_episode_steps=x), actor_critic=core.MLPActorCritic,
         ac_kwargs=dict(hidden_sizes=[args.hid] * args.l), gamma=args.gamma,
         seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs)
