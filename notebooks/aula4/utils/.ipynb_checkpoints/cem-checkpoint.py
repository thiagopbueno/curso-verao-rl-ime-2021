import json
import os
import sys
from os import path
from typing import Callable
from typing import Optional

import gym
import numpy as np
from gym import logger
from gym import wrappers
from tqdm import tqdm
from tqdm.notebook import tqdm as nb_tqdm


class BinaryActionLinearPolicy:
    def __init__(self, theta: np.ndarray):
        # from vector to row vector
        self.weight = np.expand_dims(theta[:-1], axis=0)
        self.bias = np.atleast_2d(theta[-1])

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        # from vector to column vector
        obs = np.expand_dims(obs, axis=-1)
        score = self.weight @ obs + self.bias
        # from 1x1 matrix to scalar
        act = score[..., 0, 0] < 0
        act = act.astype(np.int32)
        return act


def cem(
    f: Callable[[np.ndarray], float],
    th_mean: np.ndarray,
    batch_size: int,
    n_iter: int,
    elite_frac: float,
    initial_std: float = 1.0,
    seed: Optional[int] = None,
):
    """
    Generic implementation of the cross-entropy method for maximizing a black-box function
    Args:
        f: a function mapping from vector -> scalar
        th_mean: initial mean over input distribution
        batch_size: number of samples of theta to evaluate per batch
        n_iter: number of batches
        elite_frac: each batch, select this fraction of the top-performing samples
        initial_std: initial standard deviation over parameter vectors
        seed: random number generator seed
    returns:
        A generator of dicts. Subsequent dicts correspond to iterations of CEM algorithm.
        The dicts contain the following values:
        'ys' :  numpy array with values of function evaluated at current population
        'ys_mean': mean value of function over current population
        'theta_mean': mean value of the parameter vector over current population
    """
    n_elite = int(np.round(batch_size * elite_frac))
    th_std = np.ones_like(th_mean) * initial_std
    rng = np.random.default_rng(seed)

    for _ in range(n_iter):
        perturbations = th_std[None, :] * rng.standard_normal(
            (batch_size, th_mean.size)
        )
        ths = np.array([th_mean + dth for dth in perturbations])
        ys = np.array([f(th) for th in ths])
        elite_inds = ys.argsort()[::-1][:n_elite]
        elite_ths = ths[elite_inds]
        th_mean = elite_ths.mean(axis=0)
        th_std = elite_ths.std(axis=0)
        yield {"ys": ys, "theta_mean": th_mean, "y_mean": ys.mean()}


def do_rollout(policy, env, num_steps, render=False):
    total_rew = 0
    ob = env.reset()
    for t in range(num_steps):
        a = policy(ob)
        (ob, reward, done, _info) = env.step(a)
        total_rew += reward
        if render and t % 3 == 0:
            env.render()
        if done:
            break
    return total_rew, t + 1


def get_cem_policy(
    env: gym.Env, n_iter: int = 10, seed: Optional[int] = None, notebook: bool = False
) -> BinaryActionLinearPolicy:
    def noisy_eval(theta: np.ndarray) -> float:
        policy = BinaryActionLinearPolicy(theta)
        rew, T = do_rollout(policy, env, num_steps=500)
        return rew

    env.seed(seed)
    params = dict(n_iter=n_iter, batch_size=25, elite_frac=0.2, seed=seed)
    iterable = cem(noisy_eval, np.zeros(env.observation_space.shape[0] + 1), **params)
    iterable = (nb_tqdm if notebook else tqdm)(enumerate(iterable), total=n_iter)
    for (i, iterdata) in iterable:
        tqdm.write(f"Iteration {i:2d}. Episode mean reward: {iterdata['y_mean']:7.3f}")

    policy = BinaryActionLinearPolicy(iterdata["theta_mean"])
    return policy


def main():
    seed = 1
    env = gym.make("CartPole-v1")
    env.seed(seed)
    params = dict(n_iter=10, batch_size=25, elite_frac=0.2, seed=seed)
    num_steps = 500

    def noisy_evaluation(theta):
        policy = BinaryActionLinearPolicy(theta)
        rew, T = do_rollout(policy, env, num_steps)
        return rew

    # Train the agent, and snapshot each stage
    iterable = cem(
        noisy_evaluation, np.zeros(env.observation_space.shape[0] + 1), **params
    )
    iterable = tqdm(enumerate(iterable), total=params["n_iter"])
    for (i, iterdata) in iterable:
        tqdm.write(f"Iteration {i:2d}. Episode mean reward: {iterdata['y_mean']:7.3f}")
    #         policy = BinaryActionLinearPolicy(iterdata["theta_mean"])
    #         do_rollout(policy, env, num_steps, render=True)

    env.close()


if __name__ == "__main__":
    main()
