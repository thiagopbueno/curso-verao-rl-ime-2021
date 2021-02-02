from typing import Callable
from typing import List
from typing import Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython.display import HTML
from matplotlib import animation
from matplotlib import rc

rc("animation", html="html5")


def cast_obs(obs: np.ndarray, env: gym.Env) -> np.ndarray:
    return obs.astype(env.observation_space.dtype)


def display_frames_and_q_values(
    env: gym.Env,
    policy: Callable[[tf.Tensor], tf.Tensor],
    q_func: Callable[[tf.Tensor], tf.Tensor],
    *,
    actions: List[str],
    **kwargs
):
    figsize = kwargs.get("figsize", (17, 5))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    def _generate_episode() -> Tuple[List[np.ndarray], List[np.ndarray]]:
        frames: List[np.ndarray] = []
        observations: List[np.ndarray] = []

        obs = cast_obs(env.reset(), env)
        done = False

        while not done:
            frame = env.render(mode="rgb_array")
            frames.append(frame)
            observations.append(obs)

            action = policy(np.expand_dims(tf.convert_to_tensor(obs), axis=0))[0]
            obs, reward, done, _ = env.step(action.numpy())
            obs = cast_obs(obs, env)

        env.close()

        return frames, observations

    def _plot_fn(inputs: Tuple[np.ndarray, np.ndarray]):
        frame, obs = inputs

        ax1.clear()
        ax2.clear()

        ax1.axis("off")
        ax2.set_xlabel("Actions")

        ax1.imshow(frame)

        q_values = q_func(np.expand_dims(obs, axis=0))[0]
        best_action = np.argmax(q_values)
        barlist = ax2.bar(actions, q_values)
        barlist[best_action].set_color("r")  # highlight highest Q-value

    frames, observations = _generate_episode()
    assert len(frames) == len(observations)

    save_count = kwargs.get(
        "save_count", len(frames)
    )  # number of frames to use to generate video
    interval = kwargs.get("interval", 300)  # time interval between frames in video

    anim = animation.FuncAnimation(
        fig,
        _plot_fn,
        zip(frames, observations),
        interval=interval,
        save_count=save_count,
    )

    return anim
