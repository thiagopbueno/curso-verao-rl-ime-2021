from IPython.display import HTML
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import numpy as np
import tensorflow as tf


rc('animation', html='html5')


def display_frames_and_q_values(agent, env, **kwargs):
    figsize = kwargs.get("figsize", (17, 5))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    actions = kwargs.get("actions", env.unwrapped.get_action_meanings())

    def _generate_episode():
        frames = []
        observations = []

        obs = env.reset()
        done = False

        while not done:
            frame = env.render(mode="rgb_array")
            frames.append(frame)
            observations.append(obs)

            action = agent.step(np.expand_dims(obs, axis=0), training=False)[0]
            obs, reward, done, _ = env.step(action)

        env.close()

        return frames, observations

    def _plot_fn(inputs):
        frame, obs = inputs

        ax1.clear()
        ax2.clear()

        ax1.axis("off")
        ax2.set_xlabel("Actions")

        ax1.imshow(frame)

        q_values = agent.q_net(np.expand_dims(obs, axis=0))[0]
        #q_values = tf.nn.softmax(q_values) # try to make differences between values more salient
        best_action = np.argmax(q_values)
        barlist = ax2.bar(actions, q_values)
        barlist[best_action].set_color('r') # highlight highest Q-value

    frames, observations = _generate_episode()
    assert len(frames) == len(observations)

    save_count = kwargs.get("save_count", len(frames)) # number of frames to use to generate video
    interval = kwargs.get("interval", 300) # time interval between frames in video

    anim = animation.FuncAnimation(fig, _plot_fn, zip(frames, observations), interval=interval, save_count=save_count)

    return anim