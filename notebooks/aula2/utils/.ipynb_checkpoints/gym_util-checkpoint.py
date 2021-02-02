import warnings
from contextlib import contextmanager

import numpy as np
from gym.spaces import Box, Discrete, Space
from gym.wrappers import TimeAwareObservation


def assert_continuous(space: Space, client, nome: str = "observação"):
    assert isinstance(space, Box), f"{type(client)} requer espaço de {nome} contínuo"
    

def assert_discrete(space: Space, client, nome: str = "ação"):
    assert isinstance(space, Discrete), f"{type(client)} requer espaço de {nome} discreto"
    

@contextmanager
def suppress_box_precision_warning():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*Box bound precision.*", module="gym.logger"
        )
        yield


class TimeAwareWrapper(TimeAwareObservation):
    """Add relative timestep specific to CartPole."""

    @suppress_box_precision_warning()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        high = np.array(
            [
                self.x_threshold * 2,
                np.inf,
                self.theta_threshold_radians * 2,
                np.inf,
                1.0,
            ],
            dtype=np.float32,
        )
        low = -high.copy()
        low[-1] = 0

        self.observation_space = Box(low, high, dtype=np.float32)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return np.append(observation, self.t / self.spec.max_episode_steps)
