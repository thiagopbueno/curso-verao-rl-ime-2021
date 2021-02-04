from collections import OrderedDict

import numpy as np
from gym.spaces import Discrete


class ReplayBuffer:
    def __init__(self, obs_space, action_space, max_size, batch_size: int = 1):
        self.size = 0
        self.max_size = max_size
        self.batch_size = batch_size

        action_precision = (
            np.int32 if isinstance(action_space, Discrete) else action_space.dtype
        )
        self.spec = OrderedDict(
            {
                "obs": (obs_space.shape, obs_space.dtype),
                "action": (action_space.shape, action_precision),
                "reward": ((), np.float32),
                "terminal": ((), np.bool),
            }
        )

    def build(self):
        for key, (shape, dtype) in self.spec.items():
            setattr(self, key, np.ones([self.max_size, *shape], dtype))

    def add(self, obs, action, reward, terminal, next_obs):
        del next_obs
        transition = (obs, action, reward, terminal)
        idx = self.size % self.max_size
        for key, value in zip(self.spec.keys(), transition):
            getattr(self, key)[idx] = value
        self.size += 1

    def sample(self):
        low, high = 0, min(self.size, self.max_size - 1)
        idxs = np.random.randint(
            low,
            high,
            size=self.batch_size,
        )
        return OrderedDict(
            {
                **{key: getattr(self, key)[idxs] for key in self.spec},
                "next_obs": self.obs[idxs + 1],
            }
        )
