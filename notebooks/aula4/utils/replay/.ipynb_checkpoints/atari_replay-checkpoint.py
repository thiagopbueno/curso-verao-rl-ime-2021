from collections import OrderedDict

import numpy as np


class AtariReplayBuffer:

    def __init__(self, obs_space, action_space, max_size, batch_size):
        self.size = 0
        self.max_size = max_size

        self.batch_size = batch_size

        self.spec = OrderedDict({
            "obs": (obs_space.shape[1:], np.uint8),
            "action": (action_space.shape, np.int32),
            "reward": ((), np.float32),
            "terminal": ((), np.bool),
        })

        self._frame_history_len = obs_space.shape[0]

    def build(self):
        for key, (shape, dtype) in self.spec.items():
            setattr(self, key, np.ones([self.max_size, *shape], dtype))

    def add(self, obs, action, reward, terminal, next_obs):
        del next_obs

        idx = self.size % self.max_size

        transition = (obs[-1], action, reward, terminal)
        for key, value in zip(self.spec.keys(), transition):
            getattr(self, key)[idx] = value

        self.size += 1

    def sample(self):
        low, high = 0, min(self.size, self.max_size)
        idxs = np.random.randint(low, high, size=self.batch_size,)

        all_obs = np.stack([self._encode_observation(idx) for idx in idxs], axis=0)

        return OrderedDict({
            "obs": all_obs[:, :-1, ...],
            "next_obs": all_obs[:, 1:, ...],
            **{key: getattr(self, key)[idxs] for key in list(self.spec)[1:]}
        })

    def _encode_observation(self, idx):
        end_idx = idx + 1
        start_idx = max(0, idx - (self._frame_history_len - 1))

        for i in range(start_idx, end_idx - 1):
            if self.terminal[i]:
                start_idx = i + 1

        all_obs = self.obs[start_idx:end_idx + 1]

        missing_context = (self._frame_history_len + 1) - len(all_obs)
        if missing_context > 0:
            padding_frames = np.tile(all_obs[0], reps=[missing_context, 1, 1])
            all_obs = np.concatenate([padding_frames, all_obs], axis=0)

        return all_obs
