import os.path as osp

import tensorflow as tf


class TFLogger:

    def __init__(self, name, base_dir="logs"):
        self._logger_id = osp.join(base_dir, name)
        self._writer = tf.summary.create_file_writer(self._logger_id)

    def log(self, step, stats, label=None):
        with self._writer.as_default():
            for key, value in stats.items():
                if isinstance(value, dict):
                    self.log(step, value, label=label)
                else:
                    prefix = f"{label}/" if label else ""
                    if tf.rank(value) == 0:
                        tf.summary.scalar(f"{prefix}{key}", value, step=step)
                    else:
                        tf.summary.histogram(f"{prefix}{key}", value, step=step)

    def flush(self):
        self._writer.flush()