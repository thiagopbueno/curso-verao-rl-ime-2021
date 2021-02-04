import os

import tensorflow as tf


def set_tf_allow_growth():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)


def create_variables(network, *input_spec):
    inputs = [tf.expand_dims(tf.zeros(spec.shape, spec.dtype), axis=0) for spec in input_spec]
    network(*inputs)
    assert len(network.trainable_variables) > 0

    
def suppress_tfp_warning():
    """Suppress tensorflow_probability warning"""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
