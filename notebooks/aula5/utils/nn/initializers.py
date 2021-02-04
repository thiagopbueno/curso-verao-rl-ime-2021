import sonnet as snt


def he_initializer():
    return snt.initializers.VarianceScaling(scale=2.0, mode="fan_in", distribution="uniform")