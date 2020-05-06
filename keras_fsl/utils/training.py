import os
from functools import reduce, wraps
from unittest.mock import patch


def patch_len(fit_generator):
    """
    Patch __len__ method generator to returns steps_per_epoch args of keras.fit_generator instead of actual len. This is
    to prevent queues to be initialized with way to many items.
    """

    @wraps(fit_generator)
    def fit_generator_patch_len(*args, **kwargs):
        generator = args[1]
        steps_per_epoch = kwargs.get("steps_per_epoch", len(generator))
        patch_train_sequence_len = patch.object(generator.__class__, "__len__", return_value=steps_per_epoch)

        validation_data = kwargs.get("validation_data", [])
        validation_steps = kwargs.get("validation_steps", len(validation_data))
        patch_val_sequence_len = patch.object(validation_data.__class__, "__len__", return_value=validation_steps)

        patch_train_sequence_len.start()
        if validation_steps:
            patch_val_sequence_len.start()

        history = fit_generator(*args, **kwargs)

        patch_train_sequence_len.stop()
        if validation_steps:
            patch_val_sequence_len.stop()

        return history

    return fit_generator_patch_len


def default_workers(fit_generator):
    """
    Patch default number of workers of keras.fit_generator from 1 to os.cpu_count()
    """

    @wraps(fit_generator)
    def fit_generator_with_default_cpu_count_worker(*args, **kwargs):
        if not kwargs.get("workers"):
            kwargs["workers"] = os.cpu_count()

        return fit_generator(*args, **kwargs)

    return fit_generator_with_default_cpu_count_worker


def compose(*functions):
    """
    Returns a function that apply each function of functions from left to right recursively
    """
    return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), functions)
