"""Built-in trainer backends. Heavy imports happen inside each backend function,
so registering them costs nothing on machines without a GPU stack."""

from jevkit.training.backends import hf, unsloth  # noqa: F401
