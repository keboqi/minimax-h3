"""Request field checks shared by the engines' validators."""

import math

from h3_app.errors import H3Error


def whole_number(value, low, high, name, multiple=1):
    """``value`` as an int within ``low``–``high`` on a ``multiple`` grid."""
    if isinstance(value, bool) or not math.isfinite(float(value)) or int(value) != value:
        raise H3Error(f"{name} must be a whole number.")
    value = int(value)
    if not low <= value <= high or value % multiple:
        raise H3Error(f"{name} must be {low}–{high}, in increments of {multiple}.")
    return value


def check_seed(seed):
    """-1 asks for a random seed."""
    whole_number(seed, -1, 2**63 - 2, "Seed")


def check_cfg(value, low, high, message):
    if isinstance(value, bool) or not math.isfinite(float(value)) or not low <= value <= high:
        raise H3Error(message)
