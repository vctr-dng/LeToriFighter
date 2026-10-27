"""Mapping between Toribash joint states {0,1,2,3} and the continuous levels.

Interface (a) trains the flow-matching expert on levels {-1, -1/3, 1/3, 1} and
rounds each output component to the nearest level. The functions use only
arithmetic and `.round()` / `.clip()`, so they work on numpy arrays and torch
tensors alike.
"""

import numpy as np

NUM_STATES = 4
LEVELS = (-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0)


def states_to_levels(states):
    """{0,1,2,3} -> {-1,-1/3,1/3,1}."""
    if isinstance(states, list | tuple):
        states = np.asarray(states)
    return states * (2.0 / 3.0) - 1.0


def levels_to_states(levels):
    """Nearest of the four levels, as the state index in 0..3 (same dtype as the
    float input; cast afterwards)."""
    if isinstance(levels, list | tuple):
        levels = np.asarray(levels)
    return ((levels + 1.0) * 1.5).round().clip(0, NUM_STATES - 1)


def round_to_levels(levels):
    """Snap continuous values to the nearest of the four levels."""
    return states_to_levels(levels_to_states(levels))
