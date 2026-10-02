"""Walk-forward folds, never random splits, purged where labels would overlap.

A label at row t covers (t, t + horizon]. A training row whose label reaches into the test window
would let the test outcome into training, so the last `horizon - 1` rows before each test window
are purged. For the next-day labels (horizon 1) that is none: the last training label is the
first test row's own return, known at its close, when the forecast is made.
"""
import numpy as np


def walk_forward(n, first_train=756, step=63, horizon=1, window=None):
    """[(train, test), ...] row positions for `n` rows.

    The first fold trains on `first_train` rows and tests the next `step`; each later fold moves
    on by `step` (the last may be shorter). Training is expanding by default, or the last
    `window` rows before the purge.
    """
    if first_train < 1 or step < 1 or horizon < 1:
        raise ValueError('first_train, step and horizon must be at least 1')
    if window is not None and window < 1:
        raise ValueError('window must be at least 1, or None for an expanding window')
    if first_train - (horizon - 1) < 1:
        raise ValueError(f'first_train ({first_train}) leaves no training rows after purging '
                         f'{horizon - 1} for horizon {horizon}')
    folds, a = [], first_train
    while a < n:
        b = min(a + step, n)
        end = a - (horizon - 1)
        start = 0 if window is None else max(0, end - window)
        folds.append((np.arange(start, end), np.arange(a, b)))
        a = b
    return folds


def purged_tail(train, horizon=1, tail=0.2):
    """(fit, validation) inside a training window: the last `tail` share held out for early
    stopping, with the `horizon - 1` fit rows before it purged as at a fold boundary."""
    if not 0 < tail < 1:
        raise ValueError(f'tail must be between 0 and 1, got {tail}')
    train = np.asarray(train)
    cut = int(len(train) * (1 - tail))
    fit = train[:max(cut - (horizon - 1), 0)]
    validation = train[cut:]
    if len(fit) == 0 or len(validation) == 0:
        raise ValueError(f'{len(train)} training rows are too few to hold out {tail:.0%}')
    return fit, validation
