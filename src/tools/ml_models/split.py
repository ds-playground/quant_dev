"""Walk-forward folds, never random splits, purged where labels would overlap.

A label at row t covers (t, t + horizon]. A training row whose label reaches into the test window
would let the test outcome into training, so the last `horizon - 1` rows before each test window
are purged. For the next-day labels (horizon 1) that is none: the last training label is the
first test row's own return, known at its close, when the forecast is made.
"""
import numpy as np
import pandas as pd


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


def walk_forward_dates(index, first_train_end='2022-12-31', months=6, horizon=1, min_train=252,
                       min_test_months=3):
    """[(train, test), ...] row positions for a Date `index`, in calendar periods.

    The first fold trains on every row up to `first_train_end` (rolled forward to its month end)
    and tests the next `months` months; each later fold adds those months to the training rows
    and tests the next `months`. Training is expanding, and the last `horizon - 1` training rows
    before each test window are purged, as in `walk_forward`.

    The last test window runs to the end of the data. If that leaves it less than
    `min_test_months` months of data, it is not a fold of its own: its days join the previous
    fold's test window, which then runs to the end of the data. So with data to August 2026, the
    fold that would train to 30 June 2026 is dropped, and the one before trains to 31 December
    2025 and tests from 1 January 2026 to the end. A window counts as long enough when its data
    reaches within a week of the `min_test_months` mark, which allows for weekends, holidays and
    the last bar, which has no label yet. 0 turns the rule off.

    A fold with fewer than `min_train` training rows (a ticker whose history starts late) is left
    out, so that ticker's first test window comes later. Returns [] when no fold qualifies.
    """
    if months < 1 or horizon < 1 or min_train < 1:
        raise ValueError('months, horizon and min_train must be at least 1')
    if not 0 <= min_test_months <= months:
        raise ValueError(f'min_test_months must be between 0 and months ({months})')
    index = pd.DatetimeIndex(index)
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError('the index must be dates in order, without repeats')
    end = pd.Timestamp(first_train_end) + pd.offsets.MonthEnd(0)
    folds, ends = [], []
    while True:
        a = int(index.searchsorted(end, side='right'))          # the first row after `end`
        if a >= len(index):
            break
        stop = end + pd.offsets.MonthEnd(months)
        b = int(index.searchsorted(stop, side='right'))
        train_end = a - (horizon - 1)
        if train_end >= min_train:
            folds.append((np.arange(0, train_end), np.arange(a, b)))
            ends.append(end)
        end = stop
    if len(folds) >= 2 and min_test_months:
        enough = ends[-1] + pd.offsets.MonthEnd(min_test_months) - pd.Timedelta(days=7)
        if index[-1] < enough:                                   # the last window is too short
            train, test = folds[-2]
            folds[-2:] = [(train, np.arange(test[0], len(index)))]
    return folds


def fold_table(index, folds):
    """One row per fold: the training and test rows' count and first and last dates."""
    index = pd.DatetimeIndex(index)
    rows = [{'fold': k + 1, 'train_rows': len(train), 'train_from': index[train[0]].date(),
             'train_to': index[train[-1]].date(), 'test_rows': len(test),
             'test_from': index[test[0]].date(), 'test_to': index[test[-1]].date()}
            for k, (train, test) in enumerate(folds)]
    columns = ['fold', 'train_rows', 'train_from', 'train_to', 'test_rows', 'test_from', 'test_to']
    return pd.DataFrame(rows, columns=columns).set_index('fold')
