"""The only module in ta_tools that imports a third-party TA library."""
import pandas as pd

try:
    import talib
except ImportError as exc:
    raise ImportError(
        "ta_tools needs TA-Lib; from the repo root run: pip install -e '.[ta]'"
    ) from exc

SOURCES = ('talib', 'pandas_ta', 'derived', 'custom')

CAPABILITIES = {}


def provides(source):
    """Record the decorated primitive's source in CAPABILITIES."""
    if source not in SOURCES:
        raise ValueError(f'source must be one of {SOURCES}, got {source!r}')

    def register(fn):
        CAPABILITIES[fn.__name__] = source
        return fn

    return register


def as_float_series(values):
    """Reject non-Series input and cast to float64, the only dtype TA-Lib accepts."""
    if not isinstance(values, pd.Series):
        raise TypeError(f'expected a pandas Series, got {type(values).__name__}')
    return values.astype('float64')


def pandas_ta():
    """The pandas_ta module, imported on first use so importing ta_tools never loads the beta."""
    try:
        import pandas_ta as module
    except ImportError as exc:
        raise ImportError(
            "This indicator needs pandas_ta; from the repo root run: pip install -e '.[ta]'"
        ) from exc
    return module
