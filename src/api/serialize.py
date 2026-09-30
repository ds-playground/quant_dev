"""Turn pandas and Plotly objects into JSON-ready Python values.

Floats stay Python floats, so they survive JSON exactly. JSON has no NaN or infinity, so any
non-finite number becomes null. Dates become ISO strings.
"""
import datetime as dt
import json
import math

import numpy as np
import pandas as pd


def clean(value):
    """One value, JSON-ready."""
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    return value


def series_to_dict(series):
    return {str(k): clean(v) for k, v in series.items()}


def frame_to_table(frame):
    """A table as its column names in order plus one record per row. JavaScript objects put
    integer-like keys first, so the column order is sent explicitly."""
    return {'columns': [str(c) for c in frame.columns], 'records': frame_to_records(frame)}


def frame_to_records(frame):
    """One dict per row, column names as keys; the index is dropped (reset it first to keep it)."""
    return [{str(k): clean(v) for k, v in row.items()} for row in frame.to_dict(orient='records')]


def figure_to_json(fig):
    """A Plotly figure as the JSON object react-plotly.js takes (`data` and `layout`)."""
    return json.loads(fig.to_json())
