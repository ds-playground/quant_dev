"""Display formatting, the interactive table, and CSV export."""

import os

import numpy as np

from .params import _params
from .analysis import filter_low_probability, low_probability_view


def format_probability_table(table):
    """Display copy with `threshold` and `prob` rendered as percentages.

    Formatting is applied last, after any sorting: these columns become strings,
    and sorting strings would order '9.60%' above '10.00%'.
    """
    if table.empty:
        return table
    return table.assign(threshold=lambda d: d['threshold'].map('{:.2%}'.format),
                        prob=lambda d: d['prob'].map('{:.2%}'.format))

def interactive_low_probability(df_his, p=None, n_days=None, prob_max=None,
                                prob_min=None, change_type=None):
    """Live-filtered rare-event table: pick a holding period, event type and bounds.

    Renders ipywidgets controls that refilter `df_his` in place, so the table
    updates without re-running the cell. Requires ipywidgets (preinstalled on
    Colab; `pip install ipywidgets` locally) - without it, this falls back to a
    static table built from the current arguments and says so.

    Returns the widget container, or the static DataFrame when falling back.
    """
    p = _params(p)
    n_days = p.streak_days[0] if n_days is None else n_days
    if not isinstance(n_days, (int, np.integer)):
        n_days = list(n_days)[0]        # a dropdown shows one period at a time
    prob_max = p.prob_max if prob_max is None else prob_max
    prob_min = p.prob_min if prob_min is None else prob_min

    ALL = 'All'
    types = sorted(df_his['change_type'].dropna().unique().tolist())

    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        print('ipywidgets is not installed - showing a static table instead.')
        print('Install it for live controls:  pip install ipywidgets')
        table = low_probability_view(df_his, n_days, p, prob_max, prob_min, change_type)
        print(f'{len(table)} event(s) | holding period {n_days}d | '
              f'type {change_type or ALL} | {prob_min:.2%} < prob < {prob_max:.2%}')
        return format_probability_table(table)

    label_style = {'description_width': 'initial'}
    days_w = widgets.Dropdown(
        options=list(p.streak_days), value=n_days,
        description='Streak days:', style=label_style)
    type_w = widgets.Dropdown(
        options=[ALL] + types, value=change_type or ALL,
        description='Change type:', style=label_style)
    max_w = widgets.BoundedFloatText(
        value=prob_max, min=0.0, max=1.0, step=0.01,
        description='prob_max:', style=label_style)
    min_w = widgets.BoundedFloatText(
        value=prob_min, min=0.0, max=1.0, step=0.0001,
        description='prob_min:', style=label_style)
    out = widgets.Output()

    def render(*_):
        with out:
            out.clear_output(wait=True)
            if min_w.value >= max_w.value:
                print(f'prob_min ({min_w.value:.2%}) must be below '
                      f'prob_max ({max_w.value:.2%}).')
                return
            selected_type = None if type_w.value == ALL else type_w.value
            table = low_probability_view(df_his, days_w.value, p,
                                         max_w.value, min_w.value, selected_type)
            print(f'{len(table)} event(s) of {len(df_his)} | '
                  f'holding period {days_w.value}d | type {type_w.value} | '
                  f'{min_w.value:.2%} < prob < {max_w.value:.2%}')
            if table.empty:
                print('Nothing in range - widen prob_max, lower prob_min, '
                      'or switch change type.')
            else:
                display(format_probability_table(table))

    for w in (days_w, type_w, max_w, min_w):
        w.observe(render, names='value')
    render()

    controls = widgets.HBox([widgets.VBox([days_w, type_w]),
                             widgets.VBox([max_w, min_w])])
    box = widgets.VBox([controls, out])
    display(box)
    return box

def export_tables(tables, out_dir='.', verbose=True):
    """Write a {filename: DataFrame} mapping to CSV; returns the paths written."""
    paths = []
    for name, table in tables.items():
        path = os.path.join(out_dir, name)
        table.to_csv(path, index=False)
        paths.append(path)
    if verbose:
        print(f'Saved {len(paths)} file(s): ' + ', '.join(tables))
    return paths
