# This function is used to find the volume of each cell regions in label_grid and next_grid

import numpy as np
import pandas as pd

def compare_label_counts(grid_a, grid_b, num_labels):
    """
    Compare counts of labels 0..num_labels in two grids.

    Parameters
    ----------
    grid_a, grid_b : array-like of integers
        Label grids; can be any shape, but should contain integer labels.
    num_labels : int
        Maximum label (inclusive) to consider. Labels are assumed in [0, num_labels].

    Returns
    -------
    pandas.DataFrame
        Columns: label, count_a, count_b, difference (b - a), pct_change (relative to a).
        pct_change is NaN when count_a is zero and count_b is also zero; inf if count_a is zero but count_b > 0.
    """
    # Flatten and ensure integer arrays
    a = np.asarray(grid_a).ravel()
    b = np.asarray(grid_b).ravel()

    # Optional: warn if there are out-of-range labels
    max_seen = max(a.max() if a.size else 0, b.max() if b.size else 0)
    if max_seen > num_labels:
        raise ValueError(f"Found label {max_seen} > num_labels ({num_labels}).")

    # Count with minlength so labels with zero occurrences appear
    counts_a = np.bincount(a, minlength=num_labels+1)[: num_labels + 1]
    counts_b = np.bincount(b, minlength=num_labels+1)[: num_labels + 1]

    diff = counts_b - counts_a
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(counts_a == 0,
                       np.where(counts_b == 0, np.nan, np.inf),
                       diff / counts_a * 100.0)

    df = pd.DataFrame({
        "label": np.arange(num_labels + 1, dtype=int),
        "count_a": counts_a,
        "count_b": counts_b,
        "difference": diff,
        "pct_change(%)": pct,
    })
    return df
