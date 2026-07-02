"""
Utilities for combining an exclusive sample with a nominal sample without double-counting.
"""

import pandas as pd

__all__ = ['remove_signal_overlap', 'remove_background_overlap']


def remove_signal_overlap(reco_df: pd.DataFrame, truth_df: pd.DataFrame) -> pd.DataFrame:
    """Return rows of reco_df whose events have no matching signal interaction in truth_df.

    Use this when combining a nominal sample with an exclusive (signal-only) sample
    to avoid double-counting events that contain the exclusive signal interaction.

    Parameters
    ----------
    reco_df : pd.DataFrame
        Processed, selected nuecc DataFrame with a top-level ``signal`` column
        (output of ``define_signal`` applied after ``preprocess_mc`` + ``select``).
    truth_df : pd.DataFrame
        Exclusive mcnuecc DataFrame with a top-level ``signal`` column
        (output of ``define_signal``); rows with ``signal == 0`` define the
        exclusive signal events.

    Returns
    -------
    pd.DataFrame
        Subset of ``reco_df`` with events matching a signal interaction in
        ``truth_df`` removed.
    """
    truth_sig_df = truth_df[truth_df.signal == 0]
    drop_idx = truth_sig_df.index.droplevel('rec.mc.nu..index')
    reco_idx = reco_df.index.droplevel('rec.slc..index')
    common_keys = drop_idx.intersection(reco_idx)
    return reco_df.loc[~reco_idx.isin(common_keys)]


def remove_background_overlap(reco_df: pd.DataFrame, truth_df: pd.DataFrame) -> pd.DataFrame:
    """Return rows of reco_df that have a matching event index in truth_df.

    Use this to restrict a reco sample to events that are matched to a truth
    entry, removing unmatched (background) rows.

    Parameters
    ----------
    reco_df : pd.DataFrame
        Processed, selected nuecc DataFrame (``rec.slc..index`` level present).
    truth_df : pd.DataFrame
        mcnuecc DataFrame (``rec.mc.nu..index`` level present) whose events
        define the matched set.

    Returns
    -------
    pd.DataFrame
        Subset of ``reco_df`` containing only events with a matching index in
        ``truth_df``.
    """
    truth_key = truth_df.index.droplevel('rec.mc.nu..index')
    reco_key  = reco_df.index.droplevel('rec.slc..index')
    return reco_df.loc[reco_key.isin(truth_key)]
