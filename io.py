"""File input/output utilities for loading HDF5 data files."""
import pandas as pd
import numpy as np

# credit for first three functions to Mun! 
def get_n_split(file):
    """Get the number of splits in an HDF5 file.
    
    Parameters
    ----------
    file : str
        Path to HDF5 file.
    
    Returns
    -------
    int
        Number of splits in the file.
    """
    this_split_df = pd.read_hdf(file, key="split")
    this_n_split = this_split_df.n_split.iloc[0]
    return this_n_split

def print_keys(file):
    """Print all keys available in an HDF5 file.
    
    Parameters
    ----------
    file : str
        Path to HDF5 file.
    """
    with pd.HDFStore(file, mode='r') as store:
        keys = store.keys()       # list of all keys in the file
        print("Keys:", keys)
        
def load_dfs(file, keys2load, n_max_concat=10, start_split=0):
    """Load DataFrames from split HDF5 file.
    
    Parameters
    ----------
    file : str
        Path to HDF5 file.
    keys2load : list
        List of key names to load from the file.
    n_max_concat : int, optional
        Maximum number of splits to concatenate (default: 10).
    start_split : int, optional
        Starting split index to load from (default: 0).
    
    Returns
    -------
    dict
        Dictionary mapping key names to concatenated DataFrames.
    """
    out_df_dict = {}
    this_n_keys = get_n_split(file) - start_split
    n_concat = min(n_max_concat, this_n_keys)
    for key in keys2load:
        dfs = []  # collect all splits for this key
        for i in range(start_split, start_split + n_concat):
            this_df = pd.read_hdf(file, key=f"{key}_{i}")
            dfs.append(this_df)
        out_df_dict[key] = pd.concat(dfs, ignore_index=False)
    return out_df_dict

def correct_cosmic_weight_mevprtl_df(indf_rec, indf_truth, indf_hdr):

    #Step 1: Build total event weight in reco and truth HNL dataframes

    # Build total event weight in reco and truth HNL dataframes
    fluxw_column = ('slc', 'prtl', 'flux_weight', '', '', '')
    rayw_column = ('slc', 'prtl', 'ray_weight', '', '', '')
    decayw_column = ('slc', 'prtl', 'decay_weight', '', '', '')
    totalw_column = ('weights_mc', '', '', '', '', '')
    indf_rec[totalw_column] = indf_rec[rayw_column] * indf_rec[decayw_column] * indf_rec[fluxw_column]

    truth_fluxw_column = ('flux_weight', '')
    truth_rayw_column = ('ray_weight', '')
    truth_decayw_column = ('decay_weight', '')
    truth_totalw_column = ('weights_mc_truth', '')
    indf_truth[truth_totalw_column] = indf_truth[truth_rayw_column] * indf_truth[truth_decayw_column] * indf_truth[truth_fluxw_column]

    #-------------------------------------------------------------------------------------#
    #Step 2: Join evt from header dataframe into reco dataframe using shared (__ntuple, entry) index
    evt_col = ('evt', '', '', '', '', '')
    indf_rec[evt_col] = indf_hdr['evt'].reindex(indf_rec.index)

    #-------------------------------------------------------------------------------------#
    # Step 3: Join truth total weight into reco dataframe on (__ntuple, entry, evt)
    totalw_truth_col = ('weights_mc_truth', '', '', '', '', '')

    truth_total_weight = indf_truth[truth_totalw_column]
    truth_evt = indf_hdr['evt'].reindex(indf_truth.index).values

    truth_key = pd.MultiIndex.from_arrays(
        [
            indf_truth.index.get_level_values(0),
            indf_truth.index.get_level_values(1),
            truth_evt,
        ],
        names=['__ntuple', 'entry', 'evt'],
    )
    truth_lookup = pd.Series(truth_total_weight.values, index=truth_key)

    reco_key = pd.MultiIndex.from_arrays(
        [
            indf_rec.index.get_level_values(0),
            indf_rec.index.get_level_values(1),
            indf_rec[evt_col].values,
        ],
        names=['__ntuple', 'entry', 'evt'],
    )

    indf_rec[totalw_truth_col] = truth_lookup.reindex(reco_key).to_numpy()

    #-------------------------------------------------------------------------------------#
    # Step 4: Correct cosmic weight in reco dataframe using truth total weight for cosmic entries

    mask_cosmic = np.isnan(indf_rec[('slc', 'prtl', 'E', '', '', '')])
    indf_rec.loc[mask_cosmic, totalw_column] = indf_rec.loc[mask_cosmic, truth_totalw_column]

    #-------------------------------------------------------------------------------------#
    #Drop event column from reco dataframe
    indf_rec = indf_rec.drop(columns=[evt_col])

    return indf_rec