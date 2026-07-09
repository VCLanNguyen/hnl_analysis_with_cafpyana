#!/usr/bin/env python
"""Build detector variation HDF5 stores with preprocessing and optional selection.

Loads all ``detvar_<name>_<idx>.df`` files in the input directory.  Files
whose base name is ``cv`` are treated as CV samples; all others are DV
samples.  Files sharing the same base name are grouped into a list (multisim
pairs when two universes exist).  The CV mapping is derived automatically:
every DV maps to the CV key specified by ``--cv-key`` (default: the first CV
found in the directory).

Recombination detector variations are built from the reference CV and added
to the DV set automatically.

Selection types
---------------
preprocess  Preprocessing only (no cut-based selection); output: detvars.h5
signal      Full signal selection via select(); output: detvars_signal.h5
sideband    Sideband selection via select_sideband(); output: detvars_sideband.h5
preselect   Signal selection stopped at shower_energy stage; output: detvars_preselect.h5
all         All four of the above.

Examples
--------
    # signal + sideband stores only (default)
    python process_detvars.py -i /path/to/dfs/ -o /path/to/output/

    # write everything
    python process_detvars.py -i /path/to/dfs/ -o /path/to/output/ -s all

    # use a specific CV as the reference
    python process_detvars.py -i /path/to/dfs/ -o /path/to/output/ --cv-key cv_1
"""
from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.append("/exp/sbnd/data/users/lnguyen/cafpyana_pi0")
import hnl_analysis_with_cafpyana as nue

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SELECTION_FN = {
    "preprocess": (None,                {}),
    "signal":     (nue.select,          {}),
    "sideband":   (nue.select_sideband, {}),
    "preselect":  (nue.select,          {"stage": "shower_energy"}),
}

_OUTPUT_FILE = {
    "preprocess": "detvars.h5",
    "signal":     "detvars_signal.h5",
    "sideband":   "detvars_sideband.h5",
    "preselect":  "detvars_preselect.h5",
}

_ALL_SELECTIONS = ["preprocess", "signal", "sideband", "preselect"]

_DETVAR_RE = re.compile(r'^detvar_(.+)_(\d+)\.df$')

# Sentinel distinguishing "caller didn't pass preprocess_fn" from "caller explicitly
# passed None" (which means skip preprocessing entirely) -- same pattern as io.py's
# load_mc/load_data.
_UNSET = object()


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _parse_detvar_files(input_dir: str) -> tuple[dict, dict]:
    """Scan input_dir and split detvar_<name>_<idx>.df files into CV and DV groups.

    Each DV file is stored as its own entry keyed by ``<name>_<idx>`` so the
    index can be used to look up the matching CV.  Files whose base name is
    ``cv`` are treated as CV samples.

    Returns
    -------
    cv_files : dict[str, str]
        {cv_key: filepath}  e.g. {'cv_0': '/path/detvar_cv_0.df'}
    dv_files : dict[str, tuple[str, int]]
        {dv_key: (filepath, idx)}  e.g. {'wiremodxw_1': ('/path/...', 1)}
    """
    cv_files: dict[str, str]          = {}
    dv_files: dict[str, tuple]        = {}

    for fname in sorted(os.listdir(input_dir)):
        m = _DETVAR_RE.match(fname)
        if not m:
            continue
        base, idx = m.group(1), int(m.group(2))
        fpath = os.path.join(input_dir, fname)
        if base == "cv":
            cv_files[f"cv_{idx}"] = fpath
        else:
            dv_files[f"{base}_{idx}"] = (fpath, idx)

    return cv_files, dv_files


# ---------------------------------------------------------------------------
# Dict builders
# ---------------------------------------------------------------------------

def build_dicts(input_dir: str, cv_key: str | None = None, slc_key: str = "nuecc",
                 preprocess_fn=_UNSET):
    """Load all CV and DV files; attach recombination detvars.

    Parameters
    ----------
    input_dir : str
        Directory containing detvar_<name>_<idx>.df files.
    cv_key : str, optional
        Key of the CV used as the reference for DV matching and recombination
        variations.  Defaults to the lexicographically first CV key found.
    slc_key : str, optional
        Table key for the slice-level analysis DataFrame within each raw
        ``.df`` file, forwarded to :func:`~nueana.detvar.prepare_detvar_df`.
        Defaults to ``"nuecc"``; pass ``"rec"`` for HNL-topology samples
        produced via ``hnl_mcnu_detvar.py``.
    preprocess_fn : callable or None, optional
        Called as ``preprocess_fn(slc_df)`` on each CV/DV's slice-level table
        before it's stored. Defaults to :func:`~nueana.preprocess.preprocess_mc`
        when ``slc_key == "nuecc"`` and to ``None`` (skip preprocessing)
        otherwise -- ``preprocess_mc`` applies nueCC-specific fixes (e.g.
        ``add_phi``, which needs a ``primtrk.trk.dir.x/y`` column that doesn't
        exist in the HNL/pi0 ``'rec'`` table). Pass an explicit callable (e.g.
        a future HNL/pi0-specific preprocessing function) to override either
        default, or ``None`` to force-skip preprocessing regardless of
        ``slc_key``.

    Returns
    -------
    cv_dict, dv_dict, cv_map
    """
    cv_files, dv_files = _parse_detvar_files(input_dir)

    if not cv_files:
        sys.exit("ERROR: no CV files found (expected detvar_cv_<idx>.df)")
    if not dv_files:
        print("WARNING: no DV files found.")

    # resolve reference CV key
    available_cv_keys = sorted(cv_files.keys())
    if cv_key is None:
        cv_key = available_cv_keys[0]
    elif cv_key not in cv_files:
        sys.exit(
            f"ERROR: CV key '{cv_key}' not found. "
            f"Available: {available_cv_keys}"
        )

    if preprocess_fn is _UNSET:
        preprocess_fn = nue.preprocess_mc if slc_key == "nuecc" else None

    def _load_preprocessed(path):
        dvf = nue.prepare_detvar_df(path, slc_key=slc_key)
        if preprocess_fn is not None:
            dvf = dvf._replace(slc_df=preprocess_fn(dvf.slc_df))
        return dvf

    # load CVs
    cv_dict: dict = {}
    for key, path in sorted(cv_files.items()):
        print(f"Loading CV '{key}': {path}")
        cv_dict[key] = _load_preprocessed(path)

    # load DVs; each file maps to the CV with the matching index
    dv_dict: dict = {}
    cv_map:  dict = {}
    for key, (path, idx) in sorted(dv_files.items()):
        mapped_cv = f"cv_{idx}"
        if mapped_cv not in cv_dict:
            sys.exit(
                f"ERROR: DV '{key}' expects CV '{mapped_cv}' but it was not found. "
                f"Available CVs: {sorted(cv_dict.keys())}"
            )
        print(f"  Loading DV '{key}' → CV '{mapped_cv}'")
        dv_dict[key] = _load_preprocessed(path)
        cv_map[key]  = mapped_cv

    # recombination detvars from reference CV (already preprocessed)
    ref_cv = cv_dict[cv_key]
    print(f"\nBuilding recombination detvars from '{cv_key}'...")
    recomb_dfs = nue.make_recomb_detvars(ref_cv.slc_df)
    recomb = {
        name: [ref_cv._replace(slc_df=df) for df in dfs]
        for name, dfs in recomb_dfs.items()
    }
    dv_dict.update(recomb)
    cv_map.update({name: cv_key for name in recomb})
    print(f"  Recomb variations: {list(recomb.keys())}")

    return cv_dict, dv_dict, cv_map


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input-dir",
        required=True,
        dest="input_dir",
        help="Directory containing detvar_<name>_<idx>.df files.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        required=True,
        dest="output_dir",
        help="Directory to write output .h5 stores.",
    )
    parser.add_argument(
        "--selections", "-s",
        nargs="+",
        choices=_ALL_SELECTIONS + ["all"],
        default=["signal", "sideband"],
        metavar="SELECTION",
        help=(
            "Which stores to write. Choices: "
            + ", ".join(_ALL_SELECTIONS)
            + ", all (shorthand for all four). Default: signal sideband."
        ),
    )
    parser.add_argument(
        "--cv-key",
        default=None,
        help=(
            "CV key to use as the reference for DV matching and recombination "
            "variations (e.g. 'cv_0'). Defaults to the first CV key found."
        ),
    )
    parser.add_argument(
        "--slc-key",
        dest="slc_key",
        default="nuecc",
        help=(
            "Table key for the slice-level analysis DataFrame within each raw "
            "detvar_*.df file. Default 'nuecc' (nueCC topology). Pass 'rec' for "
            "HNL-topology samples produced via hnl_mcnu_detvar.py. Note: the "
            "'signal'/'sideband' selection presets below apply nueCC's own "
            "DEFAULT_CUTS/SIDEBAND_CUTS and are not meaningful for HNL selection "
            "cuts -- for HNL, build with '-s preprocess' only and apply the real "
            "cut sequence later via get_total_cov(..., cuts=<hnl_cuts>)."
        ),
    )
    parser.add_argument(
        "--groups", "-g",
        nargs="+",
        default=None,
        metavar="GROUP",
        help=(
            "Subset of DV group names to (re)compute. When omitted, all groups are "
            "built. When specified, only the listed groups are written; if the output "
            "file already exists the other groups are left untouched (append mode). "
            "Example: -g recomb_lo recomb_hi wiremodxw_0"
        ),
    )
    args = parser.parse_args()

    selections = _ALL_SELECTIONS if "all" in args.selections else args.selections

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- discover and load ----
    cv_dict, dv_dict, cv_map = build_dicts(args.input_dir, cv_key=args.cv_key, slc_key=args.slc_key)
    print(f"\nCV keys : {list(cv_dict.keys())}")
    print(f"DV keys : {list(dv_dict.keys())}")
    print(f"CV map  : {cv_map}")

    # ---- filter to requested groups (--groups) ----
    write_mode = 'w'
    if args.groups is not None:
        # Expand any CV key names (e.g. 'cv_0') to all DV groups mapped to that CV
        expanded: set[str] = set()
        for name in args.groups:
            if name in cv_dict:
                matched = {g for g, ck in cv_map.items() if ck == name}
                print(f"  '{name}' → {sorted(matched)}")
                expanded |= matched
            else:
                expanded.add(name)
        unknown = expanded - set(dv_dict)
        if unknown:
            print(f"\nWARNING: requested groups not found: {sorted(unknown)}")
        keep = expanded & set(dv_dict)
        dv_dict = {k: v for k, v in dv_dict.items() if k in keep}
        cv_map  = {k: v for k, v in cv_map.items()  if k in keep}
        needed_cvs = set(cv_map.values())
        cv_dict = {k: v for k, v in cv_dict.items() if k in needed_cvs}
        write_mode = 'a'
        print(f"\nFiltered to groups : {sorted(dv_dict.keys())}")
        print(f"Write mode         : append (patch existing store)")

    # ---- write each requested store ----
    for sel_name in selections:
        out_path = os.path.join(args.output_dir, _OUTPUT_FILE[sel_name])
        print(f"\n[{sel_name}] → {out_path}")
        fn, kwargs = _SELECTION_FN[sel_name]
        if fn is None:
            nue.write_detvar_store(out_path, cv_dict, dv_dict, cv_map, mode=write_mode)
        else:
            cv_sel = nue.apply_selection(cv_dict, fn, **kwargs)
            dv_sel = nue.apply_selection(dv_dict, fn, **kwargs)
            nue.write_detvar_store(out_path, cv_sel, dv_sel, cv_map, mode=write_mode)

    print("\nDone.")


if __name__ == "__main__":
    main()
