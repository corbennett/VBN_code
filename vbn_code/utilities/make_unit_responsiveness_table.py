"""
Build the master unit responsiveness table.

For each session in the spike tensor, computes per-unit stimulus responsiveness
metrics (p-value, significance, direction, and magnitude) for every image
(nonchange flashes) and for change flashes. FDR correction is applied across
all stimulus conditions per unit. Results are concatenated across sessions and
merged with the base unit table on unit_id.

Inputs
------
unit_table_file   : master_unit_table.csv  (one row per unit, unit metadata)
stim_table_file   : master_stim_table_no_filter.csv
active_tensor_file: vbnAllUnitSpikeTensor.hdf5

Output
------
unit_responsiveness_table.csv  (unit_id + responsiveness columns)
"""

import h5py
import numpy as np
import pandas as pd
import scipy.stats
from statsmodels.stats.multitest import fdrcorrection

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TABLE_DIR = (
    "/Volumes/programs/mindscope/workgroups/np-behavior/"
    "vbn_data_release/supplemental_tables"
)

unit_table_file    = f"{TABLE_DIR}/master_unit_table.csv"
stim_table_file    = f"{TABLE_DIR}/master_stim_table_no_filter.csv"
active_tensor_file = f"{TABLE_DIR}/vbnAllUnitSpikeTensor.hdf5"

output_file = f"{TABLE_DIR}/unit_responsiveness_table.csv"

# ---------------------------------------------------------------------------
# Response window parameters (in tensor time bins, 1 ms / bin)
# ---------------------------------------------------------------------------

BASE_WIN = slice(670, 750)   # pre-stimulus baseline
RESP_WIN = slice(20, 100)    # response window (decision window)

# ---------------------------------------------------------------------------
# Helper: compute responsiveness metrics for one stimulus condition
# ---------------------------------------------------------------------------

def _compute_responsiveness(pre_tensor, stim_tensor,
                             base_win=BASE_WIN, resp_win=RESP_WIN):
    """
    Given spike tensors for the pre-stimulus and stimulus periods, compute
    per-unit Wilcoxon p-values and summary response metrics.

    Parameters
    ----------
    pre_tensor  : ndarray (units, trials, time)
    stim_tensor : ndarray (units, trials, time)

    Returns
    -------
    pval, positive_modulation, mean_evoked, peak_evoked  -- each shape (units,)
    """
    base = pre_tensor[:, :, base_win].mean(axis=2)   # (units, trials)
    resp = stim_tensor[:, :, resp_win].mean(axis=2)  # (units, trials)

    mean_evoked = resp.mean(axis=1) - base.mean(axis=1)
    peak_evoked = np.max(
        resp - base.mean(axis=1, keepdims=True), axis=1
    )
    positive_modulation = mean_evoked > 0

    pval = np.array(
        [
            1.0
            if np.sum(r - b) == 0
            else scipy.stats.wilcoxon(b, r)[1]
            for b, r in zip(base, resp)
        ]
    )

    return pval, positive_modulation, mean_evoked, peak_evoked


# ---------------------------------------------------------------------------
# Helper: filter stim table indices
# ---------------------------------------------------------------------------

def _nonchange_flash_indices(session_stim, image_id=None):
    mask = (
        (session_stim["lick_for_flash"] == False)
        & (session_stim["flashes_since_last_lick"] >= 2)
        & (session_stim["is_change"] == False)
        & (session_stim["reward_rate"] >= 2)
        & (session_stim["previous_omitted"] == False)
        & (session_stim["flashes_since_change"] > 5)
    )
    if image_id is not None:
        mask &= session_stim["image_name"] == image_id
    else:
        mask &= ~session_stim["omitted"]
    return session_stim[mask].index.values


def _change_flash_indices(session_stim):
    mask = (session_stim["is_change"] == True) & (session_stim["reward_rate"] >= 2)
    return session_stim[mask].index.values


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    units = pd.read_csv(unit_table_file)
    stim_table = pd.read_csv(stim_table_file).drop(columns="Unnamed: 0", errors="ignore")

    # Image sets per image-set label
    g_images = (
        ["omitted"]
        + sorted(
            stim_table.loc[
                stim_table["stimulus_name"].str.contains("_G_")
                & ~stim_table["omitted"]
                & ~stim_table["image_name"].isin(["im083_r", "im111_r"]),
                "image_name",
            ].unique()
        )
        + ["im083_r", "im111_r"]
    )
    h_images = (
        ["omitted"]
        + sorted(
            stim_table.loc[
                stim_table["stimulus_name"].str.contains("_H_")
                & ~stim_table["omitted"]
                & ~stim_table["image_name"].isin(["im083_r", "im111_r"]),
                "image_name",
            ].unique()
        )
        + ["im083_r", "im111_r"]
    )

    session_results = []

    with h5py.File(active_tensor_file, "r") as tensor:
        session_ids = list(tensor.keys())
        print(f"Processing {len(session_ids)} sessions...")

        for session_ind, session_id in enumerate(session_ids):
            if session_ind % 10 == 0:
                print(f"  session {session_ind}/{len(session_ids)}")

            session_stim = stim_table[
                stim_table["session_id"] == int(session_id)
            ].reset_index()

            if len(session_stim) == 0:
                continue

            session_tensor = tensor[session_id]
            tensor_unit_ids = session_tensor["unitIds"][()]

            # Align unit table to tensor ordering
            session_units = (
                units.set_index("unit_id").loc[tensor_unit_ids].reset_index()
            )

            # Load spikes unit-by-unit into memory once per session.
            # The HDF5 tensor is chunked along the unit dimension, so reading
            # unit-by-unit is much faster than a single slice of the full array.
            n_units = len(tensor_unit_ids)
            spikes_h5 = session_tensor["spikes"]
            all_spikes = np.zeros(
                (n_units, spikes_h5.shape[1], spikes_h5.shape[2]), dtype=bool
            )
            for i in range(n_units):
                all_spikes[i] = spikes_h5[i]

            image_set = (
                g_images
                if "_G_" in session_stim["stimulus_name"].iloc[0]
                else h_images
            )

            # --- nonchange flashes, per image ---
            for image in image_set:
                trial_inds = _nonchange_flash_indices(session_stim, image_id=image)
                if len(trial_inds) == 0:
                    continue

                stim_sp = all_spikes[:, trial_inds, :]
                pre_sp  = all_spikes[:, trial_inds - 1, :]

                pval, pos_mod, mean_ev, peak_ev = _compute_responsiveness(
                    pre_sp, stim_sp
                )

                session_units[f"{image}_nonchange_response_pval"] = pval
                session_units[f"{image}_nonchange_positive_modulation"] = pos_mod
                session_units[f"{image}_nonchange_mean_evoked"] = mean_ev
                session_units[f"{image}_nonchange_peak_evoked"] = peak_ev

            # --- change flashes ---
            change_inds = _change_flash_indices(session_stim)
            if len(change_inds) > 0:
                stim_sp = all_spikes[:, change_inds, :]
                pre_sp  = all_spikes[:, change_inds - 1, :]

                pval, pos_mod, mean_ev, peak_ev = _compute_responsiveness(
                    pre_sp, stim_sp
                )

                session_units["change_response_pval"] = pval
                session_units["change_positive_modulation"] = pos_mod
                session_units["change_mean_evoked"] = mean_ev
                session_units["change_peak_evoked"] = peak_ev

            # --- FDR correction across all p-value columns, per unit ---
            pval_cols = [c for c in session_units.columns if "pval" in c]
            corrected_cols = [f"{c}_corrected" for c in pval_cols]
            sig_cols = [f"{c}_sig" for c in pval_cols]

            session_units[corrected_cols] = 1.0
            session_units[sig_cols] = False

            for idx, row in session_units.iterrows():
                row_pvals = row[pval_cols].values.astype(float)
                sig, corrected = fdrcorrection(row_pvals)
                session_units.loc[idx, corrected_cols] = corrected
                session_units.loc[idx, sig_cols] = sig

            session_results.append(session_units)

    # --- Concatenate all sessions ---
    print("Concatenating session results...")
    all_results = pd.concat(session_results, ignore_index=True)

    # Keep only unit_id and responsiveness columns
    resp_cols = [
        c for c in all_results.columns
        if any(k in c for k in ["_nonchange_", "change_response", "change_positive",
                                  "change_mean", "change_peak"])
    ]
    output = all_results[["unit_id"] + resp_cols]

    print(f"Saving {len(output)} rows to {output_file}")
    output.to_csv(output_file, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
