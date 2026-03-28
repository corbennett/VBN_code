"""
Build the unit RF stats table.

For each session in the VBN cache, computes receptive field metrics for every
unit using ReceptiveFieldMapping_VBN, then concatenates results into a single
table keyed by unit_id.

Inputs
------
cache_dir: local path to the VBN S3 cache

Output
------
unit_rf_table.csv  (unit_id + RF metric columns)
"""

import os
import pandas as pd
from allensdk.brain_observatory.behavior.behavior_project_cache.\
    behavior_neuropixels_project_cache \
    import VisualBehaviorNeuropixelsProjectCache
from brain_observatory_utilities.datasets.electrophysiology.\
    receptive_field_mapping import ReceptiveFieldMapping_VBN

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

cache_dir = (
    "/Volumes/programs/mindscope/workgroups/np-behavior/"
    "vbn_data_release/vbn_s3_cache"
)

output_file = (
    "/Volumes/programs/mindscope/workgroups/np-behavior/"
    "vbn_data_release/supplemental_tables/unit_rf_table.csv"
)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(
        cache_dir=cache_dir
    )

    session_table = cache.get_ecephys_session_table()
    session_ids = session_table.index.values
    print(f"Processing {len(session_ids)} sessions...")

    session_dfs = []
    for session_ind, session_id in enumerate(session_ids):
        if session_ind % 10 == 0:
            print(f"  session {session_ind}/{len(session_ids)}")

        try:
            session = cache.get_ecephys_session(ecephys_session_id=session_id)
            rf = ReceptiveFieldMapping_VBN(session)
            rf_metrics = rf.metrics
            session_dfs.append(rf_metrics)
        except Exception as e:
            print(f"  session {session_id} failed: {e}")
            continue

    print("Concatenating...")
    output = pd.concat(session_dfs, ignore_index=True)

    n_before = len(output)
    output = output.drop_duplicates(subset="unit_id")
    if len(output) < n_before:
        print(f"  Dropped {n_before - len(output)} duplicate unit rows.")

    print(f"Saving {len(output)} rows to {output_file}")
    output.to_csv(output_file, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
