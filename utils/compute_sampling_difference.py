# pylint: disable=logging-fstring-interpolation
"""Script to compute the differences in the time series for sampled data
vs. the original training data."""

# Standard imports
import logging
import numpy as np
import pandas as pd
# Use only when computing DTW or euclidean distance
from dtaidistance import dtw

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def extract_og_ts(path: str, filters: dict) -> pd.DataFrame:
    """Read the original time series data."""
    logger.debug(f"Reading the original time series data from {path}")
    og_ts = pd.read_csv(path)
    if filters:
        # Filter the data according to the filters
        for key, value in filters.items():
            og_ts = og_ts[og_ts[key] == value]
    return og_ts


def extract_sub_ts(path: str) -> pd.DataFrame:
    """Read the subsampled time series data."""
    logger.debug(f"Reading the subsampled time series data from {path}")
    sub_ts = pd.read_csv(path)
    return sub_ts

def merge_ts(og_ts: pd.DataFrame,
             sub_ts: pd.DataFrame,
             delta: float
             ):
    """Merge the original and subsampled time series data."""
    logger.debug(
        "Merging the original and subsampled time series data for each run.")

    missed_events = {}
    euclid_dists = {}
    dtw_dists = {}
    # For each run, merge the data
    for run in og_ts['Run'].unique():
        logger.debug(f"Processing Run {run}")
        missed_events[run] = 0
        og_run = og_ts[og_ts['Run'] == run][[
            'Run', 'Time', 'QueueAL', 'QueueBL', 'QueueCL'
        ]]
        sub_run = pd.DataFrame(columns=['Run', 'Time', 'Lqa', 'Lqb', 'Lqc'])
        # Extract the index of the last column
        last_col = int(sub_ts.columns[-1][3:])
        # Extract the corresponding row in sub_ts
        sub_ts_vals = sub_ts.iloc[run - 1]
        # Extract only the required data
        time_stamp = 0.0
        for i in range(last_col + 1):
            lp_val = sub_ts_vals[f'Lqa{i}']
            lfc_val = sub_ts_vals[f'Lqb{i}']
            lsc_val = sub_ts_vals[f'Lqc{i}']
            sub_run.loc[len(sub_run)] = [
                run, time_stamp, lp_val, lfc_val, lsc_val
            ]
            time_stamp += delta
        # Merge the data
        merged_ts = pd.merge(og_run, sub_run, on='Time', how='outer')
        merged_ts['Time'] = pd.to_datetime(merged_ts['Time'], unit='s')
        merged_ts = merged_ts.set_index('Time')
        merged_ts = merged_ts.sort_index()
        # Interpolate the missing values
        merged_ts['QueueAL'] = merged_ts['QueueAL'].ffill()
        merged_ts['QueueBL'] = merged_ts['QueueBL'].ffill()
        merged_ts['QueueCL'] = merged_ts['QueueCL'].ffill()
        merged_ts['Lqa'] = merged_ts['Lqa'].ffill()
        merged_ts['Lqb'] = merged_ts['Lqb'].ffill()
        merged_ts['Lqc'] = merged_ts['Lqc'].ffill()

        # For every time step, see if the vector (QueueAL, QueueBL, QueueCL)
        # is different from the vector (Lqa, Lqb, Lqc)
        for _, row in merged_ts.iterrows():
            if (row['QueueAL'] != row['Lqa'] or
                row['QueueBL'] != row['Lqb'] or
                row['QueueCL'] != row['Lqc']):
                missed_events[run] += 1

        # Compute the euclidean distance between the two time series,
        # for all queue lengths A, B and C
        qaeuclid = np.linalg.norm(merged_ts['QueueAL'] - merged_ts['Lqa'])
        qbeuclid = np.linalg.norm(merged_ts['QueueBL'] - merged_ts['Lqb'])
        qceuclid = np.linalg.norm(merged_ts['QueueCL'] - merged_ts['Lqc'])
        euclid_dists[run] = qaeuclid + qbeuclid + qceuclid
        # Calculate the DTW distance
        qadtw = dtw.distance(merged_ts['QueueAL'], merged_ts['Lqa'])
        qbdtw = dtw.distance(merged_ts['QueueBL'], merged_ts['Lqb'])
        qcdtw = dtw.distance(merged_ts['QueueCL'], merged_ts['Lqc'])
        dtw_dists[run] = qadtw + qbdtw + qcdtw

    logger.info(f"Total missed events: {sum(missed_events.values())}")
    logger.info(f'Total Euclidean distance: {sum(euclid_dists.values())}')
    logger.info(f'Total DTW distance: {sum(dtw_dists.values())}')
       
if __name__ == '__main__':
    exp_num = 4
    d = 0.05
    ogts = extract_og_ts(
        'data/simulation/time-series-exp-1.csv',
        filters=None
    )
    subts = extract_sub_ts(
        f'data/discrete_time/discrete-time-dbn-exp-{exp_num}.csv'
    )
    merge_ts(ogts, subts, d)