import pickle
import pandas as pd
import numpy as np
import os
import gc
from pathlib import Path
from tqdm.auto import tqdm
from collections import defaultdict

def get_scenario_stats(df, cid, cyc, mode, scenario, start_p, end_p, target_q, col_v, col_i, col_t, col_temp):
    """
    Slices the cycle dataframe based on mode and SOC scenario, then calculates statistics.
    """
    # 1. Mode Filtering
    if mode == "C":
        mode_df = df[df[col_i] > 0.01].copy()
    else:
        mode_df = df[df[col_i] < -0.01].copy()
        
    if mode_df.empty:
        return None

    # 2. Capacity Accumulation
    if 'capacity' in mode_df.columns:
        q_acc = (mode_df['capacity'] - mode_df['capacity'].iloc[0]).abs()
    else:
        dt = mode_df[col_t].diff().fillna(0)
        dq = (mode_df[col_i].abs() * dt) / 3600.0
        q_acc = dq.cumsum()
        
    # 3. Slicing
    q_ratio = q_acc / (target_q + 1e-9)
    mask = (q_ratio >= start_p) & (q_ratio <= end_p)
    sliced = mode_df[mask]
    
    if sliced.empty:
        return None
        
    # 4. Statistics Calculation
    stats = {
        'cell_id': cid,
        'cycle': cyc,
        'count': len(sliced),
        'time': sliced[col_t].mean()
    }
    
    for feat_name, col_name in [('V', col_v), ('I', col_i), ('T', col_temp)]:
        if col_name in sliced.columns:
            series = sliced[col_name]
            stats[f'{feat_name}_mean'] = series.mean()
            stats[f'{feat_name}_std'] = series.std()
            stats[f'{feat_name}_var'] = series.var()
            stats[f'{feat_name}_min'] = series.min()
            stats[f'{feat_name}_max'] = series.max()
        else:
            stats[f'{feat_name}_mean'] = np.nan
            stats[f'{feat_name}_std'] = np.nan
            stats[f'{feat_name}_var'] = np.nan
            stats[f'{feat_name}_min'] = np.nan
            stats[f'{feat_name}_max'] = np.nan
            
    return stats

def process_scenarios(dataset_type, all_cells, clean_cache_dir, output_root, get_batch_id_func, get_cell_labels_func):
    """
    Memory-efficiently processes all cells and cycles to generate 6 scenario CSVs.
    """
    scenarios = {
        'charge-high': ('C', 0.0, 0.3),
        'charge-mid': ('C', 0.3, 0.7),
        'charge-low': ('C', 0.7, 1.0),
        'discharge-high': ('D', 0.0, 0.3),
        'discharge-mid': ('D', 0.3, 0.7),
        'discharge-low': ('D', 0.7, 1.0)
    }
    
    # Initialize CSV files with headers
    csv_paths = {}
    cols = [
        'cell_id', 'cycle', 'count', 'time',
        'V_mean', 'V_std', 'V_var', 'V_min', 'V_max',
        'I_mean', 'I_std', 'I_var', 'I_min', 'I_max',
        'T_mean', 'T_std', 'T_var', 'T_min', 'T_max'
    ]
    
    for name in scenarios.keys():
        path = output_root / f"{dataset_type}_{name}_summary.csv"
        pd.DataFrame(columns=cols).to_csv(path, index=False)
        csv_paths[name] = path

    # Column name mapping
    if dataset_type == "hust":
        col_v, col_i, col_t, col_temp = 'Voltage (V)', 'Current (A)', 'Time (s)', 'Temp' # HUST may not have Temp
    else:
        col_v, col_i, col_t, col_temp = 'V', 'I', 't', 'T'

    # Group cells by batch to process batch by batch
    batch_groups = defaultdict(list)
    for cid in all_cells.keys():
        b_id = get_batch_id_func(cid, dataset_type)
        batch_groups[b_id].append(cid)
        
    for batch_id in tqdm(sorted(batch_groups.keys()), desc=f"Processing {dataset_type.upper()} Batches"):
        batch_cache_path = clean_cache_dir / f"batch_clean_{batch_id}.pkl"
        if not batch_cache_path.exists():
            print(f"Warning: Cache not found for batch {batch_id}")
            continue
            
        with open(batch_cache_path, "rb") as f:
            batch_data = pickle.load(f)
            
        for cid in tqdm(batch_groups[batch_id], desc=f"  Cells in {batch_id}", leave=False):
            cell_results = {name: [] for name in scenarios.keys()}
            
            if cid not in batch_data: continue
            
            # Process each cycle
            cycles = sorted(batch_data[cid].keys())
            for cyc in cycles:
                df_cyc = batch_data[cid][cyc]
                _, cap_label = get_cell_labels_func(all_cells[cid], cyc, dataset_type)
                
                if cap_label <= 0: continue
                
                for scen_name, (mode, start_p, end_p) in scenarios.items():
                    res = get_scenario_stats(
                        df_cyc, cid, cyc, mode, scen_name, 
                        start_p, end_p, cap_label,
                        col_v, col_i, col_t, col_temp
                    )
                    if res:
                        cell_results[scen_name].append(res)
            
            # Append this cell's results to the CSVs and free memory
            for scen_name, results in cell_results.items():
                if results:
                    pd.DataFrame(results).to_csv(csv_paths[scen_name], mode='a', header=False, index=False)
            
            del cell_results
            gc.collect()
            
        del batch_data
        gc.collect()

    print(f"Finished generating scenario CSVs for {dataset_type}")

# This script is intended to be used as a utility within the notebook environment.
# Example usage (to be copied/adapted into preprocess.ipynb):
# 
# if START_PHASE <= 2:
#     output_dir = PROCESSED_DATA_ROOT / "scenario_summaries"
#     output_dir.mkdir(exist_ok=True)
#     process_scenarios(DATASET_TYPE, all_cells, CLEAN_CACHE_DIR, output_dir, get_batch_id, get_cell_labels)
