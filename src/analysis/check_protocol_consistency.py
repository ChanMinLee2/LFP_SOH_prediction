
from src.data.loaders import load_hust
import pandas as pd
import numpy as np
from pathlib import Path

def check_protocol_consistency():
    # Load 3 diverse cells: Batch 1 (Early), Batch 5 (Mid), Batch 10 (Late)
    target_ids = ['1-1', '5-5', '10-6']
    print(f"Loading HUST cells: {target_ids} for protocol audit...")
    cells = load_hust(cell_ids=target_ids)
    
    results = []
    
    for cid, cell in cells.items():
        print(f"\n[Audit: {cid}]")
        all_cycles = sorted(cell.data.keys())
        
        # Test 1, Mid, and Last cycles
        test_points = [1, 10, 100, 500, 1000, 1500, 2000]
        test_points = [tp for tp in test_points if tp in all_cycles]
        if all_cycles[-1] not in test_points:
            test_points.append(all_cycles[-1])
            
        for cyc in test_points:
            stats = cell.get_cycle_stats(cycle_num=cyc)
            load = cell.get_discharge_load(cycle_num=cyc)
            
            results.append({
                "Cell ID": cid,
                "Cycle": cyc,
                "D0_dur": stats.get("D0_duration", 0),
                "D1_dur": stats.get("D1_duration", 0),
                "D2_dur": stats.get("D2_duration", 0),
                "D3_dur": stats.get("D3_duration", 0),
                "D0_C": stats.get("D0_c_rate", 0),
                "Total_Load": load
            })

    df = pd.DataFrame(results)
    
    # Format printing for easy visual check
    for cid in target_ids:
        print(f"\n--- Cell {cid} Protocol Timeline ---")
        sub_df = df[df["Cell ID"] == cid].drop(columns="Cell ID")
        print(sub_df.to_string(index=False))
        
        # Check standard deviation of Load for this cell
        std_load = sub_df["Total_Load"].std()
        print(f"Load Standard Deviation: {std_load:.2f}")
        if std_load < 5.0:
            print("Conclusion: Discharge Protocol is FIXED for this cell.")
        else:
            print("Conclusion: Discharge Protocol VARIES across cycles.")

if __name__ == "__main__":
    check_protocol_consistency()
