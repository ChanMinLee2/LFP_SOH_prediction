
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.data.loaders import load_mit, load_hust

def get_col(df, pattern):
    cols = [c for c in df.columns if pattern in c]
    return cols[0] if cols else None

def main():
    print("Loading datasets...")
    mit_cells = load_mit()
    hust_cells = load_hust()

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.set_theme(style="whitegrid")

    # Combine all batches for consistent coloring
    # MIT batches: b1, b2, b3
    # HUST batches: 1..10
    mit_batches = sorted(list(set(c.batch_id for c in mit_cells.values())))
    hust_batches = sorted(list(set(c.batch_id for c in hust_cells.values())), key=int)
    all_batches = mit_batches + hust_batches
    palette = sns.color_palette("husl", len(all_batches))
    batch_to_color = dict(zip(all_batches, palette))

    print("Plotting MIT Time-SOC (Cycle 1)...")
    for cid, cell in mit_cells.items():
        # Cycle 1 is usually the first real cycle in MIT (Cycle 0 is often empty or metadata)
        cyc_key = "1"
        if cyc_key not in cell.cycles:
            # Try first available key
            cyc_key = sorted(cell.cycles.keys(), key=int)[0]
        
        data = cell.cycles[cyc_key]
        t = data['t'].flatten()
        q = data['Qc'].flatten() # Use Charge Capacity
        
        if len(t) > 1 and q.max() > 0:
            soc = q / q.max()
            ax.plot(t, soc, color=batch_to_color[cell.batch_id], alpha=0.3, lw=0.5)

    print("Plotting HUST Time-SOC (Cycle 1)...")
    for hid, cell in hust_cells.items():
        cyc_key = min(cell.data.keys())
        df = cell.data[cyc_key]
        
        col_t = get_col(df, "Time")
        col_q = get_col(df, "Capacity")
        col_s = get_col(df, "Status")
        
        if col_t and col_q:
            # For HUST, let's filter for Charge status if it exists to get a clean 0->1 SOC
            charge_df = df[df[col_s].str.contains('charge', case=False, na=False)]
            if charge_df.empty:
                charge_df = df
            
            t = charge_df[col_t].values
            q = charge_df[col_q].values
            
            if len(t) > 1 and q.max() > 0:
                # Normalize time to start from 0 if it doesn't
                t_norm = t - t.min()
                soc = q / q.max()
                ax.plot(t_norm, soc, color=batch_to_color[cell.batch_id], alpha=0.3, lw=0.5)

    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=batch_to_color[b], lw=2, label=f'Batch {b}') for b in all_batches]
    ax.legend(handles=legend_elements, loc='upper right', title="Batches", ncol=2, fontsize='small')

    ax.set_title("First Cycle Time-SOC Profile by Batch (MIT & HUST)", fontsize=16)
    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("SOC (normalized capacity)", fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    
    output_path = Path("outputs/figures/time_soc_first_cycle.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
