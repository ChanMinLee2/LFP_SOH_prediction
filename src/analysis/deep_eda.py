
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
from tqdm import tqdm
import warnings

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.data.loaders import load_mit, load_hust, MITCell, HUSTCell

# Suppress warnings
warnings.filterwarnings('ignore')

def calculate_dqdv(v, q, window=5, poly=1):
    """Calculate dQ/dV with smoothing."""
    # Ensure sorted by V
    idx = np.argsort(v)
    v_s, q_s = v[idx], q[idx]
    
    # Remove duplicates
    v_u, unique_idx = np.unique(v_s, return_index=True)
    q_u = q_s[unique_idx]
    
    if len(v_u) < 5: return None, None
    
    # Difference
    dv = np.diff(v_u)
    dq = np.diff(q_u)
    
    # Avoid div by zero
    valid = dv > 1e-5
    dqdv = dq[valid] / dv[valid]
    v_mid = v_u[:-1][valid]
    
    # Optional: smooth the result
    if len(dqdv) > window:
        # Window must be odd
        w = window if window % 2 != 0 else window + 1
        if len(dqdv) > w:
            dqdv = savgol_filter(dqdv, w, poly)
        
    return v_mid, dqdv

def main():
    print("Starting Deep EDA...")
    
    # 1. Setup Output
    output_dir = Path("outputs/results")
    fig_dir = Path("outputs/figures/eda")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / "deep_dataset_analysis.txt"
    report = []
    report.append("="*95)
    report.append("              DEEP DATASET EDA & CHARACTERIZATION REPORT (ENHANCED)")
    report.append("="*95)
    report.append(f"Generated on: {pd.Timestamp.now()}")

    # 2. Load Data
    print("Loading MIT and HUST datasets...")
    mit_cells = load_mit()
    hust_cells = load_hust()

    # --- TASK 1: Structural Audit ---
    report.append("\n[1. STRUCTURAL AUDIT]")
    mit_lives = [c.cycle_life for c in mit_cells.values() if c.cycle_life is not None]
    report.append(f"MIT-Stanford Dataset:")
    report.append(f"  - Total Cells: {len(mit_cells)}")
    report.append(f"  - Avg Cycle Life: {np.mean(mit_lives):.1f} (min={min(mit_lives)}, max={max(mit_lives)})")
    
    hust_lives = [c.cycle_life for c in hust_cells.values()]
    report.append(f"HUST Dataset:")
    report.append(f"  - Total Cells: {len(hust_cells)}")
    report.append(f"  - Avg Cycle Life: {np.mean(hust_lives):.1f} (min={min(hust_lives)}, max={max(hust_lives)})")

    # --- TASK 2: Domain-Specific Analysis & Time-Series Characterization ---
    print("Analyzing Profiles & Time-Series Characterization...")
    
    # Figure 1: IC/DV & V-Q Characterization
    fig1, axes1 = plt.subplots(2, 2, figsize=(20, 15))
    sns.set_theme(style="whitegrid")

    # A. MIT IC Curves
    ax = axes1[0, 0]
    cid_m = list(mit_cells.keys())[0]
    cell_m = mit_cells[cid_m]
    cycs_m = [10, cell_m.n_cycles//2, cell_m.n_cycles - 10]
    v_grid = np.linspace(2.0, 3.6, 1000)
    for cyc in cycs_m:
        if str(cyc) in cell_m.cycles:
            ax.plot(v_grid, cell_m.cycles[str(cyc)]['dQdV'], label=f"Cycle {cyc}")
    ax.set_title(f"MIT [{cid_m}] IC (dQ/dV) Evolution", fontsize=14)
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("dQ/dV (Ah/V)")
    ax.legend()

    # B. HUST IC Curves
    ax = axes1[0, 1]
    hid_h = list(hust_cells.keys())[0]
    hcell_h = hust_cells[hid_h]
    cycs_h = [10, hcell_h.n_cycles//2, hcell_h.n_cycles - 1]
    for cyc in cycs_h:
        if cyc in hcell_h.data:
            df = hcell_h.data[cyc]
            col_v = [c for c in df.columns if "Voltage" in c][0]
            col_q = [c for c in df.columns if "Capacity" in c][0]
            col_s = [c for c in df.columns if "Status" in c][0]
            dis_df = df[df[col_s].str.contains('discharge', case=False, na=False)]
            if not dis_df.empty:
                v_ic, dqdv = calculate_dqdv(dis_df[col_v].values, dis_df[col_q].values / 1000.0)
                if v_ic is not None:
                    ax.plot(v_ic, dqdv, label=f"Cycle {cyc}")
    ax.set_title(f"HUST [{hid_h}] IC (dQ/dV) Evolution", fontsize=14)
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("dQ/dV (Ah/V)")
    ax.legend()

    # C. MIT V-Q Curves
    ax = axes1[1, 0]
    for cyc in cycs_m:
        if str(cyc) in cell_m.cycles:
            ax.plot(cell_m.cycles[str(cyc)]['Qd'], cell_m.cycles[str(cyc)]['V'], label=f"Cycle {cyc}")
    ax.set_title("MIT: V-Q Curve Migration", fontsize=14)
    ax.set_xlabel("Capacity (Ah)")
    ax.set_ylabel("Voltage (V)")
    ax.legend()

    # D. HUST V-Q Curves
    ax = axes1[1, 1]
    for cyc in cycs_h:
        if cyc in hcell_h.data:
            df = hcell_h.data[cyc]
            col_v = [c for c in df.columns if "Voltage" in c][0]
            col_q = [c for c in df.columns if "Capacity" in c][0]
            col_s = [c for c in df.columns if "Status" in c][0]
            dis_df = df[df[col_s].str.contains('discharge', case=False, na=False)]
            ax.plot(dis_df[col_q]/1000.0, dis_df[col_v], label=f"Cycle {cyc}")
    ax.set_title("HUST: V-Q Curve Migration", fontsize=14)
    ax.set_xlabel("Capacity (Ah)")
    ax.set_ylabel("Voltage (V)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(fig_dir / "ic_vq_characterization.png", dpi=300)

    # Figure 2: Time-Series (V, I, T vs Time)
    fig2, axes2 = plt.subplots(3, 2, figsize=(20, 18))
    
    # MIT TS (Charge)
    ax_v, ax_i, ax_t = axes2[0, 0], axes2[1, 0], axes2[2, 0]
    for cyc in cycs_m:
        if str(cyc) in cell_m.cycles:
            d = cell_m.cycles[str(cyc)]
            ax_v.plot(d['t'], d['V'], label=f"Cyc {cyc}")
            ax_i.plot(d['t'], d['I'], label=f"Cyc {cyc}")
            ax_t.plot(d['t'], d['T'], label=f"Cyc {cyc}")
    ax_v.set_title(f"MIT [{cid_m}] Charge Voltage vs Time", fontsize=14)
    ax_i.set_title(f"MIT [{cid_m}] Charge Current vs Time", fontsize=14)
    ax_t.set_title(f"MIT [{cid_m}] Charge Temperature vs Time", fontsize=14)
    ax_v.legend(); ax_i.legend(); ax_t.legend()

    # HUST TS (Discharge)
    ax_vh, ax_ih, ax_th = axes2[0, 1], axes2[1, 1], axes2[2, 1]
    for cyc in cycs_h:
        if cyc in hcell_h.data:
            df = hcell_h.data[cyc]
            col_t = [c for c in df.columns if "Time" in c][0]
            col_v = [c for c in df.columns if "Voltage" in c][0]
            col_i = [c for c in df.columns if "Current" in c][0]
            col_s = [c for c in df.columns if "Status" in c][0]
            dis_df = df[df[col_s].str.contains('discharge', case=False, na=False)]
            if not dis_df.empty:
                t = dis_df[col_t].values - dis_df[col_t].min()
                ax_vh.plot(t, dis_df[col_v], label=f"Cyc {cyc}")
                ax_ih.plot(t, dis_df[col_i], label=f"Cyc {cyc}")
                col_temp = [c for c in df.columns if any(x in c for x in ["Temp", "T(C)"])]
                if col_temp:
                    ax_th.plot(t, dis_df[col_temp[0]], label=f"Cyc {cyc}")
    ax_vh.set_title(f"HUST [{hid_h}] Discharge Voltage vs Time", fontsize=14)
    ax_ih.set_title(f"HUST [{hid_h}] Discharge Current vs Time", fontsize=14)
    ax_th.set_title(f"HUST [{hid_h}] Discharge Temperature vs Time", fontsize=14)
    ax_vh.legend(); ax_ih.legend(); ax_th.legend()

    plt.tight_layout()
    plt.savefig(fig_dir / "time_series_characterization.png", dpi=300)
    plt.close('all')

    # --- TASK 3: Aging Feature Evolution (SOH) & Paradox Analysis ---
    print("Analyzing SOH Degradation & Protocol Paradox...")
    report.append("\n[2. AGING FEATURE EVOLUTION & PROTOCOL ANALYSIS]")
    
    # Paradox Analysis based on metadata and provided figure references
    report.append("\n[MIT Protocol-Life Paradox Analysis]")
    report.append("  - Observation: High C-rate policies (e.g., 4.8C-4.8C) sometimes outlive lower ones (e.g., 5.6C-3C).")
    report.append("  - Metadata Evidence (batch1_meta.txt): MIT uses multi-step charging (C1 until SOC_x%, then C2).")
    report.append("  - Logical Explanation:")
    report.append("    1. Li-plating Risk: Plating occurs predominantly at high SOC. Policies like '5.6C(SOC_x)-3C' ")
    report.append("       may apply high stress at a critical SOC window where internal resistance is higher.")
    report.append("    2. Thermal Gradient: Steady high C-rate (e.g. 4.8C-4.8C) might maintain more uniform internal ")
    report.append("       temperature compared to sharp current steps (e.g. 5.6C to 3C), reducing mechanical stress.")
    report.append("    3. Multi-step Optimization: Some 'High C-rate' policies are actually more efficient multi-step ")
    report.append("       strategies that avoid the most damaging voltage/SOC regions.")

    report.append("\n[HUST C-rate vs Life Paradox Analysis]")
    report.append("  - Observation: Positive correlation (Higher Discharge C-rate -> Longer Life) in Batch 3 vs others.")
    report.append("  - Logical Explanation:")
    report.append("    1. Batch Heterogeneity: Batch 3 cells likely had superior initial manufacturing quality (thinner, ")
    report.append("       more stable SEI) compared to Batch 1 or 7, which masks the degradation effect of high C-rate.")
    report.append("    2. Self-Heating Benefit: Higher discharge rates cause internal heating. In certain ambient conditions, ")
    report.append("       this slight heating can reduce electrolyte viscosity and internal resistance, potentially ")
    report.append("       leading to more uniform utilization of active material if the duration is short.")
    report.append("    3. SEI Stabilization: Specific discharge conditions in Batch 3 might have favored a more robust ")
    report.append("       re-passivation of the SEI layer compared to lower-rate cycles.")

    # SOH Plot
    plt.figure(figsize=(12, 7))
    for bid in ['b1', 'b2', 'b3']:
        cells = [c for c in mit_cells.values() if c.batch_id == bid]
        if not cells: continue
        max_l = max(c.n_cycles for c in cells)
        avg_q = np.zeros(max_l); cnt = np.zeros(max_l)
        for c in cells:
            q = c.discharge_capacity
            avg_q[:len(q)] += q; cnt[:len(q)] += 1
        plt.plot(avg_q/cnt, label=f"MIT {bid} Avg SOH")
    
    for bid in ['1', '3', '10']:
        cells = [c for c in hust_cells.values() if c.batch_id == bid]
        if not cells: continue
        max_l = max(c.cycle_life for c in cells)
        avg_q = np.zeros(max_l+1); cnt = np.zeros(max_l+1)
        for c in cells:
            q = c.discharge_capacity
            idx = np.arange(len(q))
            avg_q[idx] += q; cnt[idx] += 1
        plt.plot((avg_q/cnt)/1000.0, '--', label=f"HUST B{bid} Avg SOH")
    
    plt.title("SOH Degradation Profile (Capacity Ah vs Cycle)")
    plt.xlabel("Cycle Number"); plt.ylabel("Capacity (Ah)"); plt.legend()
    plt.savefig(fig_dir / "soh_degradation_compare.png", dpi=300)
    plt.close()

    # Final Report Save
    report.append("\n[MODELING RECOMMENDATIONS]")
    report.append("1. Feature Engineering: Use dV/dt or dT/dt as secondary inputs to capture thermal dynamics.")
    report.append("2. Normalization: Min-Max scaling of Voltage (MIT vs HUST) to [0,1] is essential for transfer.")
    report.append("3. Sequence Length: 10-cycle window is adequate, but delta-features (cyc_n - cyc_1) are key.")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"Deep EDA complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
