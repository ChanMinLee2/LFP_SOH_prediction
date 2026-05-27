import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

# Set project root to the directory containing this script
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.append(str(PROJECT_ROOT))

# Import data loaders
try:
    from src.data.loaders import load_hust, load_mit
except ImportError as e:
    print(f"[Error] Failed to import loaders: {e}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

def collect_capacity_data():
    all_data = []

    # --- 1. MIT Dataset 분석 ---
    mit_path = PROJECT_ROOT / "MIT_data"
    print(f"\n[1/2] Loading MIT Dataset from {mit_path}...")
    if mit_path.exists():
        try:
            # load_mit returns a dict of Battery objects
            mit_cells = load_mit(mit_path)
            for cid, cell in tqdm(mit_cells.items(), desc="Processing MIT Cells"):
                # MIT battery objects have a 'summary' dict with 'QD' (Ah)
                q_list = cell.summary.get('QD', [])
                for cyc_idx, q_val in enumerate(q_list):
                    # Handle cases where q_val might be a list/array or scalar
                    val = q_val[0] if isinstance(q_val, (list, np.ndarray)) else q_val
                    if val > 0:
                        all_data.append({
                            'Dataset': 'MIT',
                            'Cell_ID': cid,
                            'Cycle': cyc_idx + 1,
                            'Capacity_Ah': float(val)
                        })
        except Exception as e:
            print(f"Error processing MIT: {e}")
    else:
        print("MIT data path not found.")

    # --- 2. HUST Dataset 분석 ---
    hust_path = PROJECT_ROOT / "HUST_data" / "data"
    print(f"\n[2/2] Loading HUST Dataset from {hust_path}...")
    if hust_path.exists():
        try:
            # load_hust returns a dict of Battery objects
            hust_cells = load_hust(hust_path)
            for cid, cell in tqdm(hust_cells.items(), desc="Processing HUST Cells"):
                # HUST battery objects have a 'dq' dict with mAh values
                for cyc, q_mah in cell.dq.items():
                    q_ah = q_mah / 1000.0
                    if q_ah > 0:
                        all_data.append({
                            'Dataset': 'HUST',
                            'Cell_ID': cid,
                            'Cycle': cyc,
                            'Capacity_Ah': float(q_ah)
                        })
        except Exception as e:
            print(f"Error processing HUST: {e}")
    else:
        print("HUST data path not found.")

    return pd.DataFrame(all_data)

def main():
    # Ensure output directory exists
    output_dir = PROJECT_ROOT / "outputs" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = collect_capacity_data()
    
    if df.empty:
        print("\n[Error] No capacity data collected.")
        return

    # --- Statistics ---
    print("\n" + "="*60)
    print("      CAPACITY DISTRIBUTION STATISTICS (Ah)")
    print("="*60)
    stats = df.groupby('Dataset')['Capacity_Ah'].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])
    print(stats.round(4))
    
    print("\n" + "-"*60)
    print("ANALYSIS: NOMINAL CAPACITY (1.1 Ah) VIOLATION CHECK")
    print("-"*60)
    for ds in df['Dataset'].unique():
        ds_df = df[df['Dataset'] == ds]
        over_1_1 = (ds_df['Capacity_Ah'] > 1.1).sum()
        total = len(ds_df)
        print(f"[{ds}] Samples > 1.1Ah: {over_1_1:,} / {total:,} ({over_1_1/total*100:.2f}%)")
        print(f"[{ds}] Max Capacity  : {ds_df['Capacity_Ah'].max():.4f} Ah")
        
        # Check for 1.2Ah which was specifically mentioned
        over_1_2 = (ds_df['Capacity_Ah'] > 1.2).sum()
        print(f"[{ds}] Samples > 1.2Ah: {over_1_2:,} / {total:,} ({over_1_2/total*100:.2f}%)")

    # --- Visualization ---
    plt.figure(figsize=(12, 12))
    
    # 1. Histogram / KDE
    plt.subplot(2, 1, 1)
    sns.histplot(data=df, x='Capacity_Ah', hue='Dataset', kde=True, bins=100, alpha=0.5)
    plt.axvline(1.1, color='red', linestyle='--', linewidth=2, label='Nominal 1.1Ah')
    plt.axvline(1.2, color='darkred', linestyle=':', linewidth=2, label='1.2Ah threshold')
    plt.title('Capacity Distribution by Dataset (Ah)', fontsize=15, fontweight='bold')
    plt.xlabel('Measured Discharge Capacity (Ah)')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Boxplot
    plt.subplot(2, 1, 2)
    sns.boxplot(data=df, x='Capacity_Ah', y='Dataset', palette="muted", orient='h')
    plt.axvline(1.1, color='red', linestyle='--', linewidth=2, label='Nominal 1.1Ah')
    plt.title('Capacity Range and Outliers', fontsize=15, fontweight='bold')
    plt.xlabel('Capacity (Ah)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    
    img_save_path = output_dir / "capacity_distribution_analysis.png"
    plt.savefig(img_save_path, dpi=300, bbox_inches='tight')
    
    csv_save_path = output_dir / "capacity_raw_data.csv"
    # To avoid huge files, we can save a summary or just skip if not needed, 
    # but here we save it for user verification
    # df.to_csv(csv_save_path, index=False)
    
    print("\n" + "="*60)
    print(f"[SUCCESS] Analysis complete.")
    print(f"Plot saved to: {img_save_path}")
    # print(f"Raw data saved to: {csv_save_path}")
    print("="*60)

if __name__ == "__main__":
    main()
