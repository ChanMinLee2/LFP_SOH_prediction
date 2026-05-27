import pickle
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# --- NumPy compatibility layer for pickle loading ---
if np.__version__.startswith("1."):
    import numpy.core.multiarray
    sys.modules["numpy._core"] = sys.modules.get("numpy.core")
    sys.modules["numpy._core.multiarray"] = sys.modules.get("numpy.core.multiarray")
# ----------------------------------------------------

def get_structure_info(obj, indent=0):
    """
    Recursively inspects the structure of a nested object (dict, list, ndarray, etc.)
    and returns a formatted string breakdown.
    """
    spacing = "  " * indent
    info = ""

    if isinstance(obj, dict):
        info += f"{type(obj).__name__} (Keys: {len(obj)})\n"
        # To avoid infinite recursion or massive output, only inspect first few keys if many
        keys = list(obj.keys())
        display_keys = keys[:5]
        for k in display_keys:
            info += f"{spacing}- ['{k}']: "
            info += get_structure_info(obj[k], indent + 1)
        if len(keys) > 5:
            info += f"{spacing}... and {len(keys)-5} more keys.\n"
            
    elif isinstance(obj, list):
        info += f"list (Length: {len(obj)})\n"
        if len(obj) > 0:
            info += f"{spacing}  [Sample Item]: "
            info += get_structure_info(obj[0], indent + 1)
            
    elif isinstance(obj, np.ndarray):
        info += f"numpy.ndarray (Shape: {obj.shape}, Dtype: {obj.dtype})\n"
        
    elif isinstance(obj, pd.DataFrame):
        info += f"pandas.DataFrame (Shape: {obj.shape})\n"
        info += f"{spacing}  Columns: {obj.columns.tolist()}\n"
        
    else:
        # Scalar or other type
        val_str = str(obj)
        if len(val_str) > 50: val_str = val_str[:47] + "..."
        info += f"{type(obj).__name__} (Value: {val_str})\n"

    return info

def main():
    output_file = "dataset_structure_breakdown.txt"
    report = "=== MIT & HUST Dataset Structure Breakdown ===\n\n"

    # --- 1. MIT Dataset Inspection ---
    mit_path = Path("MIT_data/batch1.pkl")
    if mit_path.exists():
        report += "--- [1] MIT Dataset (batch1.pkl) ---\n"
        try:
            with open(mit_path, "rb") as f:
                mit_data = pickle.load(f)
            report += get_structure_info(mit_data)
            del mit_data
        except Exception as e:
            report += f"Error loading MIT data: {e}\n"
    else:
        report += "MIT batch1.pkl not found.\n"

    report += "\n" + "="*50 + "\n\n"

    # --- 2. HUST Dataset Inspection ---
    hust_path = Path("HUST_data/data/1-1.pkl")
    if hust_path.exists():
        report += "--- [2] HUST Dataset (1-1.pkl) ---\n"
        try:
            with open(hust_path, "rb") as f:
                hust_data = pickle.load(f)
            report += get_structure_info(hust_data)
            del hust_data
        except Exception as e:
            report += f"Error loading HUST data: {e}\n"
    else:
        report += "HUST 1-1.pkl not found.\n"

    # Write to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"Structure breakdown saved to {output_file}")

if __name__ == "__main__":
    main()
