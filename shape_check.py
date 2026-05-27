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

def build_tree_str(obj, name="root", indent=0, max_dict_keys=5):
    """
    Recursively builds a tree-like string structure of the object.
    """
    spacing = "    " * indent
    connector = "└── " if indent > 0 else ""
    
    # Identify type and dimension/shape
    obj_type = type(obj).__name__
    shape_info = ""
    
    if isinstance(obj, np.ndarray):
        shape_info = f" [Shape: {obj.shape}, Dtype: {obj.dtype}]"
    elif isinstance(obj, pd.DataFrame):
        shape_info = f" [Shape: {obj.shape}, Columns: {obj.columns.tolist()}]"
    elif isinstance(obj, (list, tuple)):
        shape_info = f" [Length: {len(obj)}]"
    elif isinstance(obj, dict):
        shape_info = f" [Keys: {len(obj)}]"

    # Build current line
    line = f"{spacing}{connector}{name} ({obj_type}){shape_info}\n"
    
    # Recurse into nested structures
    content = ""
    if isinstance(obj, dict):
        keys = list(obj.keys())
        # To keep the tree readable, we only expand the first few keys
        # or special metadata keys
        for i, k in enumerate(keys):
            if i >= max_dict_keys:
                content += f"{spacing}    ... and {len(keys)-max_dict_keys} more keys\n"
                break
            content += build_tree_str(obj[k], name=str(k), indent=indent + 1)
            
    elif isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], (dict, list, np.ndarray)):
        # Inspect the first element of a list if it's a complex structure
        content += build_tree_str(obj[0], name="[0]", indent=indent + 1)

    return line + content

def main():
    output_file = "dataset_shape_tree.txt"
    tree_report = "=== MIT & HUST Dataset Hierarchical Shape Tree ===\n\n"

    # 1. MIT Sample (batch1.pkl)
    mit_path = Path("MIT_data/batch1.pkl")
    if mit_path.exists():
        tree_report += "--- MIT DATASET STRUCTURE ---\n"
        with open(mit_path, "rb") as f:
            data = pickle.load(f)
            # Inspect first cell only to show hierarchy
            first_cell_id = list(data.keys())[0]
            tree_report += build_tree_str({first_cell_id: data[first_cell_id]}, name="MIT_Batch1")
        del data
    
    tree_report += "\n" + "="*60 + "\n\n"

    # 2. HUST Sample (1-1.pkl)
    hust_path = Path("HUST_data/data/1-1.pkl")
    if hust_path.exists():
        tree_report += "--- HUST DATASET STRUCTURE ---\n"
        with open(hust_path, "rb") as f:
            data = pickle.load(f)
            tree_report += build_tree_str(data, name="HUST_1-1")
        del data

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(tree_report)
    
    print(f"Tree structure report saved to {output_file}")

if __name__ == "__main__":
    main()
