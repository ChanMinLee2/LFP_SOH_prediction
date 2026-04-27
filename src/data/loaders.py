import pickle
import numpy as np
import pandas as pd
import sys

# --- NumPy compatibility layer for pickle loading ---
# Fixes 'ModuleNotFoundError: No module named numpy._core' when loading newer pickles on older numpy
if np.__version__.startswith("1."):
    import numpy.core.multiarray

    sys.modules["numpy._core"] = sys.modules["numpy.core"]
    sys.modules["numpy._core.multiarray"] = sys.modules["numpy.core.multiarray"]
# ----------------------------------------------------

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIT_DATA_DIR = PROJECT_ROOT / "MIT_data"
HUST_DATA_DIR = PROJECT_ROOT / "HUST_data" / "data"

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class MITCell:
    """One cell from the MIT-Stanford dataset."""

    cell_id: str
    batch_id: str
    charge_policy: str
    cycle_life: Optional[int]
    summary: Dict[str, np.ndarray]
    cycles: Dict[str, Dict[str, np.ndarray]]

    @property
    def n_cycles(self) -> int:
        return len(self.cycles)


@dataclass
class HUSTCell:
    """One cell from the HUST dataset."""

    cell_id: str
    batch_id: str
    rul: Dict[int, float]
    dq: Dict[int, float]
    data: Dict[int, pd.DataFrame]


def calc_Q(I: np.ndarray, t: np.ndarray, is_charge: bool = True) -> np.ndarray:
    """Numerical integration of current to calculate capacity (Ah)."""
    dt = np.diff(t, prepend=t[0])
    mask = I > 0 if is_charge else I < 0
    dq = np.zeros_like(I)
    dq[mask] = np.abs(I[mask]) * dt[mask] / 3600.0
    return np.cumsum(dq)


# ---------------------------------------------------------------------------
# Loading functions
# ---------------------------------------------------------------------------


def load_mit(data_dir: Path = MIT_DATA_DIR) -> Dict[str, MITCell]:
    all_cells: Dict[str, MITCell] = {}
    batch_files = sorted(data_dir.glob("batch*.pkl"))
    data_batches = []
    for f in batch_files:
        with open(f, "rb") as fin:
            data_batches.append(pickle.load(fin))

    # Merge Batch 1 & 2 continuing cells
    b2_keys = ["b2c7", "b2c8", "b2c9", "b2c15", "b2c16"]
    b1_keys = ["b1c0", "b1c1", "b1c2", "b1c3", "b1c4"]
    if len(data_batches) >= 2:
        for i, bk in enumerate(b1_keys):
            if bk in data_batches[0] and b2_keys[i] in data_batches[1]:
                b1, b2 = data_batches[0][bk], data_batches[1][b2_keys[i]]
                for k in b1["summary"]:
                    # Safely hstack only array-like summary items
                    v1, v2 = b1["summary"][k], b2["summary"][k]
                    if isinstance(v1, (list, np.ndarray)) and isinstance(
                        v2, (list, np.ndarray)
                    ):
                        b1["summary"][k] = np.hstack((v1, v2))

                # Update cycles with shifted indices
                max_cyc = max(map(int, b1["cycles"].keys()))
                b1["cycles"].update(
                    {
                        str(int(k) + max_cyc): v
                        for k, v in b2["cycles"].items()
                        if int(k) > 0
                    }
                )
                # Re-calculate cycle life after merge
                if "IR" in b1["summary"]:
                    b1["cycle_life"] = np.array([len(b1["summary"]["IR"])])
                del data_batches[1][b2_keys[i]]

    for i, batch_dict in enumerate(data_batches):
        for cid, d in batch_dict.items():
            # Robustly extract cycle_life
            try:
                cl = d.get("cycle_life")
                if cl is not None:
                    if hasattr(cl, "item"):
                        life = int(cl.item())
                    else:
                        life = (
                            int(cl[0])
                            if isinstance(cl, (list, np.ndarray))
                            else int(cl)
                        )
                else:
                    life = None
            except:
                life = None

            # Clean summary dictionary
            summary_clean = {}
            for k, v in d.get("summary", {}).items():
                if isinstance(v, (list, np.ndarray)):
                    summary_clean[k] = np.array(v)
                else:
                    summary_clean[k] = v

            # Clean cycles dictionary with safe array conversion
            cycles_clean = {}
            for k, v in d.get("cycles", {}).items():
                if isinstance(v, dict):
                    cycles_clean[str(k)] = {
                        ik: np.array(iv) if isinstance(iv, (list, np.ndarray)) else iv
                        for ik, iv in v.items()
                    }
                else:
                    cycles_clean[str(k)] = v

            all_cells[cid] = MITCell(
                cid,
                f"b{i+1}",
                d.get("charge_policy", "unknown"),
                life,
                summary_clean,
                cycles_clean,
            )
    return all_cells


def load_hust(
    data_dir: Path = HUST_DATA_DIR, cell_ids: Optional[List[str]] = None
) -> Dict[str, HUSTCell]:
    cells: Dict[str, HUSTCell] = {}
    paths = (
        [data_dir / f"{cid}.pkl" for cid in cell_ids]
        if cell_ids
        else sorted(
            data_dir.glob("*.pkl"), key=lambda p: [int(x) for x in p.stem.split("-")]
        )
    )

    for p in tqdm(paths, desc="HUST Cells"):
        if p.exists():
            with open(p, "rb") as f:
                raw = pickle.load(f)
                cid = p.stem
                inner = raw[cid]
                cells[cid] = HUSTCell(
                    cid,
                    cid.split("-")[0],
                    {int(k): float(v) for k, v in inner["rul"].items()},
                    {int(k): float(v) for k, v in inner["dq"].items()},
                    {int(k): v for k, v in inner["data"].items()},
                )
    return cells
