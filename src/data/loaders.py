"""
Data loaders for MIT-Stanford and HUST LFP battery datasets.
Refactored for reproducibility based on guideline.md.
"""

import pickle
import re
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, List, Any

import numpy as np
import pandas as pd

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# NumPy 2.x -> 1.x compatibility hack for loading older/newer pickles
if np.__version__.startswith('1.'):
    import numpy.core.multiarray
    import numpy.core.numeric
    import numpy.core.fromnumeric
    import numpy.core.defchararray
    import numpy.core.records
    import numpy.core.memmap
    import numpy.core.function_base

    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray
    sys.modules['numpy._core.numeric'] = np.core.numeric
    sys.modules['numpy._core.fromnumeric'] = np.core.fromnumeric
    sys.modules['numpy._core.defchararray'] = np.core.defchararray
    sys.modules['numpy._core.records'] = np.core.records
    sys.modules['numpy._core.memmap'] = np.core.memmap
    sys.modules['numpy._core.function_base'] = np.core.function_base
elif np.__version__.startswith('2.'):
    try:
        sys.modules['numpy.core'] = np._core
        sys.modules['numpy.core.multiarray'] = np._core.multiarray
    except AttributeError:
        pass
        pass

# ---------------------------------------------------------------------------
# Paths (relative to project root; override via MIT_DATA_DIR / HUST_DATA_DIR)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIT_DATA_DIR = PROJECT_ROOT / "MIT_data"
HUST_DATA_DIR = PROJECT_ROOT / "HUST_data" / "data"


# ---------------------------------------------------------------------------
# Data containers (Enhanced based on guideline.md)
# ---------------------------------------------------------------------------


@dataclass
class CycleData:
    """Standardized per-cycle data."""

    cycle_number: int
    voltage_in_V: List[float]
    current_in_A: List[float]
    time_in_s: List[float]
    discharge_capacity_in_Ah: List[float]
    charge_capacity_in_Ah: List[float]
    # Metadata / Summary stats
    summary: Dict[str, float] = field(default_factory=dict)


@dataclass
class CyclingProtocol:
    """Standardized protocol stage."""

    rate_in_C: float
    start_soc: float
    end_soc: float
    type: str = "charge"  # "charge" or "discharge"


@dataclass
class BatteryData:
    """Unified battery data container."""

    cell_id: str
    cycle_data: List[CycleData]
    form_factor: str = "cylindrical_18650"
    anode_material: str = "graphite"
    cathode_material: str = "LFP"
    nominal_capacity_in_Ah: float = 1.1
    min_voltage_limit_in_V: float = 2.0
    max_voltage_limit_in_V: float = 3.5
    charge_protocol: List[CyclingProtocol] = field(default_factory=list)
    discharge_protocol: List[CyclingProtocol] = field(default_factory=list)


def calc_Q(I: np.ndarray, t: np.ndarray, is_charge: bool = True) -> np.ndarray:
    """
    Calculate cumulative capacity (Ah) using numerical integration.
    I: current in Amperes, t: time in seconds.
    Returns: numpy array of cumulative capacity.
    """
    dt = np.diff(t, prepend=t[0])
    if is_charge:
        mask = I > 0
    else:
        mask = I < 0

    dq = np.zeros_like(I)
    # Only integrate where mask is true, otherwise keep 0
    valid_dq = np.abs(I[mask]) * dt[mask] / 3600.0
    dq[mask] = valid_dq
    return np.cumsum(dq)


# ---------------------------------------------------------------------------
# Legacy compatibility classes (Wrapped around new structures or vice versa)
# ---------------------------------------------------------------------------


@dataclass
class MITCell:
    """One cell from the MIT-Stanford dataset."""

    cell_id: str
    batch_id: str  # 'b1', 'b2', 'b3'
    charge_policy: str
    cycle_life: Optional[int]
    summary: Dict[str, np.ndarray]
    cycles: Dict[str, Dict[str, np.ndarray]]

    # Lazy-loaded cached metrics
    _total_discharge_time: Optional[float] = field(default=None, init=False, repr=False)
    _throughput_wh: Optional[float] = field(default=None, init=False, repr=False)

    @property
    def n_cycles(self) -> int:
        return len(self.cycles)

    @property
    def discharge_capacity(self) -> np.ndarray:
        return self.summary["QD"]

    def _get_time_in_seconds(self, t_data: np.ndarray) -> np.ndarray:
        if len(t_data) < 2:
            return t_data
        dt = np.median(np.diff(t_data))
        if dt < 0.1:
            return t_data * 60.0
        return t_data

    def get_total_discharge_time(self) -> float:
        if self._total_discharge_time is not None:
            return self._total_discharge_time
        cyc_keys = sorted([int(k) for k in self.cycles.keys() if int(k) > 0])
        if not cyc_keys:
            return 0.0
        samples = [
            str(k)
            for k in (
                np.linspace(0, len(cyc_keys) - 1, 5, dtype=int)
                if len(cyc_keys) > 5
                else range(len(cyc_keys))
            )
        ]
        avg_duration = 0.0
        for cyc_key in samples:
            try:
                data = self.cycles[
                    str(cyc_keys[int(cyc_key)]) if cyc_key.isdigit() else cyc_key
                ]
                t_sec = self._get_time_in_seconds(data["t"])
                dis_mask = data["I"] < -0.1
                if np.any(dis_mask):
                    avg_duration += t_sec[dis_mask].max() - t_sec[dis_mask].min()
            except:
                continue
        self._total_discharge_time = (avg_duration / len(samples)) * len(cyc_keys)
        return self._total_discharge_time

    def get_throughput_wh(self) -> float:
        if self._throughput_wh is not None:
            return self._throughput_wh
        total_wh = 0.0
        for cyc_key in self.cycles:
            try:
                if int(cyc_key) <= 0:
                    continue
                data = self.cycles[cyc_key]
                t_sec = self._get_time_in_seconds(data["t"])
                i, v = np.abs(data["I"]), data["V"]
                mask = data["I"] < -0.1
                if np.any(mask):
                    dt = np.diff(t_sec[mask], prepend=t_sec[mask][0])
                    total_wh += np.sum(v[mask] * i[mask] * dt) / 3600.0
            except:
                continue
        self._throughput_wh = total_wh
        return total_wh

    def get_cumulative_throughput_ah(self) -> float:
        return np.sum(self.summary["QD"])

    def to_battery_data(self) -> BatteryData:
        cycle_list = []
        cyc_keys = sorted([int(k) for k in self.cycles.keys() if int(k) > 0])
        for k in cyc_keys:
            data = self.cycles[str(k)]
            t_sec = self._get_time_in_seconds(data["t"])
            curr = data["I"]
            qc_calc = calc_Q(curr, t_sec, is_charge=True)
            qd_calc = calc_Q(curr, t_sec, is_charge=False)
            cycle_list.append(
                CycleData(
                    cycle_number=k,
                    voltage_in_V=data["V"].tolist(),
                    current_in_A=curr.tolist(),
                    time_in_s=t_sec.tolist(),
                    discharge_capacity_in_Ah=qd_calc.tolist(),
                    charge_capacity_in_Ah=qc_calc.tolist(),
                    summary={
                        "IR": (
                            float(self.summary["IR"][k - 1])
                            if k - 1 < len(self.summary["IR"])
                            else 0.0
                        )
                    },
                )
            )
        charge_protocol = []
        stages = [x for x in self.charge_policy.split("-") if "new" not in x]
        if len(stages) == 2:
            pattern = r"(.*?)C\((.*?)%\)"
            matches = re.findall(pattern, stages[0])
            if matches:
                rate1, end_soc = matches[0]
                rate2_match = re.search(r"(\d+\.?\d*)C", stages[1])
                rate2 = float(rate2_match.group(1)) if rate2_match else 0.0
                charge_protocol = [
                    CyclingProtocol(
                        rate_in_C=float(rate1),
                        start_soc=0.0,
                        end_soc=float(end_soc) / 100.0,
                    ),
                    CyclingProtocol(
                        rate_in_C=float(rate2),
                        start_soc=float(end_soc) / 100.0,
                        end_soc=1.0,
                    ),
                ]
        return BatteryData(
            cell_id=f"MATR_{self.cell_id}",
            cycle_data=cycle_list,
            charge_protocol=charge_protocol,
        )

    def get_charge_load(self) -> float:
        pattern = re.compile(r"(\d+\.?\d*)C\((\d+)%\)-(\d+\.?\d*)C")
        match = pattern.search(self.charge_policy)
        if match:
            c1, q1, c2 = (
                float(match.group(1)),
                float(match.group(2)),
                float(match.group(3)),
            )
            return (c1 * q1) + (c2 * (80.0 - q1))
        return 0.0


@dataclass
class HUSTCell:
    """One cell from the HUST dataset."""

    cell_id: str
    batch_id: str
    rul: Dict[int, float]
    dq: Dict[int, float]
    data: Dict[int, pd.DataFrame]
    _throughput_wh: Optional[float] = field(default=None, init=False, repr=False)

    @property
    def n_cycles(self) -> int:
        return len(self.data)

    @property
    def cycle_life(self) -> int:
        return max(self.data.keys())

    def get_cycle_stats(self, cycle_num: int = 1) -> Dict[str, float]:
        if cycle_num not in self.data:
            cycle_num = sorted(self.data.keys())[0]
        df = self.data[cycle_num]
        col_s, col_t, col_i = (
            [c for c in df.columns if "Status" in c][0],
            [c for c in df.columns if "Time" in c][0],
            [c for c in df.columns if "Current" in c][0],
        )
        dis_df = df[df[col_s].str.contains("discharge", case=False, na=False)]
        stats = {
            "dis_duration": 0.0,
            "c_rate": 0.0,
            "total_duration": df[col_t].max() - df[col_t].min(),
        }
        if not dis_df.empty:
            stats["dis_duration"] = dis_df[col_t].max() - dis_df[col_t].min()
            stats["c_rate"] = abs(dis_df[col_i].mean()) / 1100.0
            for i in range(4):
                stage_df = dis_df[
                    dis_df[col_s].str.contains(f"discharge_{i}", case=False, na=False)
                ]
                if not stage_df.empty:
                    stats[f"D{i}_duration"], stats[f"D{i}_current"] = stage_df[
                        col_t
                    ].max() - stage_df[col_t].min(), abs(stage_df[col_i].mean())
                    stats[f"D{i}_c_rate"] = stats[f"D{i}_current"] / 1100.0
        return stats

    def get_effective_c_rate(self, cycle_num: int = 1) -> float:
        """Calculate simple average of C-rates for stages D0-D3 (duration independent)."""
        stats = self.get_cycle_stats(cycle_num=cycle_num)
        rates = [
            stats.get(f"D{j}_c_rate", 0.0)
            for j in range(4)
            if stats.get(f"D{j}_duration", 0.0) > 0
        ]
        return float(np.mean(rates)) if rates else 0.0

    def get_rms_c_rate(self, cycle_num: int = 1) -> float:
        if cycle_num not in self.data:
            cycle_num = sorted(self.data.keys())[0]
        df = self.data[cycle_num]
        col_s, col_t, col_i = (
            [c for c in df.columns if "Status" in c][0],
            [c for c in df.columns if "Time" in c][0],
            [c for c in df.columns if "Current" in c][0],
        )
        dis_df = df[df[col_s].str.contains("discharge", case=False, na=False)]
        if dis_df.empty:
            return 0.0
        currents_c, times = np.abs(dis_df[col_i].values) / 1100.0, dis_df[col_t].values
        dt = np.diff(times, prepend=times[0])
        return float(np.sqrt(np.sum((currents_c**2) * dt) / np.sum(dt)))

    def get_cumulative_throughput_ah(self) -> float:
        return sum(self.dq.values()) / 1000.0

    def get_avg_discharge_power(self) -> float:
        """Calculate the mean discharge power (W) averaged over ALL available cycles."""
        powers = []
        for df in self.data.values():
            col_s = [c for c in df.columns if "Status" in c][0]
            col_i = [c for c in df.columns if "Current" in c][0]
            col_v = [c for c in df.columns if "Voltage" in c][0]

            dis_df = df[df[col_s].str.contains("discharge", case=False, na=False)]
            if not dis_df.empty:
                avg_v = dis_df[col_v].mean()
                avg_i_a = abs(dis_df[col_i].mean()) / 1000.0
                powers.append(avg_v * avg_i_a)
        return float(np.mean(powers)) if powers else 0.0

    def get_throughput_wh(self) -> float:
        if self._throughput_wh is not None:
            return self._throughput_wh
        cyc_nums = sorted(self.data.keys())
        if not cyc_nums:
            return 0.0
        df = self.data[cyc_nums[len(cyc_nums) // 2]]
        col_s, col_t, col_i, col_v = (
            [c for c in df.columns if "Status" in c][0],
            [c for c in df.columns if "Time" in c][0],
            [c for c in df.columns if "Current" in c][0],
            [c for c in df.columns if "Voltage" in c][0],
        )
        dis_df = df[df[col_s].str.contains("discharge", case=False, na=False)]
        wh_rep = 0.0
        if not dis_df.empty:
            s_df = dis_df.iloc[:: max(1, len(dis_df) // 20)]
            v, i, t = (
                s_df[col_v].values,
                np.abs(s_df[col_i].values) / 1000.0,
                s_df[col_t].values,
            )
            wh_rep = np.sum(v * i * np.diff(t, prepend=t[0])) / 3600.0
        self._throughput_wh = wh_rep * len(cyc_nums)
        return self._throughput_wh

    def to_battery_data(self) -> BatteryData:
        cycle_list = []
        cyc_keys = sorted(self.data.keys())
        if self.cell_id == "7-5":
            cyc_keys = cyc_keys[2:]
        for k in cyc_keys:
            df = self.data[k]
            col_i, col_t, col_v = (
                [c for c in df.columns if "Current" in c][0],
                [c for c in df.columns if "Time" in c][0],
                [c for c in df.columns if "Voltage" in c][0],
            )
            curr_a, time_s, volt_v = (
                df[col_i].values / 1000.0,
                df[col_t].values,
                df[col_v].values,
            )
            cycle_list.append(
                CycleData(
                    k,
                    volt_v.tolist(),
                    curr_a.tolist(),
                    time_s.tolist(),
                    calc_Q(curr_a, time_s, False).tolist(),
                    calc_Q(curr_a, time_s, True).tolist(),
                    {"rul": float(self.rul[k])},
                )
            )
        return BatteryData(f"HUST_{self.cell_id}", cycle_list)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_mit(data_dir: Path = MIT_DATA_DIR) -> Dict[str, MITCell]:
    all_cells: Dict[str, MITCell] = {}
    batch_files = sorted(data_dir.glob("batch*.pkl"))
    data_batches = []
    for f in batch_files:
        with open(f, "rb") as fin:
            data_batches.append(pickle.load(fin))

    # Merge Batch 1 & 2 continuing cells per guideline
    b2_keys, b1_keys = ["b2c7", "b2c8", "b2c9", "b2c15", "b2c16"], [
        "b1c0",
        "b1c1",
        "b1c2",
        "b1c3",
        "b1c4",
    ]
    if len(data_batches) >= 2:
        for i, bk in enumerate(b1_keys):
            if bk in data_batches[0] and b2_keys[i] in data_batches[1]:
                b1, b2 = data_batches[0][bk], data_batches[1][b2_keys[i]]
                for k in b1["summary"]:
                    b1["summary"][k] = np.hstack((b1["summary"][k], b2["summary"][k]))
                b1["cycles"].update(
                    {
                        str(int(k) + max(map(int, b1["cycles"].keys()))): v
                        for k, v in b2["cycles"].items()
                        if int(k) > 0
                    }
                )
                b1["cycle_life"] = np.array([len(b1["summary"]["IR"])])
                del data_batches[1][b2_keys[i]]

    for i, batch_dict in enumerate(data_batches):
        for cid, d in batch_dict.items():
            life = (
                None
                if np.isnan(d["cycle_life"].item())
                else int(d["cycle_life"].item())
            )
            all_cells[cid] = MITCell(
                cid,
                f"b{i+1}",
                d["charge_policy"],
                life,
                {k: np.array(v) for k, v in d["summary"].items()},
                d["cycles"],
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


# if __name__ == "__main__":
