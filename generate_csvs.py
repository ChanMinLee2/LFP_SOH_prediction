import pickle
import pandas as pd
import os
import glob
from pathlib import Path
from tqdm import tqdm

import sys
import numpy.core.multiarray
import numpy.core.umath

# NumPy 2.x 환경에서 저장된 _core 모듈 호출을 NumPy 1.x의 core로 리다이렉션
sys.modules["numpy._core"] = sys.modules.get("numpy.core")
sys.modules["numpy._core.multiarray"] = sys.modules.get("numpy.core.multiarray")
sys.modules["numpy._core.umath"] = sys.modules.get("numpy.core.umath")


def convert_mit_to_csv(mit_dir, output_dir):
    """
    MIT-Stanford dataset (.pkl) files to CSV.
    Extracts summary statistics for each cell and saves them.
    """
    mit_dir = Path(mit_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_files = [sorted(mit_dir.glob("batch*.pkl"))[0]]
    # batch_files = sorted(mit_dir.glob(["batch1.pkl"]))
    print(batch_files)
    for bf in batch_files:
        # print(f"Processing MIT {bf.name}...")
        with open(bf, "rb") as f:
            batch_dict = pickle.load(f)

        # pdb.set_trace()
        # all_cell_summaries = []
        all_cell_summaries = pd.DataFrame()
        for cell_id, cell_data in batch_dict.items():
            if cell_id != "b1c0":
                continue

            for cyc, qd in cell_data["cycles"].items():
                del qd["Qdlin"]
                del qd["Tdlin"]
                del qd["dQdV"]
                del qd["Qc"]
                del qd["Qd"]
                del qd["T"]
                qd["cyc"] = int(cyc)
                qd["DH"] = qd["I"] * qd["V"]
                # print(cyc)
                # print(qd.keys())
                # print([(key, len(qd[key])) for key in qd.keys()])

                df = pd.DataFrame(qd)
                all_cell_summaries = pd.concat(
                    [all_cell_summaries, df], ignore_index=True
                )
                # all_cell_summaries.append(df)
            # if "summary" in cell_data:
            #     # df = pd.DataFrame(cell_data["summary"])
            #     df = pd.DataFrame(cell_data["cycles"])
            #     print(cell_data["cycles"]["1"].keys())
            #     df.insert(0, "cell_id", cell_id)
            #     all_cell_summaries.append(df)
        # if all_cell_summaries:
        # final_df = pd.concat(all_cell_summaries, ignore_index=True)
        # final_df = all_cell_summaries
        # csv_name = bf.stem + ".csv"
        csv_name = "test.csv"
        all_cell_summaries.to_csv(output_dir / csv_name, index=False)
        print(f"  Saved {csv_name}")


def convert_hust_to_csv(hust_dir, output_dir):
    """
    HUST dataset (.pkl) files to CSV.
    Joins raw time-series data with cycle-level labels (dq, rul).
    """
    hust_dir = Path(hust_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pkl_files = sorted(hust_dir.glob("*.pkl"))

    for pf in tqdm(pkl_files, desc="Converting HUST files"):
        with open(pf, "rb") as f:
            data = pickle.load(f)

        cell_id = list(data.keys())[0]
        cell_inner = data[cell_id]

        dq_dict = cell_inner.get("dq", {})
        rul_dict = cell_inner.get("rul", {})
        raw_data_dict = cell_inner.get("data", {})

        all_cycles = []
        for cyc in sorted(raw_data_dict.keys()):
            df = raw_data_dict[cyc].copy()
            df.insert(0, "cell_id", cell_id)
            df["dq"] = dq_dict.get(cyc, 0.0)
            df["rul"] = rul_dict.get(cyc, 0.0)

            # Reorder columns to put metadata first
            cols = ["cell_id", "Cycle number", "Status", "dq", "rul"]
            other_cols = [c for c in df.columns if c not in cols]
            df = df[cols + other_cols]
            all_cycles.append(df)

        if all_cycles:
            final_df = pd.concat(all_cycles, ignore_index=True)
            csv_name = pf.stem + ".csv"
            final_df.to_csv(output_dir / csv_name, index=False)


if __name__ == "__main__":
    # Define Paths
    MIT_INPUT = "./MIT_data"
    HUST_INPUT = "./HUST_data/data"

    # Run Conversions
    print("--- Starting MIT Conversion ---")
    # convert_mit_to_csv(MIT_INPUT, MIT_INPUT)

    print("\n--- Starting HUST Conversion ---")
    convert_hust_to_csv(HUST_INPUT, HUST_INPUT)

    print("\nConversion Complete.")
