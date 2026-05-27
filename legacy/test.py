import pickle
import pandas as pd

batch_dict = None
with open(
    "D:\chanminLee\data_store\LFP_SOH_estimation\case_5\mit_sliced_chunks\\batch_b1.pkl",
    "rb",
) as f:
    batch_dict = pickle.load(f)["segments"]

batch_dict = [
    item
    for item in batch_dict
    if item["cell"] == "b1c0"
    and item["cyc"] in [i for i in range(134, 145, 5)]
    and item["mode_label"] == 0
    and item["soc_label"] == -2
]

# 세그먼트의 전력 분산 구하기
tdf = pd.DataFrame()
for i, item in enumerate(batch_dict):
    df = item["df"]
    df["Power"] = df["Voltage (V)"] * df["Current (A)"]  # W 단위로 변환
    df["idx"] = i
    tdf = pd.concat([tdf, df], ignore_index=True)
    power_variance = df["Power"].var()

tdf.to_csv("./test2.csv", index=False)

# D:\chanminLee\data_store\LFP_SOH_estimation\case_5\mit_sliced_chunks/batch_1.pkl
