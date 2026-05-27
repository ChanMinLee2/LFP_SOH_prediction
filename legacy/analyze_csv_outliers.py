import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import glob
import pdb


def calculate_percentiles(series, percents):
    """지정된 퍼센타일 값들을 계산합니다."""
    return {f"{p}%": np.percentile(series.dropna(), p) for p in percents}


def analyze_outliers(df, dataset_name, cell_col="cell_id", cyc_col="cycle"):
    """
    1) 사이클/피처별 통계 및 퍼센타일 계산
    2) 7-시그마 기반 이상치 탐지 및 기록
    """
    print(f"\n--- Analyzing {dataset_name} ---")
    pdb.set_trace()

    # 분석할 수치형 피처만 선택
    num_cols = df.select_dtypes(include=[np.number]).columns
    # 메타데이터 컬럼은 통계에서 제외 (cyc_col 처리 방식에 따라 다름)
    exclude_cols = [cell_col, cyc_col, "Status", "batch_id", "Cycle number", "cycle"]
    features = [c for c in num_cols if c not in exclude_cols]

    # 1. 사이클별 통계 계산
    stats_records = []
    outlier_records = []

    # 데이터셋의 사이클 컬럼 확인
    cycle_column = cyc_col if cyc_col in df.columns else "Cycle number"
    if cycle_column not in df.columns:
        print(f"Warning: Cycle column not found for {dataset_name}.")
        return

    grouped = df.groupby([cell_col, cycle_column])

    for (cid, cyc), group in tqdm(grouped, desc=f"Processing {dataset_name} cells"):
        for feat in features:
            series = group[feat]

            if series.isna().all():
                continue

            mean = series.mean()
            std = series.std()
            var = series.var()

            if pd.isna(std) or std == 0:
                continue

            # 상하위 0.1%, 1%, 3%, 5% 계산
            percents = [0.1, 1.0, 3.0, 5.0, 95.0, 97.0, 99.0, 99.9]
            pct_vals = calculate_percentiles(series, percents)

            # 통계 기록 저장
            stat_row = {
                "dataset": dataset_name,
                "cell": cid,
                "cycle": cyc,
                "feature": feat,
                "mean": mean,
                "std": std,
                "var": var,
            }
            stat_row.update(pct_vals)
            stats_records.append(stat_row)

            # 2. 7-Sigma 이상치 탐지
            # 평균에서 7 시그마를 넘어서는 값을 이상치로 판단
            upper_limit = mean + 7 * std
            lower_limit = mean - 7 * std

            outliers = series[(series > upper_limit) | (series < lower_limit)]

            for idx, val in outliers.items():
                outlier_records.append(
                    f"{dataset_name} - cell:{cid} - cycle:{cyc} - row:{idx} - feature:{feat} - val:{val:.4f} (limits: [{lower_limit:.4f}, {upper_limit:.4f}])"
                )

    # 결과 저장
    out_dir = Path("./outputs/outlier_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 통계 저장
    stats_df = pd.DataFrame(stats_records)
    stats_file = out_dir / f"{dataset_name}_cycle_statistics.txt"
    stats_df.to_csv(stats_file, sep="\t", index=False)
    print(f"Saved statistics to {stats_file}")

    # 이상치 인덱스 저장
    outliers_file = out_dir / f"{dataset_name}_7sigma_outliers.txt"
    with open(outliers_file, "w") as f:
        for record in outlier_records:
            f.write(record + "\n")
    print(f"Saved outlier indices to {outliers_file} (Total: {len(outlier_records)})")


if __name__ == "__main__":
    # MIT CSV 처리
    mit_csvs = glob.glob("MIT_data/batch*.csv")
    if mit_csvs:
        print(f"Found {len(mit_csvs)} MIT CSVs.")
        mit_dfs = [
            pd.read_csv(f, encoding="utf-8", encoding_errors="replace")
            for f in mit_csvs
        ]
        mit_combined = pd.concat(mit_dfs, ignore_index=True)
        analyze_outliers(mit_combined, "MIT", cyc_col="cycle")
    else:
        print("MIT CSV files not found. Run generate_csvs.py first.")

    # HUST CSV 처리 (전체 처리 시 시간이 오래 걸릴 수 있으므로 주의)
    hust_csvs = glob.glob("HUST_data/data/*.csv")
    if hust_csvs:
        print(
            f"Found {len(hust_csvs)} HUST CSVs. Analyzing first 3 cells for demonstration..."
        )
        # 메모리 효율을 위해 일부만 샘플링 (전체를 원하면 hust_csvs[:3] -> hust_csvs 로 변경)
        hust_dfs = [
            pd.read_csv(f, encoding="utf-8", encoding_errors="replace")
            for f in hust_csvs
        ]
        hust_combined = pd.concat(hust_dfs, ignore_index=True)
        # HUST는 'Cycle number'가 사이클 정보를 담고 있습니다.
        analyze_outliers(
            hust_combined, "HUST", cell_col="cell_id", cyc_col="Cycle number"
        )

    else:
        print("HUST CSV files not found. Run generate_csvs.py first.")
