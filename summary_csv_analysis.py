import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from src.hyperparams import HYPERPARAMS


def load_scenario_data(output_root, dataset_type, scenario_name):
    """
    지정된 시나리오의 요약 CSV 파일을 로드합니다.
    """
    path = Path(output_root) / f"{dataset_type}_{scenario_name}_summary.csv"
    if not path.exists():
        print(f"Warning: File not found: {path}")
        return None
    return pd.read_csv(path)


def plot_feature_trends(
    df, scenario_name, features_to_plot, cell_ids=None, save_dir=None
):
    """
    선택한 피처들에 대해 사이클별 변화 추이를 시각화합니다.
    """
    if df is None or df.empty:
        return

    # 특정 셀만 필터링 (너무 많을 경우 대비)
    if cell_ids:
        plot_df = df[df["cell_id"].isin(cell_ids)].copy()
    else:
        # 셀이 너무 많으면 상위 5개만 샘플링하여 가독성 확보
        unique_cells = df["cell_id"].unique()
        plot_df = df[df["cell_id"].isin(unique_cells)].copy()

    num_features = len(features_to_plot)
    fig, axes = plt.subplots(
        num_features, 1, figsize=(12, 4 * num_features), sharex=True
    )

    if num_features == 1:
        axes = [axes]

    for i, feat in enumerate(features_to_plot):
        sns.lineplot(
            data=plot_df,
            x="cycle",
            y=feat,
            hue="cell_id",
            ax=axes[i],
            marker="o",
            alpha=0.7,
        )
        axes[i].set_title(f"[{scenario_name}] Trend of {feat}")
        axes[i].set_ylabel(feat)
        axes[i].grid(True, linestyle="--", alpha=0.6)
        axes[i].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.xlabel("Cycle Number")
    plt.tight_layout()

    if save_dir:
        save_path = Path(save_dir) / f"{scenario_name}_trends.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
        plt.close()


def plot_pareto_distribution(df, scenario_name, features_to_plot, save_dir=None):
    """
    각 피처별로 파레토 분포(누적 분포 함수, CDF)를 시각화합니다.
    x축: 피처 값, y축: 누적 비율 (0~1)
    """
    if df is None or df.empty:
        return

    num_features = len(features_to_plot)
    fig, axes = plt.subplots(num_features, 1, figsize=(10, 5 * num_features))

    if num_features == 1:
        axes = [axes]

    for i, feat in enumerate(features_to_plot):
        # 1. 데이터 정렬 및 누적 비율 계산
        data = df[feat].dropna().sort_values()
        if len(data) == 0:
            continue

        y_vals = np.arange(1, len(data) + 1) / len(data)

        # 2. 플랏 생성
        axes[i].plot(data, y_vals, marker=".", linestyle="none", alpha=0.5)
        axes[i].set_title(f"[{scenario_name}] Pareto (CDF) of {feat}")
        axes[i].set_xlabel(feat)
        axes[i].set_ylabel("Cumulative Proportion")
        axes[i].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    if save_dir:
        save_path = Path(save_dir) / f"{scenario_name}_pareto.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved Pareto plot to {save_path}")


def main():
    # 첫 실행 : IS_REMOVAL=False로 설정, 시나리오별로 'count' 피처의 상/하위 5% 이상치 인덱스를 식별하여 removal_index.csv에 저장
    # 이후 실행 : IS_REMOVAL=True로 설정하여 removal_index.csv에 기록된 사이클을 제거한 후 트렌드/파레토 플랏 생성
    # --- 설정 영역 ---
    DATASET_TYPE = "hust"  # "hust" 또는 "mit"
    IS_REMOVAL = True  # True일 경우 removal_index.csv에 해당하는 데이터를 제거 후 플랏

    SUMMARY_ROOT = Path(
        f"D:/chanminLee/data_store/LFP_SOH_estimation/case_{HYPERPARAMS['major_version']}/scenario_summaries"
    )
    FIGURE_DIR = Path(
        f"./outputs/figures/scenario_analysis/case_{HYPERPARAMS['major_version']}/{DATASET_TYPE}"
    )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    REMOVAL_INDEX_PATH = Path(
        f"case{HYPERPARAMS['major_version']}_{DATASET_TYPE}_removal_index.csv"
    )

    # 시나리오 리스트
    scenarios = [
        "charge-high",
        "charge-mid",
        "charge-low",
        "discharge-high",
        "discharge-mid",
        "discharge-low",
    ]

    # 분석하고 싶은 피처 리스트
    target_features = ["V_std", "count", "I_std", "T_std", "V_min", "V_max"]

    all_removal_indices = []

    # --- 실행 영역 ---
    for scenario in scenarios:
        print(f"\nAnalyzing Scenario: {scenario}")
        df = load_scenario_data(SUMMARY_ROOT, DATASET_TYPE, scenario)

        if df is not None:
            # 1. 'count' 피처 기반 상/하위 1.5% 이상치 식별
            # counts = df["count"].dropna().sort_values()

            # n = len(counts)
            # if n > 0:
            #     lower_bound = np.percentile(counts, 5)
            #     upper_bound = np.percentile(counts, 95)

            #     outliers = df[
            #         (df["count"] < lower_bound) | (df["count"] > upper_bound)
            #     ].copy()
            #     outliers["scenario"] = scenario
            #     all_removal_indices.append(
            #         outliers[["cell_id", "cycle", "scenario", "count"]]
            #     )
            #     print(
            #         f"  Identified {len(outliers)} count outliers (Limits: {lower_bound:.1f} ~ {upper_bound:.1f})"
            #     )

            # 수정1. 모든 피처 대상으로 이상치 탐지 (count뿐 아니라 V_std, I_std 등도 포함)
            for feat in target_features:
                feature_values = df[feat].dropna().sort_values()
                n = len(feature_values)
                if n > 0:
                    lower_bound = np.percentile(feature_values, 0.1)
                    upper_bound = np.percentile(feature_values, 99.9)

                    outliers = df[
                        (df[feat] < lower_bound) | (df[feat] > upper_bound)
                    ].copy()
                    outliers["scenario"] = scenario
                    outliers["feature"] = feat
                    all_removal_indices.append(
                        outliers[["cell_id", "cycle", "scenario", "feature", feat]]
                    )
                    print(
                        f"  Identified {len(outliers)} outliers in {feat} (Limits: {lower_bound:.2f} ~ {upper_bound:.2f})"
                    )

            # 2. 이상치 제거 수행 (옵션)
            if IS_REMOVAL and REMOVAL_INDEX_PATH.exists():
                removal_df = pd.read_csv(REMOVAL_INDEX_PATH)
                # 시나리오 상관없이 CSV에 존재하는 모든 (cell_id, cycle) 쌍을 제거 대상으로 취급 (Global Removal)
                unique_removals = removal_df[["cell_id", "cycle"]].drop_duplicates()

                # 병합(merge)을 이용한 차집합 연산으로 데이터 제거
                original_len = len(df)
                df = df.merge(
                    unique_removals,
                    on=["cell_id", "cycle"],
                    how="left",
                    indicator=True,
                )
                df = df[df["_merge"] == "left_only"].drop(columns=["_merge"])

                print(
                    f"  Removed {original_len - len(df)} cycles globally based on {REMOVAL_INDEX_PATH.name}"
                )

            # 3. 기본적인 결측치/이상치 통계 확인
            print(df[target_features].describe())

            # 4. 시각화 실행
            plot_feature_trends(df, scenario, target_features, save_dir=FIGURE_DIR)
            plot_pareto_distribution(df, scenario, target_features, save_dir=FIGURE_DIR)

    # 5. 모든 시나리오의 이상치 인덱스를 하나로 통합하여 저장
    if all_removal_indices:
        final_removal_df = pd.concat(all_removal_indices, ignore_index=True)
        # 중복된 cell-cycle 제거 (여러 시나리오에서 동시에 이상치일 수 있음)
        final_removal_df = final_removal_df.drop_duplicates(subset=["cell_id", "cycle"])
        final_removal_df.to_csv(REMOVAL_INDEX_PATH, index=False)
        print(
            f"\n[Success] Saved all identified removal indices to {REMOVAL_INDEX_PATH} (Total unique: {len(final_removal_df)})"
        )


# =============================================================================
# [전문가 조언: 요약 CSV 기반 이상치 제거 전략]
# =============================================================================
# 1. 'count' 기반 제거 (가장 우선순위):
#    - 특정 사이클의 데이터 포인트 개수(count)가 다른 사이클에 비해 현저히 적거나 많다면,
#      해당 사이클의 전압/전류 시계열 데이터가 잘렸거나 노이즈가 섞여 슬라이싱이 잘못된 것입니다.
#    - 예: 평균 count가 100인데 특정 사이클만 10개라면 해당 사이클은 제거 대상입니다.
#
# 2. 'std' 기반 제거 (노이즈 탐지):
#    - V_std(전압 표준편차)가 비정상적으로 높다면 해당 구간에서 전압 스파이크가 발생했음을 의미합니다.
#    - 배터리 물리 현상상 전압은 매끄럽게 변해야 하므로, std가 튀는 사이클은 학습 시 가중치를 왜곡합니다.
#
# 3. 'Mean' 트렌드 이탈 제거 (열화 경로 이탈):
#    - 사이클이 진행됨에 따라 V_mean이나 I_mean은 서서히 변해야 합니다.
#    - 갑자기 이전/이후 사이클과 비교해 값이 점프(Jump)한다면 센서 오동작일 확률이 99%입니다.
#
# 4. 물리적 임계치 (Hard Clipping):
#    - LFP 전압 범위를 벗어나는 V_min/V_max (예: 2.0V 미만 또는 3.8V 초과)가 나타나는 사이클은
#      데이터 신뢰도가 없으므로 배제하는 것이 안전합니다.
#
# 5. 구현 팁:
#    - 위 플랏에서 육안으로 확인된 이상 사이클(Cell_ID, Cycle)을 'black_list'로 만드세요.
#    - 이후 Phase 2 슬라이싱 단계에서 `if (cid, cyc) in black_list: continue` 로직을 추가하여
#      원천적으로 학습 데이터 풀에서 제외하는 것이 가장 깔끔한 방법입니다.
# =============================================================================

if __name__ == "__main__":
    main()
