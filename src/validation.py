import os
import sys
import time
from pathlib import Path

# 1. 'src' 모듈을 찾을 수 있도록 프로젝트 루트를 경로에 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
from thop import profile

# NumPy 버전 호환성 패치
if not hasattr(np, "_core"):
    sys.modules["numpy._core"] = np.core

from src.models import get_model
from src.train import get_dataloaders, ConfigNamespace, seed_everything
from src.hyperparams import HYPERPARAMS


# ==========================================
# 1. Inference Engine
# ==========================================
def evaluate_model(model, test_loader, config):
    model.eval()

    results = {"targets": [], "preds": [], "time": [], "cell_id": [], "scenario": []}

    total_inference_time = 0.0
    print(
        f"[Info] Starting Inference on Test Set ({len(test_loader.dataset)} samples)..."
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Inference")):
            # 데이터 언패킹 (데이터로더가 cell_id 등을 반환하는지 여부에 따라 유연하게 처리)
            x = batch[0].to(config.device)
            t = batch[1].to(config.device)
            y = batch[2].to(config.device)

            # 메타데이터가 있다면 저장 (없으면 임시값 생성)
            meta_cell = batch[3] if len(batch) > 3 else np.ones(y.size(0)) * batch_idx
            meta_scen = batch[4] if len(batch) > 4 else ["default"] * y.size(0)

            # 추론 시간 측정 (Metric 12)
            start_t = time.perf_counter()

            # 모델 추론
            if config.use_pi:
                preds = model(x, t=t, return_pde=False).squeeze(-1)
            else:
                preds = model(x).squeeze(-1)

            batch_time = time.perf_counter() - start_t
            total_inference_time += batch_time

            # 결과 저장
            results["targets"].extend(y.cpu().numpy())
            results["preds"].extend(preds.cpu().numpy())
            results["time"].extend(t.cpu().numpy().flatten())
            results["cell_id"].extend(meta_cell)
            results["scenario"].extend(meta_scen)

    # 전체 추론 시간을 샘플 수로 나누어 단일 샘플당 평균 추론 시간 계산
    avg_inference_time_ms = (total_inference_time / len(test_loader.dataset)) * 1000

    return pd.DataFrame(results), avg_inference_time_ms


# ==========================================
# 2. 15가지 성능 지표 계산 및 시각화
# ==========================================
def calculate_and_plot_15_metrics(
    df_results, model, config, avg_inf_time_ms, save_dir="./outputs/results"
):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    targets = df_results["targets"].values
    preds = df_results["preds"].values

    metrics = {}

    # [A] 기본 회귀 및 오차 평가 지표
    metrics["1. MAE (%)"] = mean_absolute_error(targets, preds)
    metrics["2. RMSE (%)"] = np.sqrt(mean_squared_error(targets, preds))
    metrics["3. MAPE (%)"] = np.mean(np.abs((targets - preds) / (targets + 1e-6))) * 100
    metrics["4. R2 Score"] = r2_score(targets, preds)
    metrics["5. Max Error (%)"] = np.max(np.abs(targets - preds))

    # [B] 불확실성 및 신뢰성 지표 (Empirical Method)
    residuals = targets - preds
    std_res = np.std(residuals)
    ci_upper = preds + 1.96 * std_res
    ci_lower = preds - 1.96 * std_res

    metrics["6. 95% CI Range (%)"] = np.mean(ci_upper - ci_lower)
    within_ci = np.logical_and(targets >= ci_lower, targets <= ci_upper)
    metrics["7. PICP (%)"] = np.mean(within_ci) * 100
    metrics["8. Calibration Error (ECE)"] = np.abs(
        metrics["7. PICP (%)"] - 95.0
    )  # 단순화된 경험적 ECE

    # [C] 배터리 도메인 특화 지표
    # 9. 단조 감소성 (Trend Consistency)
    monotonic_scores = []
    for cell, group in df_results.groupby("cell_id"):
        group = group.sort_values("time")
        diffs = np.diff(group["preds"].values)
        # SOH는 단조감소해야 하므로 미분값이 0 이하인 구간의 비율
        mono_score = np.sum(diffs <= 0.001) / len(diffs) if len(diffs) > 0 else 1.0
        monotonic_scores.append(mono_score)
    metrics["9. Trend Consistency (Monotonicity)"] = np.mean(monotonic_scores)

    # 10. Cell-to-Cell Variance
    cell_maes = df_results.groupby("cell_id").apply(
        lambda g: mean_absolute_error(g["targets"], g["preds"])
    )
    metrics["10. Cell-to-Cell Variance (MAE Std)"] = np.std(cell_maes)

    # 11. Scenario Robustness Error (편차)
    scen_maes = df_results.groupby("scenario").apply(
        lambda g: mean_absolute_error(g["targets"], g["preds"])
    )
    metrics["11. Scenario Robustness (MAE Std)"] = (
        np.std(scen_maes) if len(scen_maes) > 1 else 0.0
    )

    # [D] BMS 실사용 (Edge/Real-time) 지표
    metrics["12. Inference Time (ms/sample)"] = avg_inf_time_ms
    metrics["13. Parameter Count"] = sum(p.numel() for p in model.parameters())

    # 14. FLOPs (thop 라이브러리 사용)
    dummy_x = torch.randn(1, config.input_dim).to(config.device)
    if config.model_name in ["LSTM", "ITRANSFORMER"]:
        dummy_x = dummy_x.unsqueeze(1)

    if profile is not None:
        try:
            if config.use_pi:
                macs, _ = profile(
                    model,
                    inputs=(dummy_x, torch.randn(1, 1).to(config.device)),
                    verbose=False,
                )
            else:
                macs, _ = profile(model, inputs=(dummy_x,), verbose=False)
            metrics["14. FLOPs"] = macs * 2  # 1 MAC = 2 FLOPs
        except:
            metrics["14. FLOPs"] = "Error in profiler"
    else:
        metrics["14. FLOPs"] = "thop not installed"

    # [E] Feature Importance (15번)은 트리 모델(RF) 등에서 가능하므로 딥러닝은 생략하거나 추후 SHAP 추가
    metrics["15. Feature Importance"] = "Requires SHAP for DL"

    # 콘솔 출력
    print("\n" + "=" * 50)
    print(f"[{config.model_name}] 15 Performance Metrics Summary")
    print("=" * 50)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:<40}: {v:.6f}")
        else:
            print(f"{k:<40}: {v}")
    print("=" * 50)

    # ---------------------------------------------------------
    # Visualizations (다중 플랏 저장)
    # ---------------------------------------------------------
    sns.set_theme(style="whitegrid")

    # Plot 1: Regression Performance (실제 vs 예측 & 오차 분포)
    fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6))
    sns.scatterplot(x=targets, y=preds, alpha=0.3, ax=axes1[0], color="#1f77b4")
    axes1[0].plot(
        [targets.min(), targets.max()], [targets.min(), targets.max()], "r--", lw=2
    )
    axes1[0].set_title(
        f"Actual vs Predicted (RMSE: {metrics['2. RMSE (%)']:.4f})", fontsize=14
    )
    axes1[0].set_xlabel("Actual SOH")
    axes1[0].set_ylabel("Predicted SOH")

    sns.histplot(residuals, bins=50, kde=True, ax=axes1[1], color="#2ca02c")
    axes1[1].axvline(0, color="red", linestyle="--")
    axes1[1].set_title(
        f"Error Distribution (Max Error: {metrics['5. Max Error (%)']:.4f})",
        fontsize=14,
    )
    axes1[1].set_xlabel("Error (Actual - Predicted)")

    plt.tight_layout()
    fig1.savefig(
        Path(save_dir) / f"{config.model_name}_1_Basic_Regression.png", dpi=300
    )
    plt.close(fig1)

    # Plot 2: Domain-Specific (SOH Trajectory & 95% CI)
    # 임의로 첫 번째 셀을 선택하여 열화 곡선 시각화
    sample_cell = df_results["cell_id"].unique()[0]
    df_sample = df_results[df_results["cell_id"] == sample_cell].sort_values("time")

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(
        df_sample["time"],
        df_sample["targets"],
        label="Actual SOH",
        color="black",
        linewidth=2,
    )
    ax2.plot(
        df_sample["time"],
        df_sample["preds"],
        label="Predicted SOH",
        color="blue",
        linewidth=2,
    )
    ax2.fill_between(
        df_sample["time"],
        df_sample["preds"] - 1.96 * std_res,
        df_sample["preds"] + 1.96 * std_res,
        color="blue",
        alpha=0.2,
        label="95% CI (PICP: {:.1f}%)".format(metrics["7. PICP (%)"]),
    )
    ax2.set_title(
        f"SOH Degradation Trajectory & Confidence Interval (Cell: {sample_cell})\nTrend Consistency: {metrics['9. Trend Consistency (Monotonicity)']:.4f}",
        fontsize=14,
    )
    ax2.set_xlabel("Time / Cycle")
    ax2.set_ylabel("SOH")
    ax2.legend()
    plt.tight_layout()
    fig2.savefig(
        Path(save_dir) / f"{config.model_name}_2_SOH_Trajectory_CI.png", dpi=300
    )
    plt.close(fig2)

    # Plot 3: Robustness (Cell-to-Cell MAE Variance)
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    cell_maes.plot(kind="bar", color="#ff7f0e", ax=ax3)
    ax3.axhline(
        cell_maes.mean(),
        color="r",
        linestyle="--",
        label=f"Mean MAE ({cell_maes.mean():.4f})",
    )
    ax3.set_title(
        f"Cell-to-Cell Error Variance (Std: {metrics['10. Cell-to-Cell Variance (MAE Std)']:.4f})",
        fontsize=14,
    )
    ax3.set_xlabel("Cell ID")
    ax3.set_ylabel("MAE")
    ax3.legend()
    plt.tight_layout()
    fig3.savefig(Path(save_dir) / f"{config.model_name}_3_Cell_Variance.png", dpi=300)
    plt.close(fig3)

    print(f"[Info] All 15 metrics plotted and saved to '{save_dir}' successfully.")

    # 텍스트 리포트 저장
    with open(Path(save_dir) / f"{config.model_name}_15_metrics_report.txt", "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")


# ==========================================
# 3. Main Execution
# ==========================================
if __name__ == "__main__":
    # 0. 설정 로드
    config = ConfigNamespace(HYPERPARAMS)
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(config.seed)

    # 1. 데이터 로드
    _, _, test_loader = get_dataloaders(config)

    # 2. 모델 초기화
    model_name = config.model_name

    specific_params = {}
    if model_name == "MLP":
        specific_params = config.mlp_params
    elif model_name == "LSTM":
        specific_params = config.lstm_params
    elif model_name == "ITRANSFORMER":
        specific_params = config.itransformer_params

    model = get_model(
        model_name,
        use_pi=config.use_pi,
        feature_dim=config.input_dim,
        output_dim=config.output_dim,
        **specific_params,
    ).to(config.device)

    # 3. 체크포인트 로드
    base_name = f"{model_name}_{'PI_' if config.use_pi else ''}combined_capacity"
    checkpoint_path = Path("./outputs/checkpoints") / f"{base_name}.pth"

    if not checkpoint_path.exists():
        print(f"[Error] Checkpoint not found at: {checkpoint_path}")
        sys.exit(1)

    print(f"[Info] Loading weights from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=config.device))

    # 4. 추론 및 15개 지표 생성
    df_results, avg_inf_time = evaluate_model(model, test_loader, config)
    calculate_and_plot_15_metrics(df_results, model, config, avg_inf_time)
