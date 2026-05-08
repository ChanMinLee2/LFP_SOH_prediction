import os
import sys
import time
from pathlib import Path
import pickle
import copy

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
try:
    from thop import profile
except ImportError:
    profile = None

# NumPy 버전 호환성 패치
if not hasattr(np, "_core"):
    sys.modules["numpy._core"] = np.core

from src.models import get_model, PhysicsInformedWrapper
from src.train import get_dataloaders, ConfigNamespace, seed_everything
from src.hyperparams import HYPERPARAMS, MODEL_MAP


# ==========================================
# 1. Inference Engine
# ==========================================
def evaluate_model(model, test_loader, config, is_dl=True):
    results = {"targets": [], "preds": [], "time": [], "cell_id": [], "scenario": []}
    total_inference_time = 0.0
    print(f"[Info] Starting Inference on Test Set ({len(test_loader.dataset)} samples)...")

    if is_dl:
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Inference")):
                # x, t, m, y 반환 (BatterySOHDataset 참조)
                x = batch[0].to(config.device)
                t = batch[1].to(config.device)
                m = batch[2].to(config.device)
                y = batch[3].to(config.device)

                if getattr(config, "add_seq_dim", False) and len(x.shape) == 2:
                    x = x.unsqueeze(1)

                meta_cell = np.ones(y.size(0)) * batch_idx
                meta_scen = ["default"] * y.size(0)

                start_t = time.perf_counter()

                if config.use_pi:
                    preds = model(x, t=t, mode=m, return_pde=False).squeeze(-1)
                else:
                    if isinstance(model, PhysicsInformedWrapper):
                        preds = model(x, mode=m).squeeze(-1)
                    else:
                        preds = model(x).squeeze(-1)

                batch_time = time.perf_counter() - start_t
                total_inference_time += batch_time

                results["targets"].extend(y.cpu().numpy())
                results["preds"].extend(preds.cpu().numpy())
                results["time"].extend(t.cpu().numpy().flatten())
                results["cell_id"].extend(meta_cell)
                results["scenario"].extend(meta_scen)
    else:
        # ML 모델 추론
        X_test = test_loader.dataset.x.numpy()
        y_test = test_loader.dataset.y.numpy()
        t_test = test_loader.dataset.t.numpy()

        if len(X_test.shape) == 3:
            X_test = X_test.reshape(X_test.shape[0], -1)

        start_t = time.perf_counter()
        preds = model.predict(X_test)
        total_inference_time = time.perf_counter() - start_t

        results["targets"].extend(y_test)
        results["preds"].extend(preds)
        results["time"].extend(t_test.flatten())
        results["cell_id"].extend(np.ones(len(y_test)))
        results["scenario"].extend(["default"] * len(y_test))

    avg_inference_time_ms = (total_inference_time / len(test_loader.dataset)) * 1000
    return pd.DataFrame(results), avg_inference_time_ms


# ==========================================
# 2. 성능 지표 계산 및 시각화
# ==========================================
def calculate_and_plot_metrics(df_results, model, config, avg_inf_time_ms, is_dl=True):
    save_dir = config.exp_dir / "evaluation"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    targets = df_results["targets"].values
    preds = df_results["preds"].values

    metrics = {}

    # [A] 기본 회귀 지표
    metrics["1. MAE (%)"] = mean_absolute_error(targets, preds)
    metrics["2. RMSE (%)"] = np.sqrt(mean_squared_error(targets, preds))
    metrics["3. MAPE (%)"] = np.mean(np.abs((targets - preds) / (targets + 1e-6))) * 100
    metrics["4. R2 Score"] = r2_score(targets, preds)
    metrics["5. Max Error (%)"] = np.max(np.abs(targets - preds))

    # [B] 불확실성 지표
    residuals = targets - preds
    std_res = np.std(residuals)
    ci_upper = preds + 1.96 * std_res
    ci_lower = preds - 1.96 * std_res

    metrics["6. 95% CI Range (%)"] = np.mean(ci_upper - ci_lower)
    within_ci = np.logical_and(targets >= ci_lower, targets <= ci_upper)
    metrics["7. PICP (%)"] = np.mean(within_ci) * 100

    # [C] 단조 감소성 (Trend Consistency)
    monotonic_scores = []
    for cell, group in df_results.groupby("cell_id"):
        group = group.sort_values("time")
        diffs = np.diff(group["preds"].values)
        mono_score = np.sum(diffs <= 0.001) / len(diffs) if len(diffs) > 0 else 1.0
        monotonic_scores.append(mono_score)
    metrics["8. Trend Consistency"] = np.mean(monotonic_scores)

    # [D] Inference Time & Complexity
    metrics["9. Inference Time (ms/sample)"] = avg_inf_time_ms
    
    if is_dl:
        metrics["10. Parameter Count"] = sum(p.numel() for p in model.parameters())
        if profile is not None:
            try:
                dummy_x = torch.randn(1, config.input_dim).to(config.device)
                if getattr(config, "add_seq_dim", False):
                    dummy_x = dummy_x.unsqueeze(1)
                
                if config.use_pi:
                    macs, _ = profile(model, inputs=(dummy_x, torch.randn(1, 1).to(config.device), torch.randn(1, 1).to(config.device)), verbose=False)
                else:
                    if isinstance(model, PhysicsInformedWrapper):
                        macs, _ = profile(model, inputs=(dummy_x, None, torch.randn(1, 1).to(config.device)), verbose=False)
                    else:
                        macs, _ = profile(model, inputs=(dummy_x,), verbose=False)
                metrics["11. FLOPs"] = macs * 2
            except:
                metrics["11. FLOPs"] = "Profiler Error"
        else:
            metrics["11. FLOPs"] = "thop not installed"
    else:
        metrics["10. Parameter Count"] = "N/A (ML Model)"
        metrics["11. FLOPs"] = "N/A (ML Model)"

    # 콘솔 출력
    print("\n" + "=" * 50)
    print(f"[{config.model_name}] Performance Metrics Summary")
    print("=" * 50)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:<35}: {v:.6f}")
        else:
            print(f"{k:<35}: {v}")
    print("=" * 50)

    # 결과 리포트 저장
    with open(save_dir / f"{config.model_name}_metrics_report.txt", "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    # 시각화 1: 예측 성능
    sns.set_theme(style="whitegrid")
    fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6))
    sns.scatterplot(x=targets, y=preds, alpha=0.3, ax=axes1[0], color="#1f77b4")
    axes1[0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], "r--", lw=2)
    axes1[0].set_title(f"Actual vs Predicted (RMSE: {metrics['2. RMSE (%)']:.4f})")
    axes1[0].set_xlabel("Actual SOH")
    axes1[0].set_ylabel("Predicted SOH")

    sns.histplot(residuals, bins=50, kde=True, ax=axes1[1], color="#2ca02c")
    axes1[1].axvline(0, color="red", linestyle="--")
    axes1[1].set_title(f"Error Distribution")
    axes1[1].set_xlabel("Error (Actual - Predicted)")
    plt.tight_layout()
    fig1.savefig(save_dir / f"{config.model_name}_1_Regression.png", dpi=300)
    plt.close(fig1)

    print(f"[Info] Evaluation completed. Results saved to '{save_dir}'.")


# ==========================================
# 3. Main Execution
# ==========================================
if __name__ == "__main__":
    MODELS_TO_EVALUATE = ["MLP", "TABNET", "ITRANSFORMER", "XGBOOST", "LIGHTGBM", "RF", "SVR", "GPR"]
    
    seed_everything(HYPERPARAMS["seed"])
    base_config = ConfigNamespace(HYPERPARAMS)
    base_config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {base_config.device}")
    print(f"Loading Test DataLoader...")
    _, _, test_loader = get_dataloaders(base_config)

    for model_name in MODELS_TO_EVALUATE:
        model_name_upper = model_name.upper()
        print(f"\n" + "="*50)
        print(f"Evaluating Model: {model_name_upper}")
        print("="*50)

        current_params = copy.deepcopy(HYPERPARAMS)
        current_params["model_name"] = model_name_upper
        is_dl = model_name_upper in ["MLP", "TABNET", "ITRANSFORMER"]
        current_params["add_seq_dim"] = model_name_upper in ["ITRANSFORMER"]

        cfg = ConfigNamespace(current_params)
        cfg.device = base_config.device
        
        # 모델의 버전에 맞는 디렉토리 설정
        exp_dir = cfg.setup_experiment_dir()
        
        # DL 모델 로드
        if is_dl:
            specific_params = getattr(cfg, f"{model_name.lower()}_params", {})
            model = get_model(
                model_name_upper,
                use_pi=cfg.use_pi,
                feature_dim=cfg.input_dim,
                output_dim=cfg.output_dim,
                **specific_params,
            )
            
            if not cfg.checkpoint_path.exists():
                print(f"[Warning] Checkpoint not found: {cfg.checkpoint_path}. Skipping.")
                continue
                
            print(f"[Info] Loading weights from: {cfg.checkpoint_path}")
            model.load_state_dict(torch.load(cfg.checkpoint_path, map_location=cfg.device))
            model = model.to(cfg.device)
            
            df_results, avg_inf_time = evaluate_model(model, test_loader, cfg, is_dl=True)
            calculate_and_plot_metrics(df_results, model, cfg, avg_inf_time, is_dl=True)
            
        # ML 모델 로드
        else:
            model_path = cfg.checkpoint_path.with_suffix(".pkl")
            if not model_path.exists():
                print(f"[Warning] ML Model not found: {model_path}. Skipping.")
                continue
                
            print(f"[Info] Loading model from: {model_path}")
            with open(model_path, "rb") as f:
                model = pickle.load(f)
                
            df_results, avg_inf_time = evaluate_model(model, test_loader, cfg, is_dl=False)
            calculate_and_plot_metrics(df_results, model, cfg, avg_inf_time, is_dl=False)
