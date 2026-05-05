import os
import sys
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

# NumPy 버전 호환성 패치
if not hasattr(np, "_core"):
    sys.modules["numpy._core"] = np.core

from src.models import get_model
from src.train import get_dataloaders, ConfigNamespace, seed_everything
from src.hyperparams import HYPERPARAMS

def evaluate_model(model, test_loader, config):
    model.eval()
    all_preds = []
    all_targets = []
    
    print(f"[Info] Starting Inference on Test Set ({len(test_loader.dataset)} samples)...")
    
    with torch.no_grad():
        for x, t, y in tqdm(test_loader, desc="Inference"):
            x, t, y = x.to(config.device), t.to(config.device), y.to(config.device)
            
            # 모델 추론
            if config.use_pi:
                # PI 모듈은 추론 시 t를 인자로 받으며 return_pde=False 설정
                preds = model(x, t=t, return_pde=False).squeeze(-1)
            else:
                preds = model(x).squeeze(-1)
                
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            
    return np.array(all_targets), np.array(all_preds)

def plot_results(targets, preds, save_path="./outputs/results/test_performance.png"):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 지표 계산
    rmse = np.sqrt(mean_squared_error(targets, preds))
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)
    mape = np.mean(np.abs((targets - preds) / (targets + 1e-6))) * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 1. Actual vs Predicted Scatter Plot
    sns.scatterplot(x=targets, y=preds, alpha=0.4, ax=axes[0], color='#1f77b4')
    axes[0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
    axes[0].set_title(f"Actual vs Predicted Capacity\n(RMSE: {rmse:.4f}, R2: {r2:.4f})", fontsize=14)
    axes[0].set_xlabel("Actual Capacity (Ah)")
    axes[0].set_ylabel("Predicted Capacity (Ah)")
    axes[0].grid(True, alpha=0.3)

    # 2. Error Distribution (Residuals)
    errors = targets - preds
    sns.histplot(errors, bins=50, kde=True, ax=axes[1], color='#2ca02c')
    axes[1].axvline(0, color='red', linestyle='--')
    axes[1].set_title(f"Error Distribution\n(MAE: {mae:.4f}, MAPE: {mape:.2f}%)", fontsize=14)
    axes[1].set_xlabel("Error (Actual - Predicted)")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    
    print(f"\n[Final Metrics]")
    print(f" - RMSE: {rmse:.6f} Ah")
    print(f" - MAE : {mae:.6f} Ah")
    print(f" - R2  : {r2:.6f}")
    print(f" - MAPE: {mape:.2f} %")
    print(f"\nResult plot saved to: {save_path}")

if __name__ == "__main__":
    # 0. 설정 로드
    config = ConfigNamespace(HYPERPARAMS)
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(config.seed)
    
    # 1. 데이터 로드 (학습 때와 동일한 seed로 분할하여 Test Set 확보)
    _, _, test_loader = get_dataloaders(config)
    
    # 2. 모델 초기화
    model_name = config.model_name
    
    # 모델 종류에 따라 kwargs로 넘겨줄 세부 파라미터 선택
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
        **specific_params
    ).to(config.device)
    
    # 3. 체크포인트 로드
    # train.py에서 저장한 규칙에 따라 경로 설정
    base_name = f"{model_name}_{'PI_' if config.use_pi else ''}combined_capacity"
    checkpoint_path = Path("./outputs/checkpoints") / f"{base_name}.pth"
    
    if not checkpoint_path.exists():
        print(f"[Error] Checkpoint not found at: {checkpoint_path}")
        sys.exit(1)
        
    print(f"[Info] Loading weights from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=config.device))
    
    # 4. 평가 및 시각화
    targets, preds = evaluate_model(model, test_loader, config)
    plot_results(targets, preds)
