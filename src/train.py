import os
import sys
from pathlib import Path

# 1. 'src' 모듈을 찾을 수 있도록 프로젝트 루트를 경로에 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import copy
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import datetime
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm

# NumPy 버전 호환성 패치 (pickle load 오류 방지)
if not hasattr(np, "_core"):
    sys.modules["numpy._core"] = np.core

from src.models import get_model
from hyperparams import HYPERPARAMS


# 딕셔너리를 객체(Object)처럼 점(.)으로 접근하기 위한 래퍼 클래스
class ConfigNamespace:
    def __init__(self, d):
        self.__dict__.update(d)


# config 객체 생성
config = ConfigNamespace(HYPERPARAMS)
config.save_dir = Path("./outputs/checkpoints")
config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 1. Utilities (Seeding)
# ==========================================


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==========================================
# 2. Dataset & DataLoader (Data Loading)
# ==========================================
class BatterySOHDataset(Dataset):
    def __init__(self, data_list, target_col="capacity", add_seq_dim=False):
        # x: (N, 45)
        self.x = torch.tensor(
            np.array([item["x"] for item in data_list]), dtype=torch.float32
        )

        # [추가] t: (N, 1) - PI 모듈의 편미분을 위한 시간(Cycle) 데이터 추출
        # 데이터셋 딕셔너리에 'cyc' 키가 있다고 가정합니다. 없다면 인덱스(i)를 임시로 사용.
        self.t = torch.tensor(
            np.array([item.get("cyc", i) for i, item in enumerate(data_list)]),
            dtype=torch.float32,
        ).unsqueeze(-1)

        # y: (N,)
        self.y = torch.tensor(
            np.array([item[target_col] for item in data_list]), dtype=torch.float32
        )

        if add_seq_dim:
            self.x = self.x.unsqueeze(1)  # (N, 45) -> (N, 1, 45)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # [수정] DataLoader가 x(피처), t(시간), y(라벨) 3가지를 반환하도록 변경
        return self.x[idx], self.t[idx], self.y[idx]


def get_dataloaders(config):
    full_pool = []
    root_path = Path(config.processed_data_root)
    print(f"[Info] Searching for data in: {root_path}")
    for d_type in config.dataset_types:
        data_path = root_path / f"{d_type}_optimized_tensors.pkl"
        if not data_path.exists():
            print(f"[Warning] Data not found for {d_type} at {data_path}")
            continue

        print(f"[Info] Loading {d_type} data... (this may take a minute)")
        try:
            with open(data_path, "rb") as f:
                pool = pickle.load(f)
                # 데이터 출처 표시
                for item in pool:
                    item["source"] = d_type
                full_pool.extend(pool)
                print(f"[Success] Loaded {len(pool)} samples from {d_type}")
        except Exception as e:
            print(f"[Error] Failed to load {d_type}: {e}")

    if not full_pool:
        print(
            "[Critical] No data was loaded. Please check if the .pkl files exist in D:/ drive."
        )
        raise FileNotFoundError("No data files found to load.")

    # Cell-wise Split to prevent data leakage
    print("[Info] Splitting data into Train/Val/Test (6:2:2)...")
    unique_cells = list(set([(item["source"], item["cell"]) for item in full_pool]))
    np.random.shuffle(unique_cells)

    # 6:2:2 split
    n_cells = len(unique_cells)
    test_idx = int(n_cells * (1 - config.test_ratio))
    val_idx = int(test_idx * (1 - config.val_ratio / (1 - config.test_ratio)))

    train_cells = set(unique_cells[:val_idx])
    val_cells = set(unique_cells[val_idx:test_idx])
    test_cells = set(unique_cells[test_idx:])

    train_data = [
        item for item in full_pool if (item["source"], item["cell"]) in train_cells
    ]
    val_data = [
        item for item in full_pool if (item["source"], item["cell"]) in val_cells
    ]
    test_data = [
        item for item in full_pool if (item["source"], item["cell"]) in test_cells
    ]

    print(f"Dataset Split (6:2:2) Summary:")
    print(f"  Train: {len(train_cells)} cells, {len(train_data)} samples")
    print(f"  Val  : {len(val_cells)} cells, {len(val_data)} samples")
    print(f"  Test : {len(test_cells)} cells, {len(test_data)} samples")

    train_loader = DataLoader(
        BatterySOHDataset(train_data, config.target_col, config.add_seq_dim),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        BatterySOHDataset(val_data, config.target_col, config.add_seq_dim),
        batch_size=config.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        BatterySOHDataset(test_data, config.target_col, config.add_seq_dim),
        batch_size=config.batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader


# ==========================================
# 3. Utilities (Early Stopping & Engine)
# ==========================================
class EarlyStopping:
    def __init__(self, patience=20, verbose=False, path="checkpoint.pth"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def save_hyperparameters(config_dict):
    """
    하이퍼파라미터를 규칙에 맞는 파일명으로 저장합니다.
    파일명 형식: {testcase_number}_{MM-DD-HH}_{params}.json
    """
    save_root = Path(config_dict["save_root"])
    save_root.mkdir(parents=True, exist_ok=True)

    tc_num = config_dict["testcase_number"]
    now_str = datetime.datetime.now().strftime("%m-%d-%H")  # MM-DD-HH

    # params 요약 문자열 만들기 (예: MLP_PI-O_lr0.0005)
    # model_n = config_dict["model_name"]
    # pi_stat = "PI-O" if config_dict["use_pi"] else "PI-X"
    # lr_stat = f"lr{config_dict['learning_rate']}"
    # params_str = f"{model_n}_{pi_stat}_{lr_stat}"

    # 최종 파일명
    file_name = f"{tc_num:03d}_{now_str}.json"
    file_path = save_root / file_name

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False)

    print(
        f"\n[Info] 하이퍼파라미터 셋이 다음 경로에 저장되었습니다:\n -> {file_path}\n"
    )
    return file_path


def train_epoch(model, dataloader, criterion, optimizer, config):
    model.train()
    running_loss = 0.0
    for x, t, y in tqdm(dataloader, desc="  Training Batch", leave=False):  # [수정] tqdm 추가
        x, t, y = x.to(config.device), t.to(config.device), y.to(config.device)
        optimizer.zero_grad()

        # PI 옵션에 따른 분기 처리
        if config.use_pi:
            # 1. PI 모듈 추론: SOH 예측값과 PDE 잔차 반환
            preds, pde_residual = model(x, t=t, return_pde=True)
            preds = preds.squeeze(-1)

            # 2. PINN Loss 계산
            loss_data = criterion(preds, y)  # L_data (일반 예측 오차)
            loss_pde = torch.mean(pde_residual**2)  # L_pde (동역학 제약 위배 페널티)

            # 3. Total Loss 계산 (단조성 Loss는 배치 내 순서가 뒤섞여 있으므로 생략하거나 정렬 후 사용)
            loss = loss_data + (config.alpha * loss_pde)
        else:
            # PI 비활성화 시 기본 학습 방식
            preds = model(x).squeeze(-1)
            loss = criterion(preds, y)

        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)

    return running_loss / len(dataloader.dataset)


def validate_epoch(model, dataloader, criterion, config):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for x, t, y in dataloader:  # [수정] t(시간) 언패킹
            x, t, y = x.to(config.device), t.to(config.device), y.to(config.device)

            # [수정] 평가 시에도 PI Wrapper는 t를 필요로 함
            if config.use_pi:
                preds = model(x, t=t, return_pde=False).squeeze(-1)
            else:
                preds = model(x).squeeze(-1)

            loss = criterion(preds, y)
            running_loss += loss.item() * x.size(0)
    return running_loss / len(dataloader.dataset)


# ==========================================
# 4. Training Engine
# ==========================================
def fit(model, train_loader, val_loader, config, model_name="BestModel"):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.factor,
        patience=15,
        min_lr=config.min_lr,
        verbose=True,
    )

    config.save_dir.mkdir(parents=True, exist_ok=True)
    # 저장 이름에 PI 사용 여부 명시
    base_name = f"{model_name}_{'PI_' if config.use_pi else ''}combined_capacity"
    checkpoint_path = config.save_dir / f"{base_name}.pth"

    # EarlyStopping 설정: 가장 낮은 Val Loss를 가진 모델을 checkpoint_path에 저장
    early_stopping = EarlyStopping(patience=config.patience, verbose=True, path=checkpoint_path)
    history = {"train_loss": [], "val_loss": []}

    print(
        f"--- Training Start (Model: {model_name} | PI Enabled: {config.use_pi} | Target: {config.target_col}) ---"
    )
    for epoch in range(1, config.epochs + 1):
        # [수정] train_epoch에 device 대신 config 전체를 넘기도록 변경 (내부에서 use_pi 판단)
        train_loss = train_epoch(model, train_loader, criterion, optimizer, config)
        val_loss = validate_epoch(model, val_loader, criterion, config)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        scheduler.step(val_loss)
        early_stopping(val_loss, model)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch [{epoch:03d}/{config.epochs}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
            )

        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    # 학습 종료 후 가장 성능이 좋았던 모델 가중치 로드
    if checkpoint_path.exists():
        print(f"[Info] Loading best model weights from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path))
        
    return model, history


# ==========================================
# 5. Main Execution (train.py 하단)
# ==========================================
if __name__ == "__main__":
    # 0. 하이퍼파라미터 저장 (지정된 네이밍 규칙 적용)
    save_hyperparameters(HYPERPARAMS)

    seed_everything(config.seed)

    # 1. Load Data
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # [추가] 데이터 형태 및 피처 정보 출력
    x_sample, t_sample, y_sample = next(iter(train_loader))
    print(f"\n[Data Check] Input x shape: {x_sample.shape}")
    print(f"[Data Check] Time t shape: {t_sample.shape}")
    print(f"[Data Check] Target y shape: {y_sample.shape}")
    # 첫 번째 샘플의 앞쪽 5개 피처 값 확인
    sample_feats = x_sample[0, :5] if len(x_sample.shape) == 2 else x_sample[0, 0, :5]
    print(f"[Data Check] Sample features (first 5): {sample_feats.numpy()}")

    # 2. Select Model (설정 파일의 모델 이름과 종속 파라미터 가져오기)
    model_name = config.model_name

    # 모델 종류에 따라 kwargs로 넘겨줄 세부 파라미터 선택
    specific_params = {}
    if model_name == "MLP":
        specific_params = config.mlp_params
    elif model_name == "LSTM":
        specific_params = config.lstm_params
    elif model_name == "ITRANSFORMER":
        specific_params = config.itransformer_params

    # get_model 호출 시 설정된 특정 파라미터들을 언패킹(**)하여 전달
    model = get_model(
        model_name,
        use_pi=config.use_pi,
        feature_dim=config.input_dim,
        output_dim=config.output_dim,
        **specific_params,
    ).to(config.device)

    # 3. Train
    best_model, hist = fit(
        model, train_loader, val_loader, config, model_name=model_name
    )

    # 4. Final Evaluation
    test_loss = validate_epoch(best_model, test_loader, nn.MSELoss(), config)
    print(f"\n[Final Results] Test MSE: {test_loss:.6f}")
    print(f"[Final Results] Test RMSE: {np.sqrt(test_loss):.6f}")
