# ==========================================
# hyperparams.py
# ==========================================

HYPERPARAMS = {
    # 1. Experiment & Tracking
    "testcase_number": 1,  # 실험 번호 (자동 저장을 위한 ID)
    "seed": 42,
    "save_root": "../experiments/parameter_log",  # 저장할 기본 경로 (원하시는 경로로 수정)
    # 2. Data Options
    "dataset_types": ["mit", "hust"],
    "val_ratio": 0.2,
    "test_ratio": 0.2,
    "input_dim": 45,
    "target_col": "capacity",
    "add_seq_dim": False,  # MLP: False / LSTM, iTransformer: True
    # 3. Model General
    "model_name": "MLP",  # "LSTM", "ITRANSFORMER", "MLP", "RF", "SVR", "GPR"
    "output_dim": 1,
    # 4. Physics-Informed (PI) Options
    "use_pi": True,  # PI 모듈 활성화 여부
    "alpha": 0.1,  # PDE Loss 가중치
    "beta": 0.1,  # Monotonicity Loss 가중치 (필요 시)
    # 5. Deep Learning Training Parameters
    "batch_size": 512,
    "epochs": 300,
    "learning_rate": 5e-4,
    "weight_decay": 1e-4,
    "patience": 40,
    "factor": 0.5,
    "min_lr": 1e-7,
    # 6. Model-Specific Parameters (해당하는 모델의 파라미터만 자동으로 전달됩니다)
    "mlp_params": {"hidden_dims": [128, 64], "dropout": 0.2},
    "lstm_params": {"hidden_dim": 64, "num_layers": 2},
    "itransformer_params": {"seq_len": 1, "d_model": 64, "n_heads": 4, "e_layers": 2},
    "rf_params": {"n_estimators": 200, "max_depth": 15},
    "svr_params": {"C": 100, "gamma": "scale", "kernel": "rbf"},
    "gpr_params": {"length_scale": 1.0, "noise_level": 1.0},
}
