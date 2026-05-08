# ==========================================
# hyperparams.py
# ==========================================

# 모델별 마이너 번호 매핑
MODEL_MAP = {"MLP": 1, "LSTM": 2, "ITRANSFORMER": 3, "RF": 4, "SVR": 5, "GPR": 6}

HYPERPARAMS = {
    # 1. Experiment & Tracking
    "major_version": 1,  # Parameter Set Index
    "patch_version": 1,  # Fixed for now
    "seed": 42,
    # 2. Data Options
    "dataset_types": ["mit"],
    "processed_data_root": "D:/chanminLee/data_store/LFP_SOH_estimation",
    "val_ratio": 0.2,
    "test_ratio": 0.2,
    "input_dim": 48,
    "target_col": "capacity",
    "add_seq_dim": False,
    # 3. Model General
    "model_name": "MLP",
    "output_dim": 1,
    # 4. Physics-Informed (PI) Options
    "use_pi": True,
    "alpha": 100.0,
    "beta": 0.1,
    # 5. Deep Learning Training Parameters
    "batch_size": 512,
    "epochs": 300,
    "learning_rate": 5e-4,  # 0.0005
    "weight_decay": 1e-4,
    "patience": 8,
    "factor": 0.5,
    "min_lr": 1e-7,
    # 6. Model-Specific Parameters
    "mlp_params": {"hidden_dims": [128, 64], "dropout": 0.2},
    "lstm_params": {"hidden_dim": 64, "num_layers": 2},
    "itransformer_params": {"seq_len": 1, "d_model": 64, "n_heads": 4, "e_layers": 2},
    "rf_params": {"n_estimators": 200, "max_depth": 15},
    "svr_params": {"C": 100, "gamma": "scale", "kernel": "rbf"},
    "gpr_params": {"length_scale": 1.0, "noise_level": 1.0},
}
