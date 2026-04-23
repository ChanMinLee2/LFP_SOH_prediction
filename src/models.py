import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# ==========================================
# 1. 딥러닝 모델 (PyTorch 기반)
# ==========================================


class VanillaLSTM(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=64, num_layers=2, output_dim=1):
        super(VanillaLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        last_step_out = out[:, -1, :]
        return self.fc(last_step_out)


class InvertedTransformer(nn.Module):
    def __init__(
        self,
        num_variates=40,
        seq_len=1,
        d_model=64,
        n_heads=4,
        e_layers=2,
        output_dim=1,
    ):
        super(InvertedTransformer, self).__init__()
        self.num_variates = num_variates
        self.project = nn.Linear(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_variates * d_model, 128),
            nn.GELU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        x_inv = x.transpose(1, 2)
        x_emb = self.project(x_inv)
        out = self.transformer(x_emb)
        return self.fc(out)


class SimpleMLP(nn.Module):
    def __init__(self, input_dim=40, output_dim=1):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# ==========================================
# 2. Physics-Informed (PI) Wrapper
# ==========================================


class PhysicsInformedWrapper(nn.Module):
    """
    기존 딥러닝 모델(Solution Network)을 감싸서
    물리 정보 기반 신경망(Dynamics Network) 연산을 추가하는 래퍼 클래스입니다.
    """

    def __init__(self, base_model, feature_dim=40):
        super(PhysicsInformedWrapper, self).__init__()
        self.solution_net = base_model  # F(t, x)
        self.feature_dim = feature_dim

        # Dynamics Network G(t, x, u, u_t, u_x)
        # 입력 차원 계산: t(1) + x(feature_dim) + u(1) + u_t(1) + u_x(feature_dim)
        g_input_dim = 1 + feature_dim + 1 + 1 + feature_dim
        self.dynamics_net = nn.Sequential(
            nn.Linear(g_input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),  # 예측된 열화율(decay rate) 반환
        )

    def forward(self, x, t=None, return_pde=False):
        """
        return_pde=False : 일반적인 추론(Inference) 모드. SOH만 반환.
        return_pde=True  : 훈련(Training) 모드. SOH와 PDE 잔차(Residual) 반환.
        """
        if not return_pde:
            return self.solution_net(x)

        assert (
            t is not None
        ), "PDE 계산을 위해서는 시간(Cycle) 정보 't'가 명시적으로 필요합니다."

        # 자동 미분(Autograd)을 위해 기울기 계산 활성화
        x.requires_grad_(True)
        t.requires_grad_(True)

        # 1. Solution Network 예측: u = F(x)
        u = self.solution_net(x)

        # 2. 편미분 계산 (u_x, u_t)
        u_x = torch.autograd.grad(
            outputs=u,
            inputs=x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0]

        u_t = torch.autograd.grad(
            outputs=u,
            inputs=t,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0]

        # 3. 형태 맞추기 (Flatten)
        x_flat = x.view(x.size(0), -1)  # (B, feature_dim)
        u_x_flat = u_x.view(u_x.size(0), -1)  # (B, feature_dim)

        # 4. Dynamics Network 입력 생성 및 추론
        g_input = torch.cat([t, x_flat, u, u_t, u_x_flat], dim=1)  # (B, g_input_dim)
        g_out = self.dynamics_net(g_input)

        # 5. PDE Residual 계산: H = u_t - G(t, x, u, u_t, u_x)
        pde_residual = u_t - g_out

        return u, pde_residual


# ==========================================
# 3. Machine Learning Models
# ==========================================


def get_rf_model():
    return RandomForestRegressor(
        n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
    )


def get_svr_model():
    return SVR(kernel="rbf", C=100, gamma="scale")


def get_gpr_model():
    kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1)
    return GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=5, random_state=42
    )


# ==========================================
# 4. Model Factory
# ==========================================


def get_model(model_name, use_pi=False, feature_dim=40, **kwargs):
    model_name = model_name.upper()

    # --- Deep Learning Models ---
    if model_name in ["ITRANSFORMER", "LSTM", "MLP"]:
        if model_name == "ITRANSFORMER":
            base_model = InvertedTransformer(num_variates=feature_dim, **kwargs)
        elif model_name == "LSTM":
            base_model = VanillaLSTM(input_dim=feature_dim, **kwargs)
        elif model_name == "MLP":
            base_model = SimpleMLP(input_dim=feature_dim, **kwargs)

        # PI 옵션 활성화 시 Wrapper로 감싸서 반환
        if use_pi:
            return PhysicsInformedWrapper(base_model, feature_dim=feature_dim)
        else:
            return base_model

    # --- Machine Learning Models ---
    elif model_name in ["RF", "SVR", "GPR"]:
        if use_pi:
            print(
                "Warning: Machine Learning 모델은 미분(Autograd)이 불가능하여 PI 모듈을 사용할 수 없습니다. 일반 모델을 반환합니다."
            )

        if model_name == "RF":
            return get_rf_model()
        elif model_name == "SVR":
            return get_svr_model()
        elif model_name == "GPR":
            return get_gpr_model()

    else:
        raise ValueError(f"Model '{model_name}' is not supported.")
