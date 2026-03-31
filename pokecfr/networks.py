import torch
import torch.nn as nn

from pokecfr.config import EncodingConfig, NetworkConfig
from pokecfr.encoding import InfosetEncoder


def _build_mlp(input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> nn.Sequential:
    """num_layers개의 은닉층을 가진 MLP 생성"""
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for _ in range(num_layers):
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.ReLU())
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class AdvantageNetwork(nn.Module):
    """
    V(I, a | θ) — 각 행동의 누적 counterfactual regret 근사

    입력: infoset 원본 데이터
    출력: (B, num_actions) regret 값 (부호 무관, regret matching에서 clamp)
    """

    def __init__(self, enc_cfg: EncodingConfig, net_cfg: NetworkConfig):
        super().__init__()
        self.encoder = InfosetEncoder(enc_cfg)
        self.head = _build_mlp(
            input_dim=enc_cfg.infoset_dim,
            hidden_dim=net_cfg.hidden_dim,
            output_dim=net_cfg.num_actions,
            num_layers=net_cfg.num_layers,
        )

    def forward(self, infoset: dict) -> torch.Tensor:
        """Returns: (B, num_actions)"""
        z = self.encoder(**infoset)
        return self.head(z)


class StrategyNetwork(nn.Module):
    """
    Π(I | φ) — average strategy 근사

    입력: infoset 원본 데이터
    출력: (B, num_actions) 확률 분포
    """

    def __init__(self, enc_cfg: EncodingConfig, net_cfg: NetworkConfig):
        super().__init__()
        self.encoder = InfosetEncoder(enc_cfg)
        self.head = _build_mlp(
            input_dim=enc_cfg.infoset_dim,
            hidden_dim=net_cfg.hidden_dim,
            output_dim=net_cfg.num_actions,
            num_layers=net_cfg.num_layers,
        )

    def forward(self, infoset: dict, action_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            infoset: InfosetEncoder에 전달할 dict
            action_mask: (B, num_actions) 유효 행동=1, 불가=0

        Returns: (B, num_actions) 유효 행동에 대한 확률 분포
        """
        z = self.encoder(**infoset)
        logits = self.head(z)
        # 불가능한 행동은 -inf로 마스킹
        logits = logits.masked_fill(action_mask == 0, float("-inf"))
        return torch.softmax(logits, dim=-1)


class ValueNetwork(nn.Module):
    """
    V(I | ψ) — depth limit 도달 시 상태 가치 평가

    입력: infoset
    출력: 스칼라 [-1, +1] (승리 확률 근사)
    """

    def __init__(self, enc_cfg: EncodingConfig, net_cfg: NetworkConfig):
        super().__init__()
        self.encoder = InfosetEncoder(enc_cfg)
        self.head = nn.Sequential(
            *_build_mlp(
                input_dim=enc_cfg.infoset_dim,
                hidden_dim=net_cfg.hidden_dim,
                output_dim=1,
                num_layers=net_cfg.num_layers,
            ),
            nn.Tanh(),
        )

    def forward(self, infoset: dict) -> torch.Tensor:
        """Returns: (B, 1) value in [-1, +1]"""
        z = self.encoder(**infoset)
        return self.head(z)
