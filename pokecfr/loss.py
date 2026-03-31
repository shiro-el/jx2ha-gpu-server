import torch
import torch.nn.functional as F


def advantage_loss(
    predicted: torch.Tensor,   # (B, num_actions) 네트워크 출력
    target: torch.Tensor,      # (B, num_actions) instantaneous counterfactual regret
    iteration: torch.Tensor,   # (B,) 각 샘플이 수집된 CFR iteration 번호
    action_mask: torch.Tensor, # (B, num_actions) 유효 행동=1
) -> torch.Tensor:
    """
    Advantage network loss (Brown et al., 2019 Eq. 2):
        L = Σ t · ‖V_θ(I) - r‖²

    후반 iteration 샘플에 높은 가중치 — CFR 수렴 특성상
    후반 regret이 더 정확하므로.
    """
    # 유효 행동만 loss 계산
    sq_err = F.mse_loss(predicted, target, reduction="none")  # (B, num_actions)
    sq_err = sq_err * action_mask                              # 무효 행동 제거
    per_sample = sq_err.sum(dim=-1)                            # (B,)

    # iteration 가중치
    weights = iteration.float()                                # (B,)
    weighted = weights * per_sample                            # (B,)

    return weighted.mean()


def strategy_loss(
    predicted: torch.Tensor,      # (B, num_actions) 네트워크 출력 확률
    target: torch.Tensor,         # (B, num_actions) regret matching으로 계산된 전략
    iteration: torch.Tensor,      # (B,) CFR iteration 번호
    reach_prob: torch.Tensor,     # (B,) 해당 플레이어의 reach probability
    action_mask: torch.Tensor,    # (B, num_actions) 유효 행동=1
) -> torch.Tensor:
    """
    Strategy network loss (Brown et al., 2019 Eq. 3):
        L = Σ t · p · ‖Π_φ(I) - π‖²

    t: iteration 가중치 (후반일수록 전략이 Nash에 가까움)
    p: reach probability (자주 방문하는 infoset에 높은 가중치)
    """
    sq_err = F.mse_loss(predicted, target, reduction="none")  # (B, num_actions)
    sq_err = sq_err * action_mask
    per_sample = sq_err.sum(dim=-1)                            # (B,)

    weights = iteration.float() * reach_prob.clamp(min=1e-8)    # (B,)
    weighted = weights * per_sample

    total_weight = weights.sum().clamp(min=1e-8)
    return weighted.sum() / total_weight
