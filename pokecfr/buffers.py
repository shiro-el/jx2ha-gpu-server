"""Reservoir sampling 기반 메모리 버퍼.

Deep CFR은 traversal 중 수집한 (state, target) 쌍을 버퍼에 저장하고,
iteration마다 네트워크를 재학습한다. 버퍼 크기를 초과하면 reservoir sampling으로
균일하게 교체.

상태는 poke-engine 직렬화 문자열로 저장 → 학습 시 역직렬화하여
encoder + head 전체 네트워크를 학습.
"""

import random

import torch


class ReservoirBuffer:
    """고정 크기 reservoir sampling 버퍼."""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer: list[dict] = []
        self.total_seen = 0

    def add(self, sample: dict) -> None:
        self.total_seen += 1
        if len(self.buffer) < self.max_size:
            self.buffer.append(sample)
        else:
            idx = random.randint(0, self.total_seen - 1)
            if idx < self.max_size:
                self.buffer[idx] = sample

    def sample_batch(self, batch_size: int) -> list[dict]:
        """랜덤 미니배치 추출 (raw dict 리스트)."""
        indices = random.sample(range(len(self.buffer)), min(batch_size, len(self.buffer)))
        return [self.buffer[i] for i in indices]

    def clear(self) -> None:
        self.buffer.clear()
        self.total_seen = 0

    def __len__(self) -> int:
        return len(self.buffer)


class AdvantageBuffer(ReservoirBuffer):
    """Advantage memory: (state_str, player, turn, regret, action_mask, iteration)"""

    def store(
        self,
        state_str: str,               # poke-engine 직렬화 상태
        player: int,                   # 관점 플레이어
        turn: int,                     # 현재 턴
        regret: torch.Tensor,          # (num_actions,) counterfactual regret
        action_mask: torch.Tensor,     # (num_actions,)
        iteration: int,
    ) -> None:
        self.add({
            "state_str": state_str,
            "player": player,
            "turn": turn,
            "regret": regret.detach().cpu(),
            "action_mask": action_mask.detach().cpu(),
            "iteration": iteration,
        })


class StrategyBuffer(ReservoirBuffer):
    """Strategy memory: (state_str, player, turn, strategy, action_mask, iteration, reach_prob)"""

    def store(
        self,
        state_str: str,
        player: int,
        turn: int,
        strategy: torch.Tensor,       # (num_actions,) regret matching 결과
        action_mask: torch.Tensor,     # (num_actions,)
        iteration: int,
        reach_prob: float,
    ) -> None:
        self.add({
            "state_str": state_str,
            "player": player,
            "turn": turn,
            "strategy": strategy.detach().cpu(),
            "action_mask": action_mask.detach().cpu(),
            "iteration": iteration,
            "reach_prob": reach_prob,
        })
