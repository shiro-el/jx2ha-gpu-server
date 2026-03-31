"""포켓몬 배틀 게임 상태 추상 인터페이스.

poke-env 연동 시 이 인터페이스를 구현한다.
"""

from abc import ABC, abstractmethod
from enum import IntEnum

import torch


class NodeType(IntEnum):
    PLAYER_1 = 0    # P1 행동 선택
    PLAYER_2 = 1    # P2 행동 선택
    CHANCE = 2       # 확률 이벤트 (명중, 급소, 추가효과 등)
    TERMINAL = 3     # 게임 종료


class GameState(ABC):
    """포켓몬 배틀 한 상태를 나타내는 추상 클래스.

    한 턴 전개:
        P1 행동 선택 (PLAYER_1)
        → P2 행동 선택 (PLAYER_2, P1 행동 모름)
        → 스피드 비교 + 선공 실행 (CHANCE 노드들)
        → 후공 실행 (CHANCE 노드들)
        → 턴 종료 효과
        → 다음 턴 P1 행동 선택 ...
    """

    @abstractmethod
    def node_type(self) -> NodeType:
        """현재 노드 타입"""
        ...

    @abstractmethod
    def current_player(self) -> int:
        """현재 행동할 플레이어 (0 또는 1). CHANCE/TERMINAL이면 -1."""
        ...

    @abstractmethod
    def legal_actions(self) -> list[int]:
        """현재 플레이어의 유효 행동 인덱스 리스트.

        0~3: 기술 (PP 잔여 + 사용 가능한 것만)
        4~8: 교체 (기절하지 않은 벤치 포켓몬만)
        """
        ...

    @abstractmethod
    def get_infoset(self, player: int) -> dict:
        """해당 플레이어 관점의 infoset 텐서 dict.

        InfosetEncoder.forward()에 바로 전달 가능한 형태:
        {
            'my_team': {...},  'my_mask': Tensor,
            'opp_team': {...}, 'opp_mask': Tensor,
            'opp_unrevealed': Tensor,
            'field': {...},
        }
        """
        ...

    @abstractmethod
    def get_action_mask(self, player: int) -> torch.Tensor:
        """(num_actions,) 유효 행동=1, 불가=0"""
        ...

    @abstractmethod
    def apply_action(self, action: int) -> "GameState":
        """행동 적용 후 새 상태 반환 (immutable)."""
        ...

    @abstractmethod
    def sample_chance(self) -> "GameState":
        """chance 노드에서 확률에 따라 결과 하나 샘플링, 새 상태 반환."""
        ...

    @abstractmethod
    def utility(self, player: int) -> float:
        """terminal 노드에서 해당 플레이어의 유틸리티. 승리=+1, 패배=-1."""
        ...

    @abstractmethod
    def turn_number(self) -> int:
        """현재 턴 번호 (depth limit 판단용)."""
        ...
