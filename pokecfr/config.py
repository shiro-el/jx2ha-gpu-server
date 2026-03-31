from dataclasses import dataclass


@dataclass
class EncodingConfig:
    """Infoset 인코딩 차원 설정 (Small)"""
    # 임베딩 차원
    species_dim: int = 32      # 포켓몬 종 (1000+종)
    type_dim: int = 8          # 타입 (18종)
    move_dim: int = 16         # 기술 (800+종)
    item_dim: int = 16         # 아이템 (200+종)

    # 카탈로그 크기
    num_species: int = 1025
    num_types: int = 19        # 18타입 + unknown
    num_moves: int = 920
    num_items: int = 250

    # 스칼라/이진 피처 차원
    num_status: int = 7        # 독/마비/화상/잠듦/얼음/혼란/없음
    num_stat_stages: int = 7   # 공/방/특공/특방/스핏/명중/회피
    num_pp_slots: int = 4
    num_field: int = 30        # 날씨+필드+트릭룸+진영효과

    @property
    def pokemon_dim(self) -> int:
        """포켓몬 1마리 인코딩 차원"""
        return (
            self.species_dim
            + 2 * self.type_dim
            + 4 * self.move_dim
            + self.item_dim
            + 1                    # HP%
            + self.num_status      # 상태이상 one-hot
            + self.num_stat_stages # 능력치 변화
            + self.num_pp_slots    # PP 비율
            + 1                    # 기절 여부
            + 1                    # 활성 여부
        )  # = 32 + 16 + 64 + 16 + 1 + 7 + 7 + 4 + 1 + 1 = 149

    @property
    def infoset_dim(self) -> int:
        """Attention 출력 후 최종 벡터 차원"""
        return (
            self.d_model           # 활성 포켓몬 토큰 (cross-attn 후)
            + 1                    # 상대 미공개 포켓몬 수
            + self.num_field       # 필드 상태
        )

    # Attention 설정
    d_model: int = 128         # 포켓몬 벡터 → 프로젝션 차원
    num_heads: int = 4
    num_attn_layers: int = 2


@dataclass
class NetworkConfig:
    """네트워크 아키텍처 설정"""
    hidden_dim: int = 256
    num_layers: int = 3
    num_actions: int = 9       # 기술 4 + 교체 5
    dropout: float = 0.0


@dataclass
class TrainingConfig:
    """Deep CFR 학습 설정"""
    # CFR 파라미터
    num_cfr_iterations: int = 1000
    num_traversals_per_iter: int = 10000
    advantage_buffer_size: int = 2_000_000
    strategy_buffer_size: int = 2_000_000
    max_depth: int = 40            # 턴 단위 depth limit, 초과 시 value net 평가

    # 네트워크 학습
    learning_rate: float = 1e-3
    batch_size: int = 2048
    num_epochs_per_iter: int = 2

    # 인프라
    device: str = "cuda"
    seed: int = 42
