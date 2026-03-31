import torch
import torch.nn as nn

from pokecfr.config import EncodingConfig


class PokemonEncoder(nn.Module):
    """포켓몬 1마리를 고정 길이 벡터로 인코딩"""

    def __init__(self, cfg: EncodingConfig):
        super().__init__()
        self.cfg = cfg

        # 범주형 임베딩
        self.species_emb = nn.Embedding(cfg.num_species, cfg.species_dim, padding_idx=0)
        self.type_emb = nn.Embedding(cfg.num_types, cfg.type_dim, padding_idx=0)
        self.move_emb = nn.Embedding(cfg.num_moves, cfg.move_dim, padding_idx=0)
        self.item_emb = nn.Embedding(cfg.num_items, cfg.item_dim, padding_idx=0)

    def forward(
        self,
        species: torch.Tensor,     # (B, N) 종 인덱스
        types: torch.Tensor,       # (B, N, 2) 타입1, 타입2 인덱스
        moves: torch.Tensor,       # (B, N, 4) 기술 인덱스
        item: torch.Tensor,        # (B, N) 아이템 인덱스
        hp_pct: torch.Tensor,      # (B, N, 1) HP 비율 [0,1]
        status: torch.Tensor,      # (B, N, 7) 상태이상 one-hot
        stat_stages: torch.Tensor, # (B, N, 7) 능력치 변화 [-1,+1]
        pp_pct: torch.Tensor,      # (B, N, 4) PP 비율 [0,1]
        fainted: torch.Tensor,     # (B, N, 1) 기절 여부
        active: torch.Tensor,      # (B, N, 1) 활성 여부
    ) -> torch.Tensor:
        """Returns: (B, N, pokemon_dim)"""
        emb_species = self.species_emb(species)                # (B, N, 32)
        emb_types = self.type_emb(types).flatten(-2)           # (B, N, 16)
        emb_moves = self.move_emb(moves).flatten(-2)           # (B, N, 64)
        emb_item = self.item_emb(item)                         # (B, N, 16)

        return torch.cat([
            emb_species, emb_types, emb_moves, emb_item,
            hp_pct, status, stat_stages, pp_pct, fainted, active,
        ], dim=-1)


class TeamAttentionEncoder(nn.Module):
    """
    팀 6마리를 Attention으로 인코딩.

    PokemonEncoder(149d) → Project(128d) → Self-Attention
    활성 포켓몬이 벤치/상대를 attend하여 팀 시너지 + 상성 반영.
    """

    def __init__(self, cfg: EncodingConfig):
        super().__init__()
        self.pokemon_encoder = PokemonEncoder(cfg)
        self.proj = nn.Linear(cfg.pokemon_dim, cfg.d_model)

        # 활성/벤치 구분용 학습 토큰
        self.role_emb = nn.Embedding(2, cfg.d_model)  # 0=벤치, 1=활성

        # Self-attention 레이어 (팀 내 시너지)
        self.self_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.num_heads,
                dim_feedforward=cfg.d_model * 4,
                batch_first=True,
            ),
            num_layers=cfg.num_attn_layers,
        )

    def forward(
        self,
        team: dict[str, torch.Tensor],
        mask: torch.Tensor,  # (B, 6) 유효=1, 미공개/빈=0
    ) -> torch.Tensor:
        """
        Returns: (B, 6, d_model) — 각 포켓몬 슬롯의 attention 출력
        """
        pokemon_vecs = self.pokemon_encoder(**team)            # (B, 6, 149)
        tokens = self.proj(pokemon_vecs)                       # (B, 6, 128)

        # 역할 임베딩 추가 (활성 vs 벤치)
        role_ids = team["active"].squeeze(-1).long()           # (B, 6)
        tokens = tokens + self.role_emb(role_ids)

        # 미공개/빈 슬롯은 attention에서 제외
        src_key_padding_mask = mask == 0                       # True=무시

        tokens = self.self_attn(tokens, src_key_padding_mask=src_key_padding_mask)
        return tokens


class CrossTeamAttention(nn.Module):
    """
    내 팀 토큰이 상대 팀 토큰을 attend — 상성/위협 관계 학습.
    """

    def __init__(self, cfg: EncodingConfig):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 4),
            nn.ReLU(),
            nn.Linear(cfg.d_model * 4, cfg.d_model),
        )
        self.norm2 = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        my_tokens: torch.Tensor,    # (B, 6, d_model)
        opp_tokens: torch.Tensor,   # (B, 6, d_model)
        opp_mask: torch.Tensor,     # (B, 6) 유효=1
    ) -> torch.Tensor:
        """Returns: (B, 6, d_model)"""
        key_padding_mask = opp_mask == 0  # True=무시

        attn_out, _ = self.cross_attn(
            query=my_tokens,
            key=opp_tokens,
            value=opp_tokens,
            key_padding_mask=key_padding_mask,
        )
        x = self.norm(my_tokens + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class FieldEncoder(nn.Module):
    """배틀 필드 상태 인코딩"""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        weather: torch.Tensor,         # (B, 6) one-hot
        terrain: torch.Tensor,         # (B, 5) one-hot
        trick_room: torch.Tensor,      # (B, 2)
        my_side: torch.Tensor,         # (B, 8)
        opp_side: torch.Tensor,        # (B, 8)
        turn: torch.Tensor,            # (B, 1)
    ) -> torch.Tensor:
        """Returns: (B, 30)"""
        return torch.cat([weather, terrain, trick_room, my_side, opp_side, turn], dim=-1)


class InfosetEncoder(nn.Module):
    """
    전체 Infoset 인코딩 (Attention 기반):

    PokemonEncoder → TeamSelfAttn → CrossTeamAttn → 활성 토큰 추출 + 필드
    """

    def __init__(self, cfg: EncodingConfig):
        super().__init__()
        self.cfg = cfg
        self.my_team_encoder = TeamAttentionEncoder(cfg)
        self.opp_team_encoder = TeamAttentionEncoder(cfg)
        self.cross_attn = CrossTeamAttention(cfg)
        self.field_encoder = FieldEncoder()

    def forward(
        self,
        my_team: dict[str, torch.Tensor],
        my_mask: torch.Tensor,          # (B, 6)
        opp_team: dict[str, torch.Tensor],
        opp_mask: torch.Tensor,         # (B, 6)
        opp_unrevealed: torch.Tensor,   # (B, 1)
        field: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Returns: (B, infoset_dim)"""
        my_tokens = self.my_team_encoder(my_team, my_mask)      # (B, 6, 128)
        opp_tokens = self.opp_team_encoder(opp_team, opp_mask)  # (B, 6, 128)

        # 내 팀이 상대 팀을 attend
        my_tokens = self.cross_attn(my_tokens, opp_tokens, opp_mask)  # (B, 6, 128)

        # 활성 포켓몬 토큰 추출
        active_flags = my_team["active"].squeeze(-1)  # (B, 6)
        active_token = (my_tokens * active_flags.unsqueeze(-1)).sum(dim=1)  # (B, 128)

        field_vec = self.field_encoder(**field)  # (B, 30)

        return torch.cat([active_token, opp_unrevealed, field_vec], dim=-1)
