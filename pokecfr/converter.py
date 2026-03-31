"""poke-engine State → InfosetEncoder 텐서 변환.

poke-engine의 PyState/PySide/PyPokemon을 받아서
InfosetEncoder.forward()에 전달할 수 있는 dict 형태로 변환한다.
"""

import torch

from poke_engine import State as PEState

# ── ID 매핑 (문자열 → 정수 인덱스) ──
# poke-engine은 종/기술/아이템을 문자열로 관리.
# 학습 시 임베딩 룩업을 위해 정수 인덱스로 변환 필요.
# 첫 학습 전에 build_catalog()로 생성.

_species_to_id: dict[str, int] = {}
_move_to_id: dict[str, int] = {}
_item_to_id: dict[str, int] = {}
_type_to_id: dict[str, int] = {
    "normal": 1, "fire": 2, "water": 3, "electric": 4, "grass": 5,
    "ice": 6, "fighting": 7, "poison": 8, "ground": 9, "flying": 10,
    "psychic": 11, "bug": 12, "rock": 13, "ghost": 14, "dragon": 15,
    "dark": 16, "steel": 17, "fairy": 18, "typeless": 0,
}
_status_to_idx: dict[str, int] = {
    "none": 0, "poison": 1, "paralysis": 2, "burn": 3,
    "sleep": 4, "freeze": 5, "toxic": 6,
}


def register_id(name: str, catalog: dict[str, int]) -> int:
    """이름을 카탈로그에 등록하고 인덱스 반환. 0은 padding용 예약."""
    if name in catalog:
        return catalog[name]
    idx = len(catalog) + 1
    catalog[name] = idx
    return idx


def species_id(name: str) -> int:
    return register_id(name.lower(), _species_to_id)


def move_id(name: str) -> int:
    if name == "none":
        return 0
    return register_id(name.lower(), _move_to_id)


def item_id(name: str) -> int:
    if name == "none":
        return 0
    return register_id(name.lower(), _item_to_id)


def type_id(name: str) -> int:
    return _type_to_id.get(name.lower(), 0)


def _encode_pokemon(pkmn) -> dict[str, torch.Tensor]:
    """PyPokemon → 텐서 dict (배치 없이 단일 포켓몬)."""
    # 종
    sid = species_id(pkmn.id)

    # 타입
    types = [type_id(pkmn.types[0]), type_id(pkmn.types[1])]

    # 기술 (최대 4개)
    move_ids = []
    pp_pcts = []
    for m in pkmn.moves:
        move_ids.append(move_id(m.id))
        max_pp = max(m.pp, 1)  # 0 나누기 방지
        pp_pcts.append(m.pp / max_pp if m.id != "none" else 0.0)
    while len(move_ids) < 4:
        move_ids.append(0)
        pp_pcts.append(0.0)

    # HP
    hp_pct = pkmn.hp / max(pkmn.maxhp, 1)

    # 상태이상 one-hot
    status_vec = [0.0] * 7
    s_idx = _status_to_idx.get(pkmn.status.lower(), 0)
    status_vec[s_idx] = 1.0

    # 기절/활성은 Side 레벨에서 설정
    return {
        "species": torch.tensor(sid, dtype=torch.long),
        "types": torch.tensor(types, dtype=torch.long),
        "moves": torch.tensor(move_ids[:4], dtype=torch.long),
        "item": torch.tensor(item_id(pkmn.item), dtype=torch.long),
        "hp_pct": torch.tensor([hp_pct], dtype=torch.float32),
        "status": torch.tensor(status_vec, dtype=torch.float32),
        "stat_stages": torch.zeros(7, dtype=torch.float32),  # Side에서 채움
        "pp_pct": torch.tensor(pp_pcts[:4], dtype=torch.float32),
        "fainted": torch.tensor([1.0 if pkmn.hp <= 0 else 0.0], dtype=torch.float32),
        "active": torch.tensor([0.0], dtype=torch.float32),  # Side에서 설정
    }


def _encode_side(side, is_opponent: bool = False) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """PySide → (team_dict, mask).

    team_dict: 키별 (6, ...) 텐서
    mask: (6,) 유효 포켓몬 = 1
    """
    pokemon_dicts = []
    mask = []

    active_idx_str = side.active_index  # "p0"~"p5"
    active_idx = int(active_idx_str[1]) if len(active_idx_str) == 2 else 0

    for i, pkmn in enumerate(side.pokemon):
        pd = _encode_pokemon(pkmn)

        # 활성 포켓몬 마킹
        if i == active_idx:
            pd["active"] = torch.tensor([1.0], dtype=torch.float32)

        # 능력치 변화 (활성 포켓몬에만 적용)
        if i == active_idx:
            pd["stat_stages"] = torch.tensor([
                side.attack_boost / 6.0,
                side.defense_boost / 6.0,
                side.special_attack_boost / 6.0,
                side.special_defense_boost / 6.0,
                side.speed_boost / 6.0,
                side.accuracy_boost / 6.0,
                side.evasion_boost / 6.0,
            ], dtype=torch.float32)

        pokemon_dicts.append(pd)

        # 상대의 미공개 포켓몬: HP가 0이고 id가 기본값인 경우
        if is_opponent and pkmn.hp == 0 and pkmn.maxhp == 0:
            mask.append(0.0)
        else:
            mask.append(1.0)

    # dict 리스트 → 키별 stacked 텐서 (6, ...)
    team_dict = {}
    for key in pokemon_dicts[0]:
        team_dict[key] = torch.stack([pd[key] for pd in pokemon_dicts])

    return team_dict, torch.tensor(mask, dtype=torch.float32)


def _encode_field(state: PEState, my_side, opp_side) -> dict[str, torch.Tensor]:
    """배틀 필드 상태 인코딩."""
    # 날씨 one-hot (6종)
    weather_map = {"none": 0, "sun": 1, "rain": 2, "sand": 3, "snow": 4, "hail": 4, "harshsun": 1, "heavyrain": 2}
    weather_vec = [0.0] * 6
    w_idx = weather_map.get(state.weather.lower(), 0)
    weather_vec[w_idx] = 1.0

    # 필드 one-hot (5종)
    terrain_map = {"none": 0, "electric": 1, "grassy": 2, "misty": 3, "psychic": 4}
    terrain_vec = [0.0] * 5
    t_idx = terrain_map.get(state.terrain.lower(), 0)
    terrain_vec[t_idx] = 1.0

    # 트릭룸
    trick_room = [1.0 if state.trick_room else 0.0, state.trick_room_turns_remaining / 5.0]

    # 진영 효과 (8개씩)
    def side_conditions_vec(sc) -> list[float]:
        return [
            min(sc.stealth_rock, 1),
            sc.spikes / 3.0,
            sc.toxic_spikes / 2.0,
            min(sc.sticky_web, 1),
            min(sc.reflect, 1),
            min(sc.light_screen, 1),
            min(sc.tailwind, 1),
            min(sc.aurora_veil, 1),
        ]

    my_sc = side_conditions_vec(my_side.side_conditions)
    opp_sc = side_conditions_vec(opp_side.side_conditions)

    # 턴 수 (정규화)
    turn = [0.0]  # poke-engine State에 턴 카운터 없음, 외부에서 추적

    return {
        "weather": torch.tensor(weather_vec, dtype=torch.float32),
        "terrain": torch.tensor(terrain_vec, dtype=torch.float32),
        "trick_room": torch.tensor(trick_room, dtype=torch.float32),
        "my_side": torch.tensor(my_sc, dtype=torch.float32),
        "opp_side": torch.tensor(opp_sc, dtype=torch.float32),
        "turn": torch.tensor(turn, dtype=torch.float32),
    }


def state_to_infoset(state: PEState, player: int, turn: int = 0) -> dict:
    """poke-engine State → InfosetEncoder.forward() 입력 dict.

    Args:
        state: poke-engine PyState
        player: 관점 플레이어 (0=side_one, 1=side_two)
        turn: 현재 턴 수

    Returns:
        InfosetEncoder.forward()에 전달 가능한 dict
    """
    if player == 0:
        my_side, opp_side = state.side_one, state.side_two
    else:
        my_side, opp_side = state.side_two, state.side_one

    my_team, my_mask = _encode_side(my_side, is_opponent=False)
    opp_team, opp_mask = _encode_side(opp_side, is_opponent=True)

    # 상대 미공개 포켓몬 수
    unrevealed = (1.0 - opp_mask).sum().item() / 6.0

    field = _encode_field(state, my_side, opp_side)
    field["turn"] = torch.tensor([turn / 100.0], dtype=torch.float32)

    return {
        "my_team": my_team,
        "my_mask": my_mask,
        "opp_team": opp_team,
        "opp_mask": opp_mask,
        "opp_unrevealed": torch.tensor([unrevealed], dtype=torch.float32),
        "field": field,
    }
